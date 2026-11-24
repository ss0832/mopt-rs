//! GFN-FF non-bonded setup, ported from xtb's `gfnff_ini` (`src/gfnff/gfnff_ini.f90` lines for
//! the bonded ATM triples, the non-bonded repulsion exponents, the D4 `zeta` scaling, HB
//! basicities and acidities, and the HB / XB atom lists; `zeta` of the same file;
//! LGPL-3.0-or-later). xtb's quirk is kept: the amide-H acidity reduction of a neighbour is lost
//! when the neighbour comes later in the loop.
//!
//! In a periodic cell, as xtb does: the bonded ATM triples carry the cells of their outer
//! atoms (the cell of a neighbour of the other atom of the 1-4 pair is the sum of cells, which
//! may leave the 27 cells); the repulsion exponents of H...H pairs depend on the bond path to
//! the image (1-3 and 1-4 factors inside the 27 cells only); the XB acceptor is an image in
//! the 27 cells; the HB-modified bonds come from the (A, B) pairs over the setup's
//! translations, with H in the home cell, and a bond is matched to them by its stored cell
//! (`bond_hb_AHB_set`), which loses the A-H bonds that cross the boundary and are stored from
//! A.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use super::constants as k;
use super::elements::metal_of;
use super::lattice::{is_central, negate, Cell, HOME};
use super::neighbors::{dist, jth_nb, V3};
use super::topology::{amide_h, folded, xatom, Setup};

/// One bonded ATM triple: `i` in the home cell, `j` and `k` in their cells (`None` for a `k`
/// whose cell xtb's sum of translations does not reach).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Batm {
    pub i: usize,
    pub j: usize,
    pub k: usize,
    pub cj: Cell,
    pub ck: Option<Cell>,
}

/// Non-bonded lists and atomic parameters.
#[derive(Debug, Clone)]
pub struct NonBonded {
    /// Bonded ATM triples.
    pub b3list: Vec<Batm>,
    /// `sqrt(dum_i dum_j)` of the repulsion exponents for `j <= i` (row `i`, column `j`); the
    /// exponent of a pair and cell is this times [`NonBonded::pair_factor`].
    pub alphanb_root: Vec<Vec<f64>>,
    /// D4 `zeta` products `zetac6(i, j)` for `j <= i`.
    pub zetac6: Vec<Vec<f64>>,
    /// HB basicities and acidities per atom.
    pub hbbas: Vec<f64>,
    pub hbaci: Vec<f64>,
    /// H atoms of HBs.
    pub hb_h: Vec<usize>,
    /// For every atom, the indices into `hb_h` (ascending) of the H atoms one bond from it in
    /// one of the 27 cells (new: lets the HB loop visit only the H atoms that can form a term).
    pub hb_h_bonded: Vec<Vec<usize>>,
    /// A-B pairs of HBs (`i > j`).
    pub hb_ab: Vec<(usize, usize)>,
    /// XB triples `(A, B, X)` with the cells of B and X.
    pub xb_abx: Vec<(usize, usize, usize, Cell, Cell)>,
}

impl NonBonded {
    /// xtb's factor `ff` of the repulsion exponent of the pair `i`, `j` and the image of `i`
    /// in `cell` relative to `j`: `bp` is `bpair(i, j, cell)` inside the 27 cells (`None`
    /// beyond, the "far" exponent without the 1-3 and 1-4 factors).
    pub fn pair_factor(zi: u8, zj: u8, bp: Option<u8>) -> f64 {
        let mut ff = 1.0;
        if zi == 1 && zj == 1 {
            ff = k::HHFAC;
            if bp == Some(3) {
                ff *= k::HH14REP;
            }
            if bp == Some(2) {
                ff *= k::HH13REP;
            }
        }
        if (zi == 1 && metal_of(zj) > 0) || (zj == 1 && metal_of(zi) > 0) {
            ff = 0.85;
        }
        if (zi == 1 && zj == 6) || (zj == 1 && zi == 6) {
            ff = 0.91;
        }
        if (zi == 1 && zj == 8) || (zj == 1 && zi == 8) {
            ff = 1.04;
        }
        ff
    }

    /// The repulsion exponent of `i`, `j <= i` and the image cell (`bp` as for
    /// [`Self::pair_factor`]).
    pub fn alphanb(&self, s: &Setup, i: usize, j: usize, bp: Option<u8>) -> f64 {
        self.alphanb_root[i][j] * Self::pair_factor(s.el[i].z(), s.el[j].z(), bp)
    }
}

/// xtb's `zeta(at, q)`: the D4 charge scaling with `q_ref = 0`, with the effective nuclear
/// charge and chemical hardness of GFN-FF's own tables (`zeff`, `c` of `zeta` in
/// `gfnff_ini.f90`); 1 for an element without data.
pub fn zeta(z: u8, q: f64) -> f64 {
    let Some([zeff, c]) = super::elements::raw_of(z).map(|e| e.zeta) else {
        return 1.0;
    };
    let qmod = zeff + q;
    if qmod < 0.0 {
        3.0_f64.exp()
    } else {
        (3.0 * (1.0 - (c * (1.0 - zeff / qmod)).exp())).exp()
    }
}

/// The nearest neighbour (`jth_nb(1)`, over the images), if any.
fn nearest(s: &Setup, x: &[V3], i: usize) -> Option<usize> {
    jth_nb(&s.nb.nb[i], &s.frame(x), i, 1).map(|(j, _)| j)
}

/// The non-bonded setup.
pub fn nonbonded(s: &Setup, x: &[V3]) -> NonBonded {
    let n = x.len();
    let z = |a: usize| s.el[a].z();
    let nbn = |a: usize| s.nb.nb[a].len();
    // Bonded ATM triples: for a 1-4 pair (i, image of j), the neighbours of j (cells summed)
    // and of i, in xtb's loops (cell of j, cell of the neighbour, neighbour).
    let cells = s.frame(x).bond_cells();
    let mut b3list = Vec::new();
    for i in 0..n {
        for j in 0..i {
            for &cj in &cells {
                if s.bpair.get(j, i, cj) != Some(3) {
                    continue;
                }
                for &c2 in &cells {
                    for &(kk, _) in s.nb.nb[j].iter().filter(|&&(_, c)| c == c2) {
                        if (kk == i || kk == j) && cj == HOME && c2 == HOME {
                            continue;
                        }
                        b3list.push(Batm {
                            i,
                            j,
                            k: kk,
                            cj,
                            ck: s.translations.sum(cj, c2),
                        });
                    }
                    for &(kk, _) in s.nb.nb[i].iter().filter(|&&(_, c)| c == c2) {
                        if (kk == i || kk == j) && cj == HOME && c2 == HOME {
                            continue;
                        }
                        b3list.push(Batm {
                            i,
                            j,
                            k: kk,
                            cj,
                            ck: Some(c2),
                        });
                    }
                }
            }
        }
    }
    // Repulsion exponents (without the factor of the pair) and zeta products.
    let mut alphanb_root = vec![Vec::new(); n];
    let mut zetac6 = vec![Vec::new(); n];
    for i in 0..n {
        let ati = z(i);
        let fni = 1.0 + k::NREPSCAL / (1.0 + (nbn(i) * nbn(i)) as f64);
        let dum1 = s.el[i].raw.repan * (1.0 + s.qa[i] * k::QREPSCAL) * fni;
        let f1 = zeta(ati, s.qa[i]);
        for j in 0..=i {
            let atj = z(j);
            let fnj = 1.0 + k::NREPSCAL / (1.0 + (nbn(j) * nbn(j)) as f64);
            let dum2 = s.el[j].raw.repan * (1.0 + s.qa[j] * k::QREPSCAL) * fnj;
            let f2 = zeta(atj, s.qa[j]);
            alphanb_root[i].push((dum1 * dum2).sqrt());
            zetac6[i].push(f1 * f2);
        }
    }
    // HB basicity.
    let mut hbbas = vec![1.0; n];
    for i in 0..n {
        let nn = nbn(i);
        let ati = z(i);
        hbbas[i] = k::xhbas(ati);
        if ati == 6 && nn == 2 && s.nb.itag[i] == 1 {
            hbbas[i] = 1.46;
        }
        if ati == 8 && nn == 1 {
            let other = z(s.nb.nb[i][0].0);
            if other == 6 {
                hbbas[i] = 0.68;
            }
            if other == 7 {
                hbbas[i] = 0.47;
            }
        }
    }
    // HB acidity (with xtb's loop order).
    let nb_atoms = folded(&s.nb.nb);
    let mut hbaci: Vec<f64> = (0..n).map(|i| k::xhaci(z(i))).collect();
    for i in 0..n {
        let Some(nn) = nearest(s, x, i) else { continue };
        hbaci[i] = k::xhaci(z(i));
        if amide_h(&s.el, &s.nb.hyb, &nb_atoms, &s.piadr2, i) {
            hbaci[nn] *= 0.80;
        }
    }
    // H atoms of HBs.
    let mut hb_h = Vec::new();
    for i in 0..n {
        if z(i) != 1 || s.nb.hyb[i] == 1 {
            continue;
        }
        let mut ff = k::HQABTHR;
        let Some(j) = nearest(s, x, i) else { continue };
        if z(j) > 10 {
            ff -= 0.20;
        }
        if z(j) == 6 && s.nb.hyb[j] == 3 {
            ff += 0.05;
        }
        if s.qa[i] > ff {
            hb_h.push(i);
        }
    }
    // A-B pairs of HBs.
    let mut hb_ab = Vec::new();
    for i in 0..n {
        if z(i) == 6 && s.piadr2[i] == 0 {
            continue;
        }
        let mut ff = k::QABTHR;
        if z(i) > 10 {
            ff += 0.2;
        }
        if s.qa[i] > ff {
            continue;
        }
        for j in 0..i {
            let mut ff = k::QABTHR;
            if z(j) > 10 {
                ff += 0.2;
            }
            if s.qa[j] > ff {
                continue;
            }
            if hbbas[i] * hbaci[j] < 1.0e-6 && hbaci[i] * hbbas[j] < 1.0e-6 {
                continue;
            }
            if z(j) == 6 && s.piadr2[j] == 0 {
                continue;
            }
            hb_ab.push((i, j));
        }
    }
    // XB triples A-X...B: X a neighbour of A in its cell, B an image in the 27 cells.
    let mut xb_abx = Vec::new();
    for i in 0..n {
        for &(ix, cx) in &s.nb.nb[i] {
            if !xatom(z(ix)) {
                continue;
            }
            if z(ix) == 16 && nbn(ix) > 2 {
                continue;
            }
            for &cb in &cells {
                for j in 0..n {
                    if (i == j && cb == HOME) || (j == ix && cb == cx) {
                        continue;
                    }
                    // The image of X relative to the image of B.
                    let Some(dum) = s.translations.sum(negate(cb), cx) else {
                        continue;
                    };
                    if !is_central(dum) {
                        continue;
                    }
                    if s.bpair.get(ix, j, dum).is_none_or(|v| v <= 3) {
                        continue;
                    }
                    if k::xhbas(z(j)) < 1.0e-6 {
                        continue;
                    }
                    if s.el[j].raw.group == 4 && (s.piadr2[j] == 0 || s.qa[j] > 0.05) {
                        continue;
                    }
                    xb_abx.push((i, j, ix, cb, cx));
                }
            }
        }
    }
    // The HB hydrogens bonded to each atom (at any cell of the pairs), for the HB loop.
    let mut hb_h_bonded = vec![Vec::new(); n];
    for (kh, &h) in hb_h.iter().enumerate() {
        for (a, list) in hb_h_bonded.iter_mut().enumerate() {
            let bonded = s
                .bpair
                .cells()
                .iter()
                .any(|&c| s.bpair.get(a, h, c) == Some(1));
            if bonded {
                list.push(kh);
            }
        }
    }
    NonBonded {
        b3list,
        alphanb_root,
        zetac6,
        hbbas,
        hbaci,
        hb_h,
        hb_h_bonded,
        hb_ab,
        xb_abx,
    }
}

/// One A-H bond with its acceptors: A in cell `ca` relative to H (in the home cell), and the
/// acceptors B with their cells.
#[derive(Debug, Clone, PartialEq)]
pub struct AhBond {
    pub a: usize,
    pub h: usize,
    pub ca: Cell,
    pub b: Vec<(usize, Cell)>,
}

/// The HB-modified bonds (xtb's `bond_hbset`, `bond_hb_AHB_set1`, `bond_hb_AHB_set`, and the
/// `hb_mapABH` mapping of `gfnff_ini`).
#[derive(Debug, Clone, Default)]
pub struct BondHb {
    /// A-H bonds with their acceptors.
    pub ah: Vec<AhBond>,
    /// `nr_hb` per bond (xtb keeps the previous value for bonds without acceptors).
    pub nr_hb: Vec<usize>,
    /// Atoms taking part (`isABH`).
    pub is_abh: Vec<bool>,
    /// `hb_mapABH`: running index (1-based) among the H atoms or among the A/B atoms.
    pub map_abh: Vec<usize>,
}

/// The HB-modified bond lists.
pub fn bond_hb(s: &Setup, x: &[V3], nbl: &NonBonded) -> BondHb {
    let n = x.len();
    let f = s.frame(x);
    let z = |a: usize| s.el[a].z();
    // bond_hbset: (A, B, H, cell of A, cell of B), H in the home cell, over the translations
    // of the setup.
    let mut hbl: Vec<(usize, usize, usize, Cell, Cell)> = Vec::new();
    let tr = &s.translations.cells;
    for &(i, j) in &nbl.hb_ab {
        for &ci in tr {
            for &cj in tr {
                let rab = dist(f.at(i, ci), f.at(j, cj)).powi(2);
                if rab > k::HBTHR1 {
                    continue;
                }
                let nonbond = match s.translations.sum(negate(ci), cj) {
                    Some(dum) if is_central(dum) => s.bpair.get(j, i, dum) != Some(1),
                    _ => true,
                };
                for &nh in &nbl.hb_h {
                    if is_central(ci) && s.bpair.get(i, nh, ci) == Some(1) && nonbond {
                        hbl.push((i, j, nh, ci, cj));
                    }
                    if is_central(cj) && s.bpair.get(j, nh, cj) == Some(1) && nonbond {
                        hbl.push((j, i, nh, cj, ci));
                    }
                }
            }
        }
    }
    let mut out = BondHb {
        nr_hb: vec![0; s.bonds.len()],
        is_abh: vec![false; n],
        map_abh: vec![0; n],
        ..BondHb::default()
    };
    let mut b_count = 1usize;
    let mut prev: Option<(usize, usize, Cell)> = None;
    for (bi, &(jj, ii, cb)) in s.bonds.iter().enumerate() {
        let (h, a) = if z(ii) == 1 {
            (ii, jj)
        } else if z(jj) == 1 {
            (jj, ii)
        } else {
            continue;
        };
        if z(a) != 7 && z(a) != 8 {
            continue;
        }
        for &(aa, bb, hh, ca, cbb) in &hbl {
            // xtb compares the stored cell of the bond with the cell of A (H is always in the
            // home cell).
            if aa == a && hh == h && cb == ca && (z(bb) == 7 || z(bb) == 8) {
                out.is_abh[bb] = true;
                if prev == Some((a, h, ca)) {
                    b_count += 1;
                    if let Some(last) = out.ah.last_mut() {
                        last.b.push((bb, cbb));
                    }
                } else {
                    b_count = 1;
                    out.ah.push(AhBond {
                        a,
                        h,
                        ca,
                        b: vec![(bb, cbb)],
                    });
                }
                prev = Some((a, h, ca));
            }
        }
        out.is_abh[a] = true;
        out.is_abh[h] = true;
        out.nr_hb[bi] = b_count;
    }
    let (mut nh, mut nab) = (0, 0);
    for i in 0..n {
        if out.is_abh[i] {
            if z(i) == 1 {
                nh += 1;
                out.map_abh[i] = nh;
            } else {
                nab += 1;
                out.map_abh[i] = nab;
            }
        }
    }
    out
}
