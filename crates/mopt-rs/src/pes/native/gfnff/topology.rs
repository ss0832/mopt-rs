//! GFN-FF topology, ported from xtb (`gfnff_ini` in `src/gfnff/gfnff_ini.f90`, with
//! `goedeckera`, `qheavy`, `nbondmat_pbc`, `mrecgffPBC`, `amide`, `amideH` of `gfnff_ini2.F90`
//! / `mrec.f90`; LGPL-3.0-or-later): bonds, pi atoms, topology charges from a neighbour-only
//! EEQ, the atomic EEQ parameters, rings, and the bond distance matrix. xtb's quirks are kept
//! (e.g. `piadr` where `piadr2` is meant in the `dgam` loop).
//!
//! In a periodic cell, as xtb 6.7.1 does: a bond is `(j, i, n)`, the image of `j` in cell `n`
//! bonded to `i`, stored once from the atom with the lower index; the bond paths
//! (`nbondmat_pbc`) are counted to the images in the 27 cells around the home cell only; the
//! fragments, the topology charges, and the rings are those of the folded graph (bonds through
//! any cell join atom indices); counts of neighbours are summed over the cells.
//!
//! Not reproduced: reading a `charges` file or PDB residue charges (the fragment charges are
//! the total charge on the first fragment, as xtb does for plain structure files).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use super::constants as k;
use super::elements::{Atom, AUTOAA};
use super::lattice::{is_central, negate, Cell, Frame, Lattice, Nb, Translations, HOME};
use super::neighbors::{dist, log_cn, neighbors, Neighbors, V3};
use super::rings::{getring36, AtomRings};
use crate::error::{MoptError, MoptResult};

/// `bpair`: the number of bonds between `x_i` and the image `x_j + L n` (1, 2, 3, or 5 when
/// farther), for the cells `n` of the 27 around the home cell (the home cell alone for a
/// molecule); 0 for an atom and itself, and for an atom and an image of itself that no bond
/// path of up to three bonds reaches.
#[derive(Debug, Clone)]
pub struct BondPairs {
    n: usize,
    cells: Vec<Cell>,
    data: Vec<u8>,
}

impl BondPairs {
    fn zeros(n: usize, cells: Vec<Cell>) -> Self {
        let size = n * n * cells.len();
        Self {
            n,
            cells,
            data: vec![0; size],
        }
    }

    fn slot(&self, j: usize, i: usize, cell: Cell) -> Option<usize> {
        let c = if self.cells.len() == 1 {
            (cell == HOME).then_some(0)?
        } else {
            if !is_central(cell) {
                return None;
            }
            // xtb's order of the 27 cells: z slowest, x fastest, the home cell first.
            let k = (cell[2] + 1) * 9 + (cell[1] + 1) * 3 + (cell[0] + 1);
            let k = k as usize;
            match k {
                13 => 0,
                k if k < 13 => k + 1,
                k => k,
            }
        };
        Some((c * self.n + i) * self.n + j)
    }

    /// `bpair(j, i, n)`; `None` for a cell outside the 27 (xtb's lists do not hold those).
    pub fn get(&self, j: usize, i: usize, cell: Cell) -> Option<u8> {
        self.slot(j, i, cell).map(|s| self.data[s])
    }

    /// The cells the pairs are held for.
    pub fn cells(&self) -> &[Cell] {
        &self.cells
    }

    /// `bpair(j, i, 0)` of the home cell.
    pub fn home(&self, j: usize, i: usize) -> u8 {
        self.data[i * self.n + j]
    }

    fn set(&mut self, j: usize, i: usize, cell: Cell, value: u8) {
        if let Some(s) = self.slot(j, i, cell) {
            self.data[s] = value;
        }
    }
}

/// xtb's `imetal` of an atom with `neighbours` bonded atoms: the `metal` of its element,
/// except that Sn, Pb, and Bi (metals beyond group 3) with at most four neighbours are taken
/// as non-metals.
pub fn imetal_of(a: &Atom, neighbours: usize) -> i32 {
    if neighbours <= 4 && a.raw.group > 3 {
        0
    } else {
        a.metal()
    }
}

/// `pilist`: atoms that can be pi atoms (B, C, N, O, F, S, Cl).
pub fn pilist(z: u8) -> bool {
    matches!(z, 5 | 6 | 7 | 8 | 9 | 16 | 17)
}

/// `nofs`: N, O, F, S, Cl.
pub fn nofs(z: u8) -> bool {
    matches!(z, 7 | 8 | 9 | 16 | 17)
}

/// `xatom`: halogen-bond donors X in A-X...B.
pub fn xatom(z: u8) -> bool {
    matches!(z, 17 | 35 | 53 | 16 | 34 | 52 | 15 | 33 | 51)
}

/// The setup-time topology.
#[derive(Debug, Clone)]
pub struct Setup {
    /// Element data.
    pub el: Vec<Atom>,
    /// Neighbours and hybridization.
    pub nb: Neighbors,
    /// Bonds `(j, i, n)` in xtb's order: the image of `j` in cell `n` bonded to `i`, from the
    /// atom with the lower index (for a molecule `j > i` and `n` the home cell).
    pub bonds: Vec<(usize, usize, Cell)>,
    /// Pi atoms (`piadr`): atom of each pi index.
    pub piadr: Vec<usize>,
    /// Pi index + 1 of each atom (`piadr2`; 0 when not a pi atom).
    pub piadr2: Vec<usize>,
    /// Pi fragment (1-based) of each pi atom (`pimvec`) and their number (`picount`).
    pub pimvec: Vec<usize>,
    pub picount: usize,
    /// Pi electrons removed by fragment charges, per pi fragment (`ipis`).
    pub ipis: Vec<i64>,
    /// Topology charges `qa`.
    pub qa: Vec<f64>,
    /// The charges of the last neighbour search (the first pass's `qa`).
    pub qa_neighbors: Vec<f64>,
    /// The metallic character of the setup's neighbour search.
    pub mchar: Vec<f64>,
    /// EEQ parameters (atomic): `chi`, `gam`, `alp` (squared widths).
    pub chieeq: Vec<f64>,
    pub gameeq: Vec<f64>,
    pub alpeeq: Vec<f64>,
    /// Fragments (1-based) and fragment charges.
    pub fraglist: Vec<usize>,
    pub qfrag: Vec<f64>,
    /// Number of bonds between atoms and images (1, 2, 3, or 5 when farther).
    pub bpair: BondPairs,
    /// Rings of every atom.
    pub rings: Vec<AtomRings>,
    /// EEQ electronegativity corrections `dxi`.
    pub dxi: Vec<f64>,
    /// xtb's `imetal` of every atom ([`imetal_of`]).
    pub imetal: Vec<i32>,
    /// The lattice of a periodic cell.
    pub lattice: Option<Lattice>,
    /// The translations of the setup (cutoff `sqrt(hbthr2)`; the home cell for a molecule).
    pub translations: Translations,
}

impl Setup {
    /// The frame of the positions `x` in the cell of the setup.
    pub fn frame<'a>(&'a self, x: &'a [V3]) -> Frame<'a> {
        Frame {
            x,
            lattice: self.lattice.as_ref(),
        }
    }
}

/// The atoms of the neighbour lists, without the cells (the folded graph).
pub fn folded(nb: &[Vec<Nb>]) -> Vec<Vec<usize>> {
    nb.iter()
        .map(|l| l.iter().map(|&(j, _)| j).collect())
        .collect()
}

/// `mrecgffPBC`: connected components numbered by their lowest atom (1-based).
pub fn components(nb: &[Vec<usize>]) -> (usize, Vec<usize>) {
    let n = nb.len();
    let mut id = vec![0usize; n];
    let mut count = 0;
    for start in 0..n {
        if id[start] != 0 {
            continue;
        }
        count += 1;
        let mut stack = vec![start];
        id[start] = count;
        while let Some(a) = stack.pop() {
            for &b in &nb[a] {
                if id[b] == 0 {
                    id[b] = count;
                    stack.push(b);
                }
            }
        }
    }
    (count, id)
}

/// The neighbour lists without the bonds that only one of the two atoms lists (a metal and
/// the eta-coordinated atoms, whose lists leave the metal out), as xtb's `nbondmat_pbc` makes
/// them, and the bonds it took out. xtb takes a neighbour out of the list of its cell by moving
/// the entries of the original list after it one place forward and counting one entry less,
/// so with several such neighbours in one list the first is taken out and the list is cut at
/// its end; that is followed here, since the bond paths and the energies depend on it.
fn paired_lists(nb: &[Vec<Nb>]) -> (Vec<Vec<Nb>>, Vec<(usize, usize, Cell)>) {
    let n = nb.len();
    let mut reduced = Vec::with_capacity(n);
    let mut removed = Vec::new();
    for i in 0..n {
        // The cells of the list in their order, each with its entries in order.
        let mut cells: Vec<Cell> = Vec::new();
        for &(_, c) in &nb[i] {
            if !cells.contains(&c) {
                cells.push(c);
            }
        }
        let mut out = Vec::new();
        for c in cells {
            let list: Vec<usize> = nb[i]
                .iter()
                .filter(|(_, cc)| *cc == c)
                .map(|&(j, _)| j)
                .collect();
            let mut kept = list.clone();
            let mut count = list.len();
            for (pos, &k) in list.iter().enumerate() {
                if nb[k].iter().any(|&(l, _)| l == i) {
                    continue;
                }
                // Shift the original entries after `pos` forward.
                for l in pos..kept.len() {
                    kept[l] = list.get(l + 1).copied().unwrap_or(usize::MAX);
                }
                count -= 1;
                removed.push((k, i, c));
            }
            out.extend(
                kept.into_iter()
                    .take(count)
                    .filter(|&j| j != usize::MAX)
                    .map(|j| (j, c)),
            );
        }
        reduced.push(out);
    }
    (reduced, removed)
}

/// `nbondmat_pbc`: the shortest bond path up to 3 between each atom and the images in the
/// cells of `cells` (27 around the home cell, or the home cell), otherwise 5. A path through a
/// neighbour in another cell continues with the sums of cells of `tr` (xtb's `fTrSum`); sums
/// outside the 27 cells are not followed. The paths go over the bonds both atoms list
/// ([`paired_lists`]); a bond only one of them lists (a metal and its eta-coordinated atoms)
/// is set to 1 both ways at the end, as xtb does.
pub fn bond_pairs(nb_full: &[Vec<Nb>], cells: Vec<Cell>, tr: &Translations) -> BondPairs {
    let (reduced, removed) = paired_lists(nb_full);
    let nb = &reduced[..];
    let n = nb.len();
    let mut pair = BondPairs::zeros(n, cells.clone());
    for i in 0..n {
        for &(j, c) in &nb[i] {
            pair.set(j, i, c, 1);
        }
        for cval in 1..=2u8 {
            // countf: the cval-th neighbours of i.
            let mut current = Vec::new();
            for &c in &cells {
                for l in 0..n {
                    if pair.get(l, i, c) == Some(cval) {
                        current.push((l, c));
                    }
                }
            }
            for (inew, cnew) in current {
                if cnew == HOME {
                    for &(j, c) in &nb[inew] {
                        if pair.get(j, i, c) != Some(0) || (j == i && c == HOME) {
                            continue;
                        }
                        pair.set(j, i, c, cval + 1);
                    }
                } else {
                    for &(j, _) in nb[inew].iter().filter(|&&(_, c)| c == HOME) {
                        if pair.get(j, i, cnew) != Some(0) {
                            continue;
                        }
                        pair.set(j, i, cnew, cval + 1);
                    }
                    for &(j, c) in nb[inew].iter().filter(|(_, c)| *c != HOME) {
                        let Some(sum) = tr.sum(c, cnew) else { continue };
                        if (sum == HOME && j == i) || !is_central(sum) {
                            continue;
                        }
                        if pair.get(j, i, sum) != Some(0) {
                            continue;
                        }
                        pair.set(j, i, sum, cval + 1);
                    }
                }
            }
        }
    }
    for i in 0..n {
        for j in 0..n {
            for &c in &cells {
                if pair.get(j, i, c) == Some(0) && j != i {
                    pair.set(j, i, c, 5);
                }
            }
        }
    }
    for (k, i, c) in removed {
        pair.set(k, i, c, 1);
        pair.set(i, k, negate(c), 1);
    }
    pair
}

/// The bond `(j, i, n)` as `(a, b, m)` with `a <= b` (the same bond seen from the other atom
/// is `(i, j, -n)`), and of an atom with its own image the cell that comes first.
pub fn bond_key(j: usize, i: usize, n: Cell) -> (usize, usize, Cell) {
    match i.cmp(&j) {
        std::cmp::Ordering::Less => (i, j, n),
        std::cmp::Ordering::Greater => (j, i, negate(n)),
        std::cmp::Ordering::Equal => (i, i, n.max(negate(n))),
    }
}

/// `qheavy`: H charges condensed onto their neighbours (atoms of the folded graph).
fn qheavy(el: &[Atom], nb: &[Vec<usize>], q: &[f64]) -> Vec<f64> {
    let mut out = q.to_vec();
    for i in 0..q.len() {
        if el[i].z() != 1 {
            continue;
        }
        out[i] = 0.0;
        let nn = nb[i].len() as f64;
        for &kk in &nb[i] {
            out[kk] += q[i] / nn;
        }
    }
    out
}

/// `amide(..., pi, a)` with the given pi array (`pi[a] != 0` means a pi atom).
pub fn amide(el: &[Atom], hyb: &[i32], nb: &[Vec<usize>], pi: &[usize], a: usize) -> bool {
    if pi[a] == 0 || hyb[a] != 3 || el[a].z() != 7 {
        return false;
    }
    let mut nc = 0;
    let mut ic = 0;
    for &j in &nb[a] {
        if el[j].z() == 6 && pi[j] != 0 {
            nc += 1;
            ic = j;
        }
    }
    let mut no = 0;
    if nc == 1 {
        for &j in &nb[ic] {
            if el[j].z() == 8 && pi[j] != 0 && nb[j].len() == 1 {
                no += 1;
            }
        }
    }
    no == 1
}

/// `amideH`: an H on an amide N that carries one sp3 C.
pub fn amide_h(el: &[Atom], hyb: &[i32], nb: &[Vec<usize>], pi: &[usize], a: usize) -> bool {
    if nb[a].len() != 1 {
        return false;
    }
    let nn = nb[a][0];
    if !amide(el, hyb, nb, pi, nn) {
        return false;
    }
    nb[nn]
        .iter()
        .filter(|&&j| el[j].z() == 6 && hyb[j] == 3)
        .count()
        == 1
}

/// The matrix of xtb's `goedeckera` (EEQ with the topology distances `pair`, Bohr, and the
/// fragment constraints), factorized once: the charges of the pi fragments (`ipis`) solve it
/// again with other fragment charges only. xtb factorizes it for every right-hand side; the
/// factors are a function of the matrix alone and every right-hand side is solved alone, so
/// the charges are the same bits (a test compares them with xtb's way, [`goedeckera`]).
struct Goedecker {
    lu: crate::math::linalg::DenseLu,
    n: usize,
    nfrag: usize,
}

impl Goedecker {
    fn new(
        gam: &[f64],
        alp: &[f64],
        pair: &[Vec<f64>],
        fraglist: &[usize],
        nfrag: usize,
    ) -> MoptResult<Self> {
        const TSQRT2PI: f64 = 0.797_884_560_802_866;
        let n = gam.len();
        let m = n + nfrag;
        let mut a = Mat::<f64>::zeros(m, m);
        for i in 0..n {
            a[(i, i)] = gam[i] + TSQRT2PI / alp[i].sqrt();
        }
        for i in 0..n {
            for j in 0..i {
                let rij = pair[i][j];
                let gammij = 1.0 / (alp[i] + alp[j]).sqrt();
                let tmp = libm::erf(gammij * rij) / rij;
                a[(j, i)] = tmp;
                a[(i, j)] = tmp;
            }
        }
        for f in 0..nfrag {
            for j in 0..n {
                if fraglist[j] == f + 1 {
                    a[(n + f, j)] = 1.0;
                    a[(j, n + f)] = 1.0;
                }
            }
        }
        Ok(Self {
            lu: crate::math::linalg::DenseLu::new(a.as_ref())?,
            n,
            nfrag,
        })
    }

    /// `goedeckera`'s charges for the electronegativities `chi` and the fragment charges
    /// `qfrag`.
    fn charges(&self, chi: &[f64], qfrag: &[f64]) -> MoptResult<Vec<f64>> {
        let n = self.n;
        let x = Col::<f64>::from_fn(
            n + self.nfrag,
            |i| {
                if i < n {
                    chi[i]
                } else {
                    qfrag[i - n]
                }
            },
        );
        let sol = self.lu.solve(x.as_mat());
        let mut q: Vec<f64> = (0..n).map(|i| sol[(i, 0)]).collect();
        if q.iter().any(|v| !v.is_finite()) {
            return Err(MoptError::calc(
                "GFN-FF topology charges: singular EEQ system",
            ));
        }
        if n == 1 {
            q[0] = qfrag[0];
        }
        Ok(q)
    }
}

/// The parts of the setup whose algorithm is not xtb's but whose result is (a test runs xtb's
/// algorithms as the reference and compares the whole setup).
#[derive(Clone, Copy)]
struct SetupKernels {
    /// The topology distances.
    distances: fn(&[Atom], &[Vec<usize>]) -> Vec<Vec<f64>>,
    /// xtb's EEQ: the matrix made and factorized for every right-hand side ([`goedeckera`]).
    #[cfg(test)]
    reference_eeq: bool,
}

/// The kernels of the setup.
const FAST: SetupKernels = SetupKernels {
    distances: topology_distances,
    #[cfg(test)]
    reference_eeq: false,
};

/// The charges of `goedeckera` from the factorized `eeq`, or (the reference of the tests) as
/// xtb computes them.
#[allow(clippy::too_many_arguments)]
fn eeq_charges(
    kernels: SetupKernels,
    eeq: &Goedecker,
    p: &Pass,
    pair: &[Vec<f64>],
    fraglist: &[usize],
    qfrag: &[f64],
    nfrag: usize,
) -> MoptResult<Vec<f64>> {
    #[cfg(test)]
    if kernels.reference_eeq {
        return goedeckera(
            &p.chieeq, &p.gameeq, &p.alpeeq, pair, fraglist, qfrag, nfrag,
        );
    }
    let _ = (kernels, pair, fraglist, nfrag);
    eeq.charges(&p.chieeq, qfrag)
}

/// `goedeckera`: EEQ with the topology distances `pair` (Bohr) and fragment constraints.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn goedeckera(
    chi: &[f64],
    gam: &[f64],
    alp: &[f64],
    pair: &[Vec<f64>],
    fraglist: &[usize],
    qfrag: &[f64],
    nfrag: usize,
) -> MoptResult<Vec<f64>> {
    const TSQRT2PI: f64 = 0.797_884_560_802_866;
    let n = chi.len();
    let m = n + nfrag;
    let mut a = Mat::<f64>::zeros(m, m);
    let mut x = Col::<f64>::zeros(m);
    for i in 0..n {
        x[i] = chi[i];
        a[(i, i)] = gam[i] + TSQRT2PI / alp[i].sqrt();
    }
    for i in 0..n {
        for j in 0..i {
            let rij = pair[i][j];
            let gammij = 1.0 / (alp[i] + alp[j]).sqrt();
            let tmp = libm::erf(gammij * rij) / rij;
            a[(j, i)] = tmp;
            a[(i, j)] = tmp;
        }
    }
    for f in 0..nfrag {
        x[n + f] = qfrag[f];
        for j in 0..n {
            if fraglist[j] == f + 1 {
                a[(n + f, j)] = 1.0;
                a[(j, n + f)] = 1.0;
            }
        }
    }
    let sol = crate::math::linalg::DenseLu::new(a.as_ref())?.solve(x.as_mat());
    let mut q: Vec<f64> = (0..n).map(|i| sol[(i, 0)]).collect();
    if q.iter().any(|v| !v.is_finite()) {
        return Err(MoptError::calc(
            "GFN-FF topology charges: singular EEQ system",
        ));
    }
    if n == 1 {
        q[0] = qfrag[0];
    }
    Ok(q)
}

/// One pass of the topology loop (`qloop_count`), returning what the next pass and the rest of
/// the setup need.
struct Pass {
    nb: Neighbors,
    bonds: Vec<(usize, usize, Cell)>,
    piadr: Vec<usize>,
    piadr2: Vec<usize>,
    pimvec: Vec<usize>,
    picount: usize,
    dxi: Vec<f64>,
    chieeq: Vec<f64>,
    gameeq: Vec<f64>,
    alpeeq: Vec<f64>,
}

/// The bonds xtb's neighbour search finds at `f` with the topology charges `qa` (as
/// [`bond_key`]s, sorted), for detecting a changed bonding pattern. A molecule takes the
/// metallic character at `f`, as a setup there would. In a periodic cell the metallic character
/// of the setup is reused (`setup`), since its lattice sum would cost more than the energy; it
/// is an approximation there: the metallic character enters the neighbour search only through
/// the list without metals (a non-metal atom above 0.25 is left out of it like the metals),
/// which decides the eta coordination of light atoms to metals, and its factor
/// `exp(-0.005 EN^8)` keeps it small except for atoms of low electronegativity (0.28 at
/// EN = 2.0, 0.004 at 2.4, 1e-4 for C), such as B, Si, Ge.
pub fn bonds_at(
    el: &[Atom],
    f: &Frame<'_>,
    qa: &[f64],
    setup: Option<&[f64]>,
) -> MoptResult<Vec<(usize, usize, Cell)>> {
    Ok(bonds_with_margin(el, f, qa, setup)?.0)
}

/// [`bonds_at`] with its margin: the smallest `|r - r_thr|` (Bohr) over every pair of atoms
/// and image, `r_thr` the bond-detection distance of the pair, so that no bond can form or
/// break while every atom moves by less than half of it. Without metal atoms the bonds are
/// those of the full neighbour list (the list without metals replaces it only for atoms
/// eta-coordinated to a metal), which needs neither the metallic character nor the other
/// lists; with metals the margin is `-inf` (no bound: the metallic character, a function of
/// the positions, decides the eta coordination). New (xtb has no bonding check).
pub fn bonds_with_margin(
    el: &[Atom],
    f: &Frame<'_>,
    qa: &[f64],
    setup: Option<&[f64]>,
) -> MoptResult<(Vec<(usize, usize, Cell)>, f64)> {
    if el.iter().all(|a| a.metal() == 0) {
        return Ok(nonmetal_bonds(el, f, qa));
    }
    let mchar = match (f.lattice, setup) {
        (Some(_), Some(m)) => m.to_vec(),
        _ => setup_mchar(el, f)?,
    };
    let nb = neighbors(el, f, qa, &mchar);
    let mut out: Vec<(usize, usize, Cell)> = (0..f.len())
        .flat_map(|i| nb.nb[i].iter().map(move |&(j, c)| bond_key(j, i, c)))
        .collect();
    out.sort_unstable();
    out.dedup();
    Ok((out, f64::NEG_INFINITY))
}

/// The bonds of a structure without metal atoms (the full neighbour list of
/// [`super::neighbors::neighbors`], every radius as there) as sorted [`bond_key`]s, with the
/// margin of [`bonds_with_margin`].
fn nonmetal_bonds(el: &[Atom], f: &Frame<'_>, qa: &[f64]) -> (Vec<(usize, usize, Cell)>, f64) {
    let n = f.len();
    let mut out = Vec::new();
    let mut margin = f64::INFINITY;
    for cell in f.bond_cells() {
        for i in 0..n {
            for j in 0..n {
                let r = dist(f.x[i], f.at(j, cell));
                if r <= 0.0 {
                    continue;
                }
                let thr = k::RTHR * super::neighbors::bond_radius(el, qa, i, j);
                margin = margin.min((r - thr).abs());
                if r < thr {
                    out.push(bond_key(j, i, cell));
                }
            }
        }
    }
    out.sort_unstable();
    out.dedup();
    (out, margin)
}

/// The metallic character of the setup: from the molecular CN (`gfnff_dlogcoord`), or in a
/// periodic cell from the CN over the lattice points (`getCoordinationNumber`).
fn setup_mchar(el: &[Atom], f: &Frame<'_>) -> MoptResult<Vec<f64>> {
    match f.lattice {
        None => Ok(log_cn(el, f.x).1),
        Some(l) => super::periodic_cn::setup_mchar(el, f.x, l),
    }
}

/// One pass of xtb's neighbour search and what follows it in `gfnff_ini`. As in xtb 6.7.1,
/// the H atoms bonded to a lanthanide stay its neighbours (later states of xtb take those with
/// a topology charge above -0.0281 out in the second pass, `adjust_NB_LnH_AnH`).
fn pass_neighbors(el: &[Atom], f: &Frame<'_>, qa: &[f64], mchar: &[f64]) -> Pass {
    let n = f.len();
    let nb = neighbors(el, f, qa, mchar);
    // Bonds: each once, in xtb's order (neighbour, atom, cell) with the atom ascending; a bond
    // seen from its other atom is the same bond with the opposite cell.
    let mut seen = std::collections::HashSet::new();
    let mut bonds = Vec::new();
    for i in 0..n {
        for &(l, c) in &nb.nb[i] {
            if seen.insert((l, i, c)) {
                seen.insert((i, l, negate(c)));
                bonds.push((l, i, c));
            }
        }
    }
    // Pi atoms.
    let hyb = &nb.hyb;
    let mut piadr = Vec::new();
    let mut piadr2 = vec![0usize; n];
    for i in 0..n {
        let zi = el[i].z();
        let mut piat = (hyb[i] == 1 || hyb[i] == 2) && pilist(zi);
        let mut kk = 0;
        for &(jj, _) in &nb.nb[i] {
            if zi == 8 && el[jj].z() == 16 && hyb[jj] == 5 {
                piat = false;
                continue;
            }
            if hyb[jj] == 1 || hyb[jj] == 2 {
                kk += 1;
            }
        }
        let picon = kk > 0 && nofs(zi);
        if zi == 7 && nb.nb[i].len() > 3 {
            continue;
        }
        if zi == 16 && hyb[i] == 5 {
            continue;
        }
        if picon || piat {
            piadr.push(i);
            piadr2[i] = piadr.len();
        }
    }
    // Pi fragments.
    let pi_nb: Vec<Vec<usize>> = piadr
        .iter()
        .map(|&i| {
            nb.nb[i]
                .iter()
                .filter(|&&(kk, _)| piadr2[kk] > 0)
                .map(|&(kk, _)| piadr2[kk] - 1)
                .collect()
        })
        .collect();
    let (picount, pimvec) = components(&pi_nb);
    // EEQ electronegativity corrections.
    let mut dxi = vec![0.0; n];
    for i in 0..n {
        let ati = el[i].z();
        let nn = nb.nb[i].len();
        if nn == 0 {
            continue;
        }
        let ip = piadr2[i];
        // jth_nb(1): the closest neighbour (over the images).
        let ji = super::neighbors::jth_nb(&nb.nb[i], f, i, 1)
            .unwrap_or(nb.nb[i][0])
            .0;
        let nh = nb.nb[i].iter().filter(|&&(j, _)| el[j].z() == 1).count() as f64;
        let nm = nb.nb[i]
            .iter()
            .filter(|&&(j, _)| imetal_of(&el[j], nb.nb[j].len()) != 0)
            .count();
        if ati == 5 {
            dxi[i] += nh * 0.015;
        }
        if ati == 6 && nn == 2 && nb.itag[i] == 1 {
            dxi[i] = -0.15;
        }
        if ati == 6 && nn == 1 && el[ji].z() == 8 && nb.nb[ji].len() == 1 {
            dxi[ji] = 0.15;
        }
        if ati == 8 && nn == 1 && ip != 0 && el[ji].z() == 7 && piadr2[ji] != 0 {
            dxi[i] = 0.05;
        }
        if ati == 8 && nn == 2 && nh == 2.0 {
            dxi[i] = -0.02;
        }
        if el[i].raw.group == 6 && nn > 2 {
            dxi[i] += nn as f64 * 0.005;
        }
        if ati == 8 || ati == 16 {
            dxi[i] -= nh * 0.005;
        }
        if el[i].raw.group == 7 && ati > 9 && nn > 1 {
            if nm == 0 {
                dxi[i] -= nn as f64 * 0.021;
            } else {
                dxi[i] += nn as f64 * 0.05;
            }
        }
    }
    let mut chieeq = vec![0.0; n];
    let mut gameeq = vec![0.0; n];
    let mut alpeeq = vec![0.0; n];
    for i in 0..n {
        let dum = (nb.nb[i].len() as f64).min(k::CNMAX);
        chieeq[i] = -el[i].raw.chi + dxi[i] + el[i].raw.cnf * dum.sqrt();
        // The "true" charges of transition metals are small; their topology charges are
        // taken with a less electronegative metal (`mchishift`).
        if imetal_of(&el[i], nb.nb[i].len()) == 2 {
            chieeq[i] -= k::MCHISHIFT;
        }
        gameeq[i] = el[i].raw.gam;
        alpeeq[i] = el[i].raw.alp.powi(2);
    }
    Pass {
        nb,
        bonds,
        piadr,
        piadr2,
        pimvec,
        picount,
        dxi,
        chieeq,
        gameeq,
        alpeeq,
    }
}

/// The distance of the topology above which pairs are cut off (Angstrom, xtb's
/// `rabd_cutoff`).
const RABD_CUTOFF: f32 = 13.0;

/// The bond graph with xtb's edge lengths: `rabd` (single precision, Angstrom) with the sums
/// of the bonded radii on the bonds, 0 on the diagonal and [`RABD_CUTOFF`] elsewhere.
fn bonded_radii(el: &[Atom], nb: &[Vec<usize>]) -> Vec<Vec<f32>> {
    let n = el.len();
    let mut rabd = vec![vec![RABD_CUTOFF; n]; n];
    for i in 0..n {
        rabd[i][i] = 0.0;
        for &j in &nb[i] {
            let v = (el[i].raw.rad + el[j].raw.rad) as f32;
            rabd[j][i] = v;
            rabd[i][j] = v;
        }
    }
    rabd
}

/// `rabd` scaled to the topology distances of the EEQ (Bohr), pairs beyond `tdist_thr` at the
/// cutoff.
fn scaled_distances(rabd: &[Vec<f32>]) -> Vec<Vec<f64>> {
    let n = rabd.len();
    let thr = k::TDIST_THR as f32;
    let mut out = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..i {
            let mut d = rabd[j][i];
            if d > thr {
                d = RABD_CUTOFF;
            }
            let v = k::RFGOED1 * f64::from(d) / AUTOAA;
            out[i][j] = v;
            out[j][i] = v;
        }
    }
    out
}

/// Topology distances scaled to Bohr as in xtb, on the folded graph: xtb's Floyd-Warshall
/// over the bonded radii (single precision), whose steps skip every pair farther than
/// `tdist_thr` from the intermediate atom.
///
/// The same steps in the same order, without visiting the pairs they skip: in step `kk` the
/// row and the column of `kk` do not change (`rabd[kk][kk]` is 0), so the atoms within the
/// threshold of `kk` are listed once per step and only their pairs are visited; and `rabd`
/// stays symmetric (it starts so, and the update of `(i, j)` and of `(j, i)` add the same two
/// numbers), so the column is the row. The work is `n^2` for the lists plus the pairs within
/// the threshold instead of `n^3`, and every number is the one xtb's loop computes (a test
/// compares the whole setup with [`floyd_warshall_distances`] on the golden structures).
fn topology_distances(el: &[Atom], nb: &[Vec<usize>]) -> Vec<Vec<f64>> {
    let n = el.len();
    let mut rabd = bonded_radii(el, nb);
    let thr = k::TDIST_THR as f32;
    let mut near: Vec<(usize, f32)> = Vec::new();
    for kk in 0..n {
        near.clear();
        near.extend(
            rabd[kk]
                .iter()
                .enumerate()
                .filter(|&(_, &d)| d <= thr)
                .map(|(j, &d)| (j, d)),
        );
        for &(i, dik) in &near {
            let row = &mut rabd[i];
            for &(j, dkj) in &near {
                let via = dik + dkj;
                if row[j] > via {
                    row[j] = via;
                }
            }
        }
    }
    scaled_distances(&rabd)
}

/// xtb's loop of [`topology_distances`] as it is written (the reference of the tests).
#[cfg(test)]
fn floyd_warshall_distances(el: &[Atom], nb: &[Vec<usize>]) -> Vec<Vec<f64>> {
    let n = el.len();
    let mut rabd = vec![vec![RABD_CUTOFF; n]; n];
    for i in 0..n {
        rabd[i][i] = 0.0;
        for &j in &nb[i] {
            let v = (el[i].raw.rad + el[j].raw.rad) as f32;
            rabd[j][i] = v;
            rabd[i][j] = v;
        }
    }
    let thr = k::TDIST_THR as f32;
    for kk in 0..n {
        for i in 0..n {
            if rabd[i][kk] > thr {
                continue;
            }
            for j in 0..n {
                if rabd[kk][j] > thr {
                    continue;
                }
                if rabd[i][j] > rabd[i][kk] + rabd[kk][j] {
                    rabd[i][j] = rabd[i][kk] + rabd[kk][j];
                }
            }
        }
    }
    let mut out = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..i {
            let mut d = rabd[j][i];
            if d > thr {
                d = RABD_CUTOFF;
            }
            let v = k::RFGOED1 * f64::from(d) / AUTOAA;
            out[i][j] = v;
            out[j][i] = v;
        }
    }
    out
}

/// The topology setup of xtb's `gfnff_ini` up to the ring search, for a molecule
/// (`lattice` `None`) or a three-dimensional periodic cell (positions already wrapped into
/// the cell).
pub fn setup(el: Vec<Atom>, x: &[V3], lattice: Option<Lattice>, charge: f64) -> MoptResult<Setup> {
    setup_with(el, x, lattice, charge, FAST)
}

/// [`setup`] with the algorithms `kernels`.
fn setup_with(
    el: Vec<Atom>,
    x: &[V3],
    lattice: Option<Lattice>,
    charge: f64,
    kernels: SetupKernels,
) -> MoptResult<Setup> {
    let n = x.len();
    let f = Frame {
        x,
        lattice: lattice.as_ref(),
    };
    for cell in f.bond_cells() {
        for i in 0..n {
            for j in 0..n {
                if (j < i || cell != HOME) && super::neighbors::dist(x[i], f.at(j, cell)) < 1.0e-3 {
                    return Err(MoptError::config("GFN-FF: particularly close atoms"));
                }
            }
        }
    }
    let translations = match &lattice {
        Some(l) => Translations::new(l, k::HBTHR2.sqrt())?,
        None => Translations::molecule(),
    };
    let cells = f.bond_cells();
    let mchar = setup_mchar(&el, &f)?;
    let mut qa = vec![0.0; n];
    let mut nfrag = 0usize;
    let mut fraglist = vec![0usize; n];
    let mut qfrag = vec![0.0; n.max(1)];
    qfrag[0] = charge;
    let mut bpair = BondPairs::zeros(n, cells.clone());
    let mut ipis = Vec::new();
    let mut last = None;
    let mut qa_neighbors = qa.clone();
    for qloop in 0..2 {
        qa_neighbors.clone_from(&qa);
        let p = pass_neighbors(&el, &f, &qa, &mchar);
        if qloop == 1 {
            bpair = bond_pairs(&p.nb.nb, cells.clone(), &translations);
        }
        let flat = folded(&p.nb.nb);
        let pair = (kernels.distances)(&el, &flat);
        if nfrag <= 1 {
            let (nf, fl) = components(&flat);
            nfrag = nf;
            fraglist = fl;
        }
        let eeq = Goedecker::new(&p.gameeq, &p.alpeeq, &pair, &fraglist, nfrag)?;
        qa = eeq_charges(kernels, &eeq, &p, &pair, &fraglist, &qfrag, nfrag)?;
        if p.picount > 0 && qloop > 0 {
            let qtmp = qa.clone();
            let qah = qheavy(&el, &flat, &qa);
            ipis = vec![0i64; p.picount];
            for (pis, ip) in ipis.iter_mut().enumerate() {
                let first = (0..p.piadr.len())
                    .find(|&kk| p.pimvec[kk] == pis + 1)
                    .map(|kk| p.piadr[kk]);
                let Some(atom) = first else { continue };
                let ifrag = fraglist[atom] - 1;
                let saved = qfrag[ifrag];
                qfrag[ifrag] = 0.0;
                let qn = eeq_charges(kernels, &eeq, &p, &pair, &fraglist, &qfrag, nfrag)?;
                qfrag[ifrag] = saved;
                let qn = qheavy(&el, &flat, &qn);
                let mut dum = 0.0;
                for kk in 0..p.piadr.len() {
                    if p.pimvec[kk] == pis + 1 {
                        dum += qah[p.piadr[kk]] - qn[p.piadr[kk]];
                    }
                }
                dum *= 1.1;
                *ip = dum.round() as i64;
            }
            qa = qtmp;
        }
        last = Some(p);
    }
    let p = last.ok_or_else(|| MoptError::calc("GFN-FF topology loop did not run"))?;

    // EEQ J change with the estimated q (a third-order term) and the final atomic parameters.
    let hyb = &p.nb.hyb;
    let mut gameeq = vec![0.0; n];
    let mut chieeq = vec![0.0; n];
    let mut alpeeq = vec![0.0; n];
    // The folded graph and xtb's `piadr` as flags, made once (they were made for every atom).
    let flat = folded(&p.nb.nb);
    let piadr_as_flags: Vec<usize> = (0..n)
        .map(|a| p.piadr.get(a).map_or(0, |&v| v + 1))
        .collect();
    for i in 0..n {
        let z = el[i].z();
        let mut ff = 0.0;
        if z == 1 {
            ff = -0.08;
        }
        if z == 5 {
            ff = -0.05;
        }
        if z == 6 {
            ff = -0.27;
            if hyb[i] < 3 {
                ff = -0.45;
            }
            if hyb[i] < 2 {
                ff = -0.34;
            }
        }
        if z == 7 {
            ff = -0.13;
            // xtb passes `piadr` (pi atoms by index) here.
            if p.piadr.get(i).is_some_and(|&v| v + 1 != 0) {
                ff = -0.14;
            }
            if amide(&el, hyb, &flat, &piadr_as_flags, i) {
                ff = -0.16;
            }
        }
        if z == 8 {
            ff = -0.15;
            if hyb[i] < 3 {
                ff = -0.08;
            }
        }
        if z == 9 {
            ff = 0.10;
        }
        if z > 10 {
            ff = -0.02;
        }
        if z == 17 {
            ff = -0.02;
        }
        if z == 35 {
            ff = -0.11;
        }
        if z == 53 {
            ff = -0.07;
        }
        let imetal = imetal_of(&el[i], p.nb.nb[i].len());
        if imetal == 1 {
            ff = -0.08;
        }
        if imetal == 2 {
            ff = -0.9;
        }
        if el[i].raw.group == 8 {
            ff = 0.0;
        }
        gameeq[i] = el[i].raw.gam + qa[i] * ff;
        chieeq[i] = -el[i].raw.chi + p.dxi[i];
        if amide_h(&el, hyb, &flat, &p.piadr2, i) {
            chieeq[i] -= 0.02;
        }
        let mut fa = 0.0;
        if z == 6 {
            fa = 0.09;
        }
        if z == 7 {
            fa = -0.21;
        }
        if el[i].raw.group == 6 {
            fa = -0.03;
        }
        if el[i].raw.group == 7 {
            fa = 0.50;
        }
        if imetal == 1 {
            fa = 0.3;
        }
        if imetal == 2 {
            fa = -0.1;
        }
        alpeeq[i] = (el[i].raw.alp + fa * qa[i]).powi(2);
    }

    // Rings from the `nbm` list of the folded graph (the neighbours of all cells
    // concatenated, duplicates kept, as xtb's `nbrngs`).
    let nbm = folded(&p.nb.nbm);
    let rings: Vec<AtomRings> = (0..n).map(|i| getring36(&nbm, &p.nb.cluster, i)).collect();

    let imetal: Vec<i32> = (0..n)
        .map(|i| imetal_of(&el[i], p.nb.nb[i].len()))
        .collect();
    Ok(Setup {
        el,
        bonds: p.bonds,
        piadr: p.piadr,
        piadr2: p.piadr2,
        pimvec: p.pimvec,
        picount: p.picount,
        ipis,
        qa,
        qa_neighbors,
        mchar,
        chieeq,
        gameeq,
        alpeeq,
        fraglist,
        qfrag,
        bpair,
        rings,
        dxi: p.dxi,
        imetal,
        nb: p.nb,
        lattice,
        translations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core_infra::json::JsonValue;
    use crate::domain::{Element, MolecularSystem, SpinCharge};

    fn number(v: &JsonValue) -> MoptResult<f64> {
        v.as_f64()
            .ok_or_else(|| MoptError::config("expected a number"))
    }

    /// Without metal atoms, the bonds of the shortcut of `bonds_with_margin` are those of the
    /// whole neighbour search, at the golden structures and at structures stretched and
    /// squeezed by 10 % (where bonds form and break); and no distance is closer to its
    /// threshold than the margin.
    #[test]
    fn the_bonds_without_metals_are_those_of_the_whole_search() -> MoptResult<()> {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/golden/gfnff.json");
        let text = std::fs::read_to_string(path)?;
        let doc = JsonValue::parse(&text)?;
        let cases = doc
            .get("cases")
            .and_then(JsonValue::as_array)
            .ok_or_else(|| MoptError::config("cases"))?;
        let mut tested = 0;
        for case in cases {
            let symbols = case
                .get("symbols")
                .and_then(JsonValue::as_string_vec)
                .ok_or_else(|| MoptError::config("symbols"))?;
            let el: Vec<Element> = symbols
                .iter()
                .map(|s| Element::parse_token(s))
                .collect::<MoptResult<_>>()?;
            let x: Vec<f64> = case
                .get("positions_bohr")
                .and_then(JsonValue::as_array)
                .ok_or_else(|| MoptError::config("positions"))?
                .iter()
                .map(number)
                .collect::<MoptResult<_>>()?;
            let charge = number(case.get("charge").ok_or_else(|| MoptError::config("q"))?)?;
            let sys = MolecularSystem::from_flat_bohr(
                el,
                &Col::from_fn(x.len(), |i| x[i]),
                SpinCharge::new(charge as i32, 1),
                None,
            )?;
            let Ok(topo) = super::super::GfnffTopology::new(&sys) else {
                continue;
            };
            let s = &topo.setup;
            if s.el.iter().any(|a| a.metal() != 0) {
                continue;
            }
            for scale in [1.0, 1.1, 0.9] {
                let pos: Vec<V3> = (0..sys.n_atoms())
                    .map(|i| [0, 1, 2].map(|c| scale * x[3 * i + c]))
                    .collect();
                let f = Frame {
                    x: &pos,
                    lattice: None,
                };
                let mchar = setup_mchar(&s.el, &f)?;
                let nb = neighbors(&s.el, &f, &s.qa_neighbors, &mchar);
                let mut whole: Vec<(usize, usize, Cell)> = (0..f.len())
                    .flat_map(|i| nb.nb[i].iter().map(move |&(j, c)| bond_key(j, i, c)))
                    .collect();
                whole.sort_unstable();
                whole.dedup();
                let (fast, margin) = nonmetal_bonds(&s.el, &f, &s.qa_neighbors);
                assert_eq!(whole, fast, "{symbols:?} scaled by {scale}");
                // Infinite for a single atom (no pair to bound).
                assert!(margin >= 0.0 && (margin.is_finite() || f.len() < 2));
                tested += 1;
            }
        }
        assert!(tested > 50, "{tested}");
        Ok(())
    }

    /// The restricted loop against xtb's for the bond graph of a grid of water molecules: the
    /// same numbers, and the times (648 atoms on every run; with `MOPT_FULL_MEASUREMENTS` set
    /// also 3993 atoms and the enzyme model at `MOPT_ENZYME_XYZ`, if given).
    #[test]
    fn time_of_the_topology_distances() -> MoptResult<()> {
        let mut files: Vec<(String, MolecularSystem)> = Vec::new();
        let full = std::env::var_os("MOPT_FULL_MEASUREMENTS").is_some();
        for m in if full { vec![6, 11] } else { vec![6] } {
            let (r, half) = (1.81, 0.912_f64);
            let mut el = Vec::new();
            let mut x = Vec::new();
            for a in 0..m {
                for b in 0..m {
                    for c in 0..m {
                        let o = [a, b, c].map(|v| v as f64 * 5.8);
                        let (s, co) = (0.7 * (a + 2 * b + 3 * c) as f64).sin_cos();
                        el.extend([Element::O, Element::H, Element::H]);
                        x.extend_from_slice(&o);
                        x.extend([
                            o[0] + r * half.sin() * co,
                            o[1] + r * half.sin() * s,
                            o[2] + r * half.cos(),
                        ]);
                        x.extend([
                            o[0] - r * half.sin() * co,
                            o[1] - r * half.sin() * s,
                            o[2] + r * half.cos(),
                        ]);
                    }
                }
            }
            let sys = MolecularSystem::from_flat_bohr(
                el,
                &Col::from_fn(x.len(), |i| x[i]),
                SpinCharge::new(0, 1),
                None,
            )?;
            files.push((format!("water grid {}", sys.n_atoms()), sys));
        }
        if let Some(path) = std::env::var_os("MOPT_ENZYME_XYZ").filter(|_| full) {
            let sys = crate::io::xyz::read_xyz_file(
                std::path::Path::new(&path),
                SpinCharge::default(),
                None,
            )?;
            files.push((format!("enzyme {}", sys.n_atoms()), sys));
        }
        for (name, sys) in files {
            let el: Vec<Atom> = sys
                .elements
                .iter()
                .map(|&e| super::super::elements::atom(e))
                .collect::<MoptResult<_>>()?;
            let x = super::super::positions(&sys);
            let f = Frame {
                x: &x,
                lattice: None,
            };
            let mchar = setup_mchar(&el, &f)?;
            let p = pass_neighbors(&el, &f, &vec![0.0; x.len()], &mchar);
            let flat = folded(&p.nb.nb);
            let start = std::time::Instant::now();
            let fast = topology_distances(&el, &flat);
            let t_fast = start.elapsed().as_secs_f64();
            let start = std::time::Instant::now();
            let reference = floyd_warshall_distances(&el, &flat);
            let t_reference = start.elapsed().as_secs_f64();
            assert!(fast == reference, "{name}");
            let (nfrag, fraglist) = components(&flat);
            let start = std::time::Instant::now();
            let eeq = Goedecker::new(&p.gameeq, &p.alpeeq, &fast, &fraglist, nfrag)?;
            eeq.charges(&p.chieeq, &vec![0.0; x.len().max(nfrag)])?;
            let t_eeq = start.elapsed().as_secs_f64();
            println!(
                "{name}: distances {t_fast:.3} s (xtb's loop {t_reference:.3} s); EEQ of the \
                 topology charges ({nfrag} fragments) {t_eeq:.3} s"
            );
        }
        Ok(())
    }

    /// xtb's algorithms as they are written: the Floyd-Warshall loop over all pairs and the
    /// EEQ factorized for every right-hand side.
    const REFERENCE: SetupKernels = SetupKernels {
        distances: floyd_warshall_distances,
        reference_eeq: true,
    };

    /// The systems of a golden file (`periodic`: with their cells).
    fn golden_systems(file: &str, periodic: bool) -> MoptResult<Vec<MolecularSystem>> {
        use crate::domain::{Bohr, PeriodicCell, Vector3};
        let path = format!("{}/../../tests/golden/{file}", env!("CARGO_MANIFEST_DIR"));
        let doc = JsonValue::parse(&std::fs::read_to_string(path)?)?;
        let cases = doc
            .get("cases")
            .and_then(JsonValue::as_array)
            .ok_or_else(|| MoptError::config("cases"))?;
        let floats = |v: Option<&JsonValue>| -> MoptResult<Vec<f64>> {
            v.and_then(JsonValue::as_array)
                .ok_or_else(|| MoptError::config("expected an array"))?
                .iter()
                .map(number)
                .collect()
        };
        let mut out = Vec::new();
        for case in cases {
            let el: Vec<Element> = case
                .get("symbols")
                .and_then(JsonValue::as_string_vec)
                .ok_or_else(|| MoptError::config("symbols"))?
                .iter()
                .map(|s| Element::parse_token(s))
                .collect::<MoptResult<_>>()?;
            let x = floats(case.get("positions_bohr"))?;
            let charge = number(case.get("charge").ok_or_else(|| MoptError::config("q"))?)?;
            let cell = if periodic {
                let rows: Vec<Vec<f64>> = case
                    .get("lattice_bohr")
                    .and_then(JsonValue::as_array)
                    .ok_or_else(|| MoptError::config("lattice_bohr"))?
                    .iter()
                    .map(|r| floats(Some(r)))
                    .collect::<MoptResult<_>>()?;
                let v = |k: usize| Vector3::<Bohr>::new([rows[k][0], rows[k][1], rows[k][2]]);
                Some(PeriodicCell::from_lattice_vectors(
                    v(0),
                    v(1),
                    v(2),
                    [true; 3],
                )?)
            } else {
                None
            };
            out.push(MolecularSystem::from_flat_bohr(
                el,
                &Col::from_fn(x.len(), |i| x[i]),
                SpinCharge::new(charge as i32, 1),
                cell,
            )?);
        }
        Ok(out)
    }

    /// The setup of `sys` with `kernels`, as `GfnffTopology::new` makes it.
    fn setup_of(sys: &MolecularSystem, kernels: SetupKernels) -> MoptResult<Setup> {
        let lattice = super::super::lattice_of(sys)?;
        let el = sys
            .elements
            .iter()
            .map(|&e| super::super::elements::atom(e))
            .collect::<MoptResult<Vec<_>>>()?;
        let x = match &lattice {
            Some(l) => l.wrap(&super::super::positions(sys))?,
            None => super::super::positions(sys),
        };
        setup_with(el, &x, lattice, f64::from(sys.spin_charge.charge), kernels)
    }

    /// The whole setup (bonds, neighbours, hybridizations, pi systems, charges, EEQ
    /// parameters, fragments, bond paths, rings) with the restricted Floyd-Warshall loop and
    /// the EEQ factorized once is that of xtb's algorithms, bit for bit, on every golden
    /// molecule and every periodic reference.
    #[test]
    fn the_setup_is_that_of_the_reference_algorithms() -> MoptResult<()> {
        let mut systems = golden_systems("gfnff.json", false)?;
        let molecules = systems.len();
        systems.extend(golden_systems("gfnff_pbc_reference.json", true)?);
        let mut compared = 0;
        for sys in &systems {
            let fast = setup_of(sys, FAST);
            let reference = setup_of(sys, REFERENCE);
            match (fast, reference) {
                (Ok(a), Ok(b)) => {
                    // Everything but the translations (made before, from the cell alone, and
                    // printed from a hash map in no fixed order).
                    let whole = |s: &Setup| {
                        format!(
                            "{:?}",
                            (
                                (&s.el, &s.nb, &s.bonds, &s.piadr, &s.piadr2, &s.pimvec),
                                (s.picount, &s.ipis, &s.qa, &s.qa_neighbors, &s.mchar),
                                (&s.chieeq, &s.gameeq, &s.alpeeq, &s.fraglist, &s.qfrag),
                                (&s.bpair, &s.rings, &s.dxi, &s.imetal, &s.lattice),
                            )
                        )
                    };
                    let (a, b) = (whole(&a), whole(&b));
                    if a != b {
                        let at = a
                            .bytes()
                            .zip(b.bytes())
                            .position(|(x, y)| x != y)
                            .unwrap_or(a.len().min(b.len()));
                        let from = at.saturating_sub(300);
                        return Err(MoptError::calc(format!(
                            "{} atoms: the setups differ at\n{}\nagainst\n{}",
                            sys.n_atoms(),
                            &a[from..(at + 100).min(a.len())],
                            &b[from..(at + 100).min(b.len())]
                        )));
                    }
                    compared += 1;
                }
                (Err(a), Err(b)) => assert_eq!(a.to_string(), b.to_string()),
                (a, b) => {
                    return Err(MoptError::calc(format!(
                        "one setup failed: {:?} against {:?}",
                        a.err(),
                        b.err()
                    )))
                }
            }
        }
        assert!(molecules >= 170 && compared > molecules, "{compared}");
        Ok(())
    }
}
