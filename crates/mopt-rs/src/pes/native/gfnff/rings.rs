//! Ring search and ring queries of GFN-FF, ported from xtb (`src/gfnff/gfnff_ini2.F90`:
//! `getring36`, `ssort`, `chkrng`, `ringsatom`, `ringsbond`, `ringsbend`, `ringstors`,
//! `ringstorl`; LGPL-3.0-or-later) for isolated molecules, keeping xtb's quirks (the skipped
//! neighbour that is atom 1, the `s(m,j)` comparison in `ringsbend`).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

/// Rings of one atom: up to 19 rings of 3 to 6 atoms (atom lists end with the atom itself).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AtomRings {
    /// Atoms of each ring (0-based).
    pub rings: Vec<Vec<usize>>,
}

impl AtomRings {
    fn count(&self) -> usize {
        self.rings.len()
    }

    fn size(&self, k: usize) -> usize {
        self.rings[k].len()
    }
}

/// `chkrng`: every atom of the candidate occurs once.
fn chkrng(n: usize, c: &[usize]) -> bool {
    let mut count = vec![0usize; n];
    for &a in c {
        count[a] += 1;
    }
    count.iter().filter(|&&v| v == 1).count() == c.len()
}

/// `getring36` for atom `a0`; `nb` are the neighbour lists (0-based) and `cluster` the atoms
/// excluded from the ring search (`nb(numnb-1, i) == 1` in xtb).
pub fn getring36(nb: &[Vec<usize>], cluster: &[bool], a0: usize) -> AtomRings {
    let n = nb.len();
    if n <= 2 || cluster[a0] {
        return AtomRings::default();
    }
    const MAXR: usize = 500;
    let nn = nb[a0].len();
    let mut found: Vec<Vec<usize>> = Vec::new();
    'outer: for m in 0..nn {
        // xtb: `if (nb(m, a0) == 1) cycle`, i.e. the neighbour that is atom 1 is skipped.
        if nb[a0][m] == 0 {
            continue;
        }
        // Terminal atoms get no neighbours; the m-th neighbour of a0 goes first.
        let mut work: Vec<Vec<usize>> = nb
            .iter()
            .map(|l| if l.len() == 1 { Vec::new() } else { l.clone() })
            .collect();
        let mut order = vec![nb[a0][m]];
        order.extend((0..nn).filter(|&k| k != m).map(|k| nb[a0][k]));
        work[a0] = if nb[a0].len() == 1 { Vec::new() } else { order };
        let mut c = [0usize; 6];
        for &a1 in &work[a0] {
            if a1 == a0 {
                continue;
            }
            for &a2 in &work[a1] {
                if a2 == a1 {
                    continue;
                }
                for &a3 in &work[a2] {
                    if a3 == a2 {
                        continue;
                    }
                    c[0] = a1;
                    c[1] = a2;
                    c[2] = a3;
                    if a3 == a0 && chkrng(n, &c[..3]) {
                        if found.len() == MAXR {
                            break 'outer;
                        }
                        found.push(c[..3].to_vec());
                    }
                    for &a4 in &work[a3] {
                        if a4 == a3 {
                            continue;
                        }
                        c[3] = a4;
                        if a4 == a0 && chkrng(n, &c[..4]) {
                            if found.len() == MAXR {
                                break 'outer;
                            }
                            found.push(c[..4].to_vec());
                        }
                        for &a5 in &work[a4] {
                            if a5 == a4 {
                                continue;
                            }
                            c[4] = a5;
                            if a5 == a0 && chkrng(n, &c[..5]) {
                                if found.len() == MAXR {
                                    break 'outer;
                                }
                                found.push(c[..5].to_vec());
                            }
                            for &a6 in &work[a5] {
                                if a6 == a5 {
                                    continue;
                                }
                                c[5] = a6;
                                if a6 == a0 && chkrng(n, &c[..6]) {
                                    if found.len() == MAXR {
                                        break 'outer;
                                    }
                                    found.push(c[..6].to_vec());
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    // Remove rings with the same atoms (and size).
    let mut same = vec![false; found.len()];
    for i in 0..found.len() {
        for j in i + 1..found.len() {
            if found[i].len() != found[j].len() || same[j] {
                continue;
            }
            let mut a = found[i].clone();
            let mut b = found[j].clone();
            a.sort_unstable();
            b.sort_unstable();
            same[j] = a == b;
        }
    }
    let mut rings = Vec::new();
    for (i, r) in found.into_iter().enumerate() {
        if !same[i] {
            rings.push(r);
            if rings.len() > 19 {
                rings.truncate(19);
                break;
            }
        }
    }
    AtomRings { rings }
}

/// `ringsatom`: smallest ring of atom `i` (99 when none).
pub fn ringsatom(r: &[AtomRings], i: usize) -> usize {
    r[i].rings.iter().map(Vec::len).min().unwrap_or(99).min(99)
}

/// `ringsbond`: smallest ring containing bond `i-j` (0 when none).
pub fn ringsbond(r: &[AtomRings], i: usize, j: usize) -> usize {
    let side = |a: usize, b: usize| {
        r[a].rings
            .iter()
            .filter(|ring| ring.contains(&b))
            .map(Vec::len)
            .min()
            .unwrap_or(99)
    };
    let v = side(i, j).min(side(j, i));
    if v == 99 {
        0
    } else {
        v
    }
}

/// `ringsbend`: smallest ring containing angle `i-j-k` (0 when none), with xtb's comparison of
/// the third atom's ring sizes against those of `j`.
pub fn ringsbend(r: &[AtomRings], i: usize, j: usize, k: usize) -> usize {
    if r[i].count() == 0 || r[j].count() == 0 || r[k].count() == 0 {
        return 0;
    }
    let hits = |a: usize, m: usize, p: usize, q: usize| {
        r[a].rings[m].iter().filter(|&&x| x == p || x == q).count()
    };
    let mut r1 = 99;
    for m in 0..r[i].count() {
        if hits(i, m, j, k) == 2 && r[i].size(m) < r1 {
            r1 = r[i].size(m);
        }
    }
    let mut r2 = 99;
    for m in 0..r[j].count() {
        if hits(j, m, i, k) == 2 && r[j].size(m) < r2 {
            r2 = r[j].size(m);
        }
    }
    let mut r3 = 99;
    for m in 0..r[k].count() {
        // xtb compares s(m, j) (the m-th ring of j, 0 when j has fewer rings) here.
        let s_j = if m < r[j].count() { r[j].size(m) } else { 0 };
        if hits(k, m, i, j) == 2 && s_j < r3 {
            r3 = r[k].size(m);
        }
    }
    let v = r1.min(r2).min(r3);
    if v == 99 {
        0
    } else {
        v
    }
}

fn hits3(r: &[AtomRings], a: usize, m: usize, p: [usize; 3]) -> usize {
    r[a].rings[m].iter().filter(|x| p.contains(x)).count()
}

/// `ringstors`: smallest ring containing torsion `i-j-k-l` (0 when none).
pub fn ringstors(r: &[AtomRings], i: usize, j: usize, k: usize, l: usize) -> usize {
    let atoms = [i, j, k, l];
    if atoms.iter().any(|&a| r[a].count() == 0) {
        return 0;
    }
    let mut best = 99;
    for (pos, &a) in atoms.iter().enumerate() {
        let others: Vec<usize> = atoms
            .iter()
            .enumerate()
            .filter(|(p, _)| *p != pos)
            .map(|(_, &x)| x)
            .collect();
        let others = [others[0], others[1], others[2]];
        let mut v = 99;
        for m in 0..r[a].count() {
            if hits3(r, a, m, others) == 3 && r[a].size(m) < v {
                v = r[a].size(m);
            }
        }
        best = best.min(v);
    }
    if best == 99 {
        0
    } else {
        best
    }
}

/// `ringstorl`: largest ring containing torsion `i-j-k-l` (0 when none).
pub fn ringstorl(r: &[AtomRings], i: usize, j: usize, k: usize, l: usize) -> usize {
    let atoms = [i, j, k, l];
    if atoms.iter().any(|&a| r[a].count() == 0) {
        return 0;
    }
    let mut best: i64 = -99;
    for (pos, &a) in atoms.iter().enumerate() {
        let others: Vec<usize> = atoms
            .iter()
            .enumerate()
            .filter(|(p, _)| *p != pos)
            .map(|(_, &x)| x)
            .collect();
        let others = [others[0], others[1], others[2]];
        let mut v: i64 = -99;
        for m in 0..r[a].count() {
            if hits3(r, a, m, others) == 3 && (r[a].size(m) as i64) > v {
                v = r[a].size(m) as i64;
            }
        }
        best = best.max(v);
    }
    if best == -99 {
        0
    } else {
        best as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn benzene_ring_and_chain() {
        // Six-ring 0..5 with a substituent 6 on atom 0.
        let mut nb = vec![Vec::new(); 7];
        let bonds = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 6)];
        for &(a, b) in &bonds {
            nb[a].push(b);
            nb[b].push(a);
        }
        for l in &mut nb {
            l.sort_unstable();
        }
        let cluster = vec![false; 7];
        let r: Vec<AtomRings> = (0..7).map(|i| getring36(&nb, &cluster, i)).collect();
        assert_eq!(ringsatom(&r, 2), 6);
        assert_eq!(ringsatom(&r, 6), 99);
        assert_eq!(ringsbond(&r, 0, 1), 6);
        assert_eq!(ringsbond(&r, 0, 6), 0);
        assert_eq!(ringstors(&r, 0, 1, 2, 3), 6);
    }
}
