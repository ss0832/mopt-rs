//! The iterative Hueckel calculation of GFN-FF's pi bond orders, ported from xtb
//! (`gfnff_ini.f90` "do Hueckel"; `gfnffqmsolve` of `gfnff_qm.f90`; `occu`, `fermismear`, and
//! `dmat` of `scc_core.f90`; LGPL-3.0-or-later), including the Fermi-level update that is
//! applied even when its guard rejects it.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Mat;

use super::constants as k;
use super::neighbors::dist;
use super::topology::Setup;
use crate::error::MoptResult;
use crate::math::linalg::symmetric_eigen;

/// xtb's Boltzmann constant (Hartree / K) and Hartree in eV.
const KB: f64 = 3.166_808_578_545_117e-6;
const AUTOEV: f64 = 27.211_385_05;

/// `occu` with `nopen = 0`: alpha / beta occupations and the HOMO indices (1-based, 0 none).
fn occu(ndim: usize, nel: usize) -> (Vec<f64>, Vec<f64>, usize, usize) {
    let mut focc = vec![0.0f64; ndim];
    let mut fa = vec![0.0; ndim];
    let mut fb = vec![0.0; ndim];
    if nel.is_multiple_of(2) {
        for v in focc.iter_mut().take(nel / 2) {
            *v = 2.0;
        }
    } else {
        let na = nel / 2 + 1;
        let nb = nel / 2;
        for v in focc.iter_mut().take(na) {
            *v += 1.0;
        }
        for v in focc.iter_mut().take(nb) {
            *v += 1.0;
        }
    }
    for i in 0..ndim {
        if focc[i] == 2.0 {
            fa[i] = 1.0;
            fb[i] = 1.0;
        }
        if focc[i] == 1.0 {
            fa[i] = 1.0;
        }
    }
    let mut ha = 0;
    let mut hb = 0;
    for i in 0..ndim {
        if fa[i] > 0.99 {
            ha = i + 1;
        }
        if fb[i] > 0.99 {
            hb = i + 1;
        }
    }
    (fa, fb, ha, hb)
}

/// `fermismear` for `nel` electrons (1-based HOMO index) at temperature `t`, starting from the
/// occupations `occ` (which xtb leaves unchanged when there is no virtual orbital).
fn fermismear(eig: &[f64], nel: usize, t: f64, mut occ: Vec<f64>) -> Vec<f64> {
    let norbs = eig.len();
    let boltz = KB * AUTOEV;
    let bkt = boltz * t;
    let thr = 1e-9;
    if nel + 1 > norbs {
        return occ;
    }
    let mut e_fermi = 0.5 * (eig[nel - 1] + eig[nel]);
    let occt = nel as f64;
    for _ in 0..200 {
        let mut total = 0.0;
        let mut total_d = 0.0;
        for i in 0..norbs {
            let x = (eig[i] - e_fermi) / bkt;
            let mut f = 0.0;
            let df;
            if x < 50.0 {
                f = 1.0 / (x.exp() + 1.0);
                df = x.exp() / (bkt * (x.exp() + 1.0).powi(2));
            } else {
                df = 0.0;
            }
            occ[i] = f;
            total += f;
            total_d += df;
        }
        let change = (occt - total) / total_d;
        e_fermi += change;
        if (occt - total).abs() <= thr {
            break;
        }
    }
    occ
}

/// `gfnffqmsolve` (ZDO case): the density of `a` with Fermi smearing at `et` (Kelvin) for
/// `nel` electrons, and the band energy.
pub fn qmsolve(
    a: &Mat<f64>,
    et: f64,
    nel: usize,
) -> MoptResult<(Mat<f64>, f64, Vec<f64>, Vec<f64>)> {
    let ndim = a.nrows();
    let (vals, vecs) = symmetric_eigen(a)?;
    let e: Vec<f64> = vals.iter().map(|v| v * 0.1 * 27.211_395_7).collect();
    let mut focc;
    if et > 1.0e-3 {
        let (mut fa, mut fb, ha, hb) = occu(ndim, nel);
        if ha <= ndim && ha > 0 {
            fa = fermismear(&e, ha, et, fa);
        } else {
            fa.iter_mut().for_each(|v| *v = 0.0);
        }
        if hb <= ndim && hb > 0 {
            fb = fermismear(&e, hb, et, fb);
        } else {
            fb.iter_mut().for_each(|v| *v = 0.0);
        }
        focc = fa.iter().zip(&fb).map(|(x, y)| x + y).collect::<Vec<f64>>();
        if ha < ndim && ha > 0 && (focc[ha - 1] - focc[ha]).abs() < 1.0e-4 {
            // A perfect biradical: break the symmetry (anti-aromatic, as COT or CB).
            focc = vec![0.0; ndim];
            for v in focc.iter_mut().take(nel / 2) {
                *v = 2.0;
            }
        }
    } else {
        focc = vec![0.0; ndim];
        for v in focc.iter_mut().take(nel / 2) {
            *v = 2.0;
        }
        if nel % 2 == 1 {
            focc[nel / 2] = 1.0;
        }
    }
    let eel: f64 = focc.iter().zip(&e).map(|(f, v)| f * v).sum();
    let p = Mat::from_fn(ndim, ndim, |i, j| {
        (0..ndim)
            .map(|m| vecs[(i, m)] * focc[m] * vecs[(j, m)])
            .sum()
    });
    Ok((p, eel, focc, e))
}

/// Pi bond orders of all bonds (`pibo`, 0 for non-pi bonds) and the pi atoms after Hueckel
/// (`piadr = itmp` of xtb: 1 for atoms in a pi bond).
pub fn pi_bond_orders(s: &Setup, x: &[[f64; 3]]) -> MoptResult<(Vec<f64>, Vec<bool>)> {
    let n = x.len();
    let f = s.frame(x);
    let mut pibo = vec![0.0; s.bonds.len()];
    let mut in_pi = vec![false; n];
    if s.picount == 0 {
        return Ok((pibo, in_pi));
    }
    let hyb = &s.nb.hyb;
    for pis in 1..=s.picount {
        let mut piadr3 = Vec::new();
        let mut piadr4 = vec![0usize; n];
        let mut piel = vec![0i64; n];
        let mut nelpi: i64 = 0;
        for (kk, &atom) in s.piadr.iter().enumerate() {
            if s.pimvec[kk] != pis {
                continue;
            }
            let ati = s.el[atom].z();
            let hybi = hyb[atom];
            let tag = s.nb.itag[atom];
            let before = nelpi;
            if ati == 5 && hybi == 1 {
                nelpi += 1;
            }
            if ati == 6 && tag != 1 {
                nelpi += 1;
            }
            if ati == 7 && hybi == 2 && tag == 1 {
                nelpi += 1;
            }
            if ati == 7 && hybi <= 2 {
                nelpi += 1;
            }
            if ati == 7 && hybi == 3 {
                nelpi += 2;
            }
            if ati == 8 && hybi == 1 {
                nelpi += 1;
            }
            if ati == 8 && hybi == 2 {
                nelpi += 1;
            }
            if ati == 8 && hybi == 3 {
                nelpi += 2;
            }
            if ati == 9 && hybi != 1 {
                nelpi += 2;
            }
            if ati == 9 && hybi == 1 {
                nelpi += 3;
            }
            if ati == 16 && hybi == 1 {
                nelpi += 1;
            }
            if ati == 16 && hybi == 2 {
                nelpi += 1;
            }
            if ati == 16 && hybi == 3 {
                nelpi += 2;
            }
            if ati == 17 && hybi == 0 {
                nelpi += 2;
            }
            if ati == 17 && hybi == 1 {
                nelpi += 3;
            }
            piadr3.push(atom);
            piadr4[atom] = piadr3.len();
            piel[atom] = (nelpi - before).min(2);
        }
        nelpi -= s.ipis.get(pis - 1).copied().unwrap_or(0);
        let npi = piadr3.len();
        if npi < 2 || nelpi < 1 {
            continue;
        }
        let nel = nelpi as usize;
        let build = |pold: &Mat<f64>| {
            let mut api = Mat::<f64>::zeros(npi, npi);
            for i in 0..npi {
                let ii = piadr3[i];
                api[(i, i)] = k::hdiag(s.el[ii].z()) + s.qa[ii] * k::HUECKELP3
                    - (piel[ii] - 1) as f64 * k::PILPF;
            }
            // Indexed by atom: bonds of one atom pair through different cells assign the same
            // element, the later one last (xtb, no k-points).
            for &(jj, ii, c) in &s.bonds {
                let (ia, ja) = (piadr4[ii], piadr4[jj]);
                if ia > 0 && ja > 0 {
                    let mut dum = 1.0e-9 * dist(x[ii], f.at(jj, c));
                    dum = (k::hoffdiag(s.el[ii].z()) * k::hoffdiag(s.el[jj].z())).sqrt() - dum;
                    let mut dum2 = k::HITER;
                    if hyb[ii] == 1 {
                        dum2 *= k::HTRIPLE;
                    }
                    if hyb[jj] == 1 {
                        dum2 *= k::HTRIPLE;
                    }
                    let v = -dum * (1.0 - dum2 * (2.0 / 3.0 - pold[(ja - 1, ia - 1)]));
                    api[(ja - 1, ia - 1)] = v;
                    api[(ia - 1, ja - 1)] = v;
                }
            }
            api
        };
        let mut pold = Mat::from_fn(npi, npi, |_, _| 2.0 / 3.0);
        let mut eold = 0.0;
        let mut p = Mat::<f64>::zeros(npi, npi);
        let mut last_api = Mat::<f64>::zeros(npi, npi);
        let mut pisip = 0.0;
        for _ in 0..k::MAXHITER {
            let api = build(&pold);
            let (dens, e, occ, eps) = qmsolve(&api, 4000.0, nel)?;
            for i in 0..npi {
                if occ[i] > 0.5 {
                    pisip = eps[i];
                }
            }
            last_api = api;
            p = dens;
            if (e - eold).abs() < 1.0e-4 {
                break;
            }
            pold = p.clone();
            eold = e;
        }
        if pisip > 0.40 {
            // Probably a wrong pi occupation: second attempt with one electron less.
            let (dens, _, _, _) = qmsolve(&last_api, 4000.0, nel - 1)?;
            p = dens;
        }
        for (b, &(jj, ii, _)) in s.bonds.iter().enumerate() {
            let (ia, ja) = (piadr4[ii], piadr4[jj]);
            if ia > 0 && ja > 0 {
                pibo[b] = p[(ja - 1, ia - 1)];
                in_pi[ii] = true;
                in_pi[jj] = true;
            }
        }
    }
    Ok((pibo, in_pi))
}
