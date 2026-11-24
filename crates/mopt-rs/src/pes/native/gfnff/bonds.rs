//! GFN-FF bond parameters (`vbond` and the bond types), ported from xtb's `gfnff_ini`
//! ("bonds" section of `src/gfnff/gfnff_ini.f90`, with `ctype` of `gfnff_ini2.F90`;
//! LGPL-3.0-or-later), with the branch of bonds to metals: the main-group ones and the
//! transition metals (metal-metal bonds by row and metallic character, the factors of their
//! ligands, CO, CN and NC on a metal, the metal and eta shifts).
//! In a periodic cell the rules are those of the molecule with the neighbours of all cells
//! counted and the rings of the folded graph (by atom index), as in xtb.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use super::constants as k;
use super::elements::row6;
use super::rings::{ringsatom, ringsbond};
use super::topology::Setup;

/// Parameters of one bond.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BondParams {
    /// `vbond(1)`: shift of the reference bond length (Bohr).
    pub shift: f64,
    /// `vbond(2)`: steepness.
    pub alpha: f64,
    /// `vbond(3)`: prefactor (negative).
    pub prefactor: f64,
    /// Bond type `btyp` (1 single, 2 pi, 3 sp, 4 hypervalent).
    pub btyp: i32,
}

/// `ctype`: 1 for a C=O carbon (one pi O neighbour and itself a pi atom).
fn ctype(s: &Setup, pi: &[bool], a: usize) -> i32 {
    let no = s.nb.nb[a]
        .iter()
        .filter(|&&(j, _)| s.el[j].z() == 8 && pi[j])
        .count();
    if no == 1 && pi[a] {
        1
    } else {
        0
    }
}

/// Bond parameters of every bond; `pibo` are the Hueckel pi bond orders, `pi` the pi atoms
/// after the Hueckel step.
pub fn bond_params(s: &Setup, pibo: &[f64], pi: &[bool]) -> Vec<BondParams> {
    let hyb = &s.nb.hyb;
    let mut out = Vec::with_capacity(s.bonds.len());
    for (b, &(jj, ii, _)) in s.bonds.iter().enumerate() {
        let nni = s.nb.nb[ii].len();
        let nnj = s.nb.nb[jj].len();
        let (ia, ja) = (s.el[ii].z(), s.el[jj].z());
        let (gi, gj) = (s.el[ii].raw.group, s.el[jj].raw.group);
        let rings = ringsbond(&s.rings, ii, jj);
        let mut shift = 0.0;
        let mut fxh = 1.0;
        let mut ringf = 1.0;
        let mut fpi = 1.0;
        let mut fheavy = 1.0;
        let mut fcn = 1.0;
        let mut fsrb2 = k::SRB2;
        let (mi, mj) = (s.imetal[ii], s.imetal[jj]);
        let mut bridge = false;
        let mut btyp = 1;
        if hyb[ii] == 2 && hyb[jj] == 2 {
            btyp = 2;
        }
        if hyb[ii] == 3 && hyb[jj] == 2 && ia == 7 {
            btyp = 2;
        }
        if hyb[jj] == 3 && hyb[ii] == 2 && ja == 7 {
            btyp = 2;
        }
        if hyb[ii] == 1 || hyb[jj] == 1 {
            btyp = 3;
        }
        if (gi == 7 || ia == 1) && hyb[ii] == 1 {
            btyp = 3;
            bridge = true;
        }
        if (gj == 7 || ja == 1) && hyb[jj] == 1 {
            btyp = 3;
            bridge = true;
        }
        if hyb[ii] == 5 || hyb[jj] == 5 {
            btyp = 4;
        }
        if mi > 0 || mj > 0 {
            btyp = 5;
        }
        if mi == 2 && mj == 2 {
            btyp = 7;
        }
        if mj == 2 && s.nb.itag[ii] == -1 && pi[ii] {
            btyp = 6;
        }
        if mi == 2 && s.nb.itag[jj] == -1 && pi[jj] {
            btyp = 6;
        }
        let mut bbtyp = btyp;
        let fqq;
        let mut bstrength;
        if bbtyp < 5 {
            let hybi = hyb[ii].max(hyb[jj]);
            let hybj = hyb[ii].min(hyb[jj]);
            bstrength = if hybi == 5 || hybj == 5 {
                k::BSTREN[3]
            } else {
                k::bsmat(hybi as usize, hybj as usize).unwrap_or(-999.0)
            };
            if hybi == 3 && hybj == 2 && (ia == 7 || ja == 7) {
                bstrength = k::BSTREN[1] * 1.04;
            }
            if bridge {
                if gi == 7 {
                    bstrength = k::BSTREN[0] * 0.50;
                }
                if gj == 7 {
                    bstrength = k::BSTREN[0] * 0.50;
                }
                if ia == 1 || ia == 9 {
                    bstrength = k::BSTREN[0] * 0.30;
                }
                if ja == 1 || ja == 9 {
                    bstrength = k::BSTREN[0] * 0.30;
                }
            }
            if bbtyp == 4 {
                shift = k::HYPER_SHIFT;
            }
            if ia == 1 || ja == 1 {
                shift = k::RABSHIFTH;
            }
            if ia == 9 && ja == 9 {
                shift = 0.22;
            }
            if hyb[ii] == 3 && hyb[jj] == 0 {
                shift -= 0.022;
            }
            if hyb[ii] == 0 && hyb[jj] == 3 {
                shift -= 0.022;
            }
            if hyb[ii] == 1 && hyb[jj] == 0 {
                shift += 0.14;
            }
            if hyb[ii] == 0 && hyb[jj] == 1 {
                shift += 0.14;
            }
            if ia == 1 && ja == 6 {
                if ringsatom(&s.rings, jj) == 3 {
                    fxh = 1.05;
                }
                if ctype(s, pi, jj) == 1 {
                    fxh = 0.95;
                }
            }
            if ia == 6 && ja == 1 {
                if ringsatom(&s.rings, ii) == 3 {
                    fxh = 1.05;
                }
                if ctype(s, pi, ii) == 1 {
                    fxh = 0.95;
                }
            }
            if (ia == 1 && ja == 5) || (ja == 1 && ia == 5) {
                fxh = 1.10;
            }
            if (ia == 1 && ja == 7) || (ja == 1 && ia == 7) {
                fxh = 1.06;
            }
            if (ia == 1 && ja == 8) || (ja == 1 && ia == 8) {
                fxh = 0.93;
            }
            if bbtyp == 3 && ((ia == 6 && ja == 8) || (ia == 8 && ja == 6)) {
                bstrength = k::BSTREN[2] * 0.90;
            }
            if bbtyp == 3 && (hyb[ii] == 0 || hyb[jj] == 0) {
                bbtyp = 1;
            }
            if bbtyp == 3 && (hyb[ii] == 3 || hyb[jj] == 3) {
                bbtyp = 1;
            }
            if bbtyp == 3 && (hyb[ii] == 2 || hyb[jj] == 2) {
                bbtyp = 2;
            }
            if pibo[b] > 0.0 {
                shift = k::HUECKELP * (k::BZREF - pibo[b]);
                if bbtyp != 3 && pibo[b] > 0.1 {
                    btyp = 2;
                    bbtyp = 2;
                }
                fpi = 1.0 - k::HUECKELP2 * (k::BZREF2 - pibo[b]);
            }
            if ia > 10 && ja > 10 {
                fcn /= 1.0 + 0.007 * (nni as f64).powi(2);
                fcn /= 1.0 + 0.007 * (nnj as f64).powi(2);
            }
            let qafac = s.qa[ii] * s.qa[jj] * 70.0;
            fqq = 1.0 + k::QFACBM0 / (1.0 + (15.0 * qafac).exp());
        } else {
            // A bond to a metal.
            shift = 0.0;
            bstrength = k::BSTREN[bbtyp as usize - 1];
            if bbtyp == 7 {
                // Two transition metals: 4d/5d pairs and 3d with 4d/5d have their own
                // strengths, weakened by the metallic character.
                let (ri, rj) = (row6(ia), row6(ja));
                if ri > 4 && rj > 4 {
                    bstrength = k::BSTREN[7];
                }
                if (ri == 4 && rj > 4) || (rj == 4 && ri > 4) {
                    bstrength = k::BSTREN[8];
                }
                let dum = (2.0 * s.mchar[ii] + 2.0 * s.mchar[jj]).min(0.5);
                bstrength *= 1.0 - dum;
            }
            let mtyp = |g: i32, m: i32| {
                if m == 2 {
                    4
                } else if g == 1 {
                    1
                } else if g == 2 {
                    2
                } else if g > 2 && m == 1 {
                    3
                } else {
                    0
                }
            };
            let (mtyp1, mtyp2) = (mtyp(gi, mi), mtyp(gj, mj));
            let qafac = s.qa[ii] * s.qa[jj] * 25.0;
            let dum = 1.0 / (1.0 + (15.0 * qafac).exp());
            fqq = 1.0 + dum * (k::QFACBM[mtyp1] + k::QFACBM[mtyp2]) * 0.5;
            // Ligands of a transition metal: heavy, P, chalcogen, halogen.
            for (m, z) in [(mi, ja), (mj, ia)] {
                if m == 2 && z > 10 {
                    fheavy = 0.65;
                }
            }
            for (m, z) in [(mi, ja), (mj, ia)] {
                if m == 2 && z == 15 {
                    fheavy = 1.60;
                }
            }
            for (m, g) in [(mi, gj), (mj, gi)] {
                if m == 2 && g == 6 {
                    fheavy = 0.85;
                }
            }
            for (m, g) in [(mi, gj), (mj, gi)] {
                if m == 2 && g == 7 {
                    fheavy = 1.30;
                }
            }
            // Hydrogen on a 3d/4d or a 5d metal.
            for (m, z, metal_z) in [(mi, ja, ia), (mj, ia, ja)] {
                if m == 2 && z == 1 {
                    fxh = if row6(metal_z) <= 5 { 0.80 } else { 1.00 };
                }
            }
            if (mi == 1 && ja == 1) || (mj == 1 && ia == 1) {
                fxh = 1.20;
            }
            // CO, CN, and NC on a transition metal.
            for (m, ligand, z, nnl) in [(mj, ii, ia, nni), (mi, jj, ja, nnj)] {
                if m == 2 && hyb[ligand] == 1 {
                    if z == 6 {
                        fpi = 1.5;
                        shift = -0.45;
                    }
                    if z == 7 && nnl != 1 {
                        fpi = 0.4;
                        shift = 0.47;
                    }
                }
            }
            if mi == 2 {
                shift += k::METAL2_SHIFT;
            }
            if mj == 2 {
                shift += k::METAL2_SHIFT;
            }
            if mi == 1 && gi <= 2 {
                shift += k::METAL1_SHIFT;
            }
            if mj == 1 && gj <= 2 {
                shift += k::METAL1_SHIFT;
            }
            if mtyp1 == 3 {
                shift += k::METAL3_SHIFT;
            }
            if mtyp2 == 3 {
                shift += k::METAL3_SHIFT;
            }
            // Eta coordination to a transition metal.
            if bbtyp == 6 && s.el[ii].metal() == 2 {
                shift += k::ETA_SHIFT * nni as f64;
            }
            if bbtyp == 6 && s.el[jj].metal() == 2 {
                shift += k::ETA_SHIFT * nnj as f64;
            }
            for (m, nnx) in [(mtyp1, nni), (mtyp2, nnj)] {
                let c = match m {
                    1 | 2 => 0.100,
                    3 => 0.030,
                    4 => 0.036,
                    _ => 0.0,
                };
                fcn /= 1.0 + c * (nnx as f64).powi(2);
            }
            fsrb2 = if mtyp1 == 4 || mtyp2 == 4 {
                -k::SRB2 * 0.22
            } else {
                k::SRB2 * 0.28
            };
        }
        let _ = bbtyp;
        if ia > 10 && ja > 10 {
            shift += k::HSHIFT3;
            if ia > 18 {
                shift += k::HSHIFT4;
            }
            if ja > 18 {
                shift += k::HSHIFT4;
            }
            if ia > 36 {
                shift += k::HSHIFT5;
            }
            if ja > 36 {
                shift += k::HSHIFT5;
            }
        }
        if rings > 0 {
            ringf = 1.0 + k::FRINGBO * (6.0 - rings as f64).powi(2);
        }
        let den = s.el[ii].raw.en - s.el[jj].raw.en;
        let alpha = k::SRB1 * (1.0 + fsrb2 * den.powi(2) + k::SRB3 * bstrength);
        let prefactor = -s.el[ii].raw.bond
            * s.el[jj].raw.bond
            * ringf
            * bstrength
            * fqq
            * fheavy
            * fpi
            * fxh
            * fcn;
        out.push(BondParams {
            shift: k::RABSHIFT + shift,
            alpha,
            prefactor,
            btyp,
        });
    }
    out
}
