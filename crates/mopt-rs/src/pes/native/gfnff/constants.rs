//! Fixed GFN-FF constants: the topology generator (`newGFNFFGenerator`) and the potential
//! parameters set in `gfnff_set_param` of xtb (`src/gfnff/gfnff_param.f90`, LGPL-3.0-or-later),
//! with the values of the default version (`angewChem2020_2`), and the accuracy thresholds of
//! `gfnff_thresholds` at the default accuracy 0.1.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(missing_docs)]

// ---- generator (topology and force-constant setup) ----
pub const CNMAX: f64 = 4.4;
pub const LINTHR: f64 = 160.0;
pub const FCTHR: f64 = 1.0e-3;
pub const TDIST_THR: f64 = 12.0;
pub const RTHR: f64 = 1.25;
pub const RTHR2: f64 = 1.00;
pub const RQSHRINK: f64 = 0.23;
pub const HQABTHR: f64 = 0.01;
pub const QABTHR: f64 = 0.10;
pub const SRB1: f64 = 0.3731;
pub const SRB2: f64 = 0.3171;
pub const SRB3: f64 = 0.2538;
pub const QREPSCAL: f64 = 0.3480;
pub const NREPSCAL: f64 = -0.1270;
pub const HHFAC: f64 = 0.6290;
pub const HH13REP: f64 = 1.4580;
pub const HH14REP: f64 = 0.7080;
/// Bond strengths: single, double, triple, hypervalent, M-X, M eta, M-M, M-M, (7+8)/2.
pub const BSTREN: [f64; 9] = [1.00, 1.24, 1.98, 1.22, 1.00, 0.78, 3.40, 3.40, 3.40];
pub const QFACBEN: f64 = -0.54;
pub const QFACTOR: f64 = 12.0;
pub const FR3: f64 = 0.3;
pub const FR4: f64 = 1.0;
pub const FR5: f64 = 1.5;
pub const FR6: f64 = 5.7;
/// `torsf(1..8)`: single, pi, improper, (4 unused), pi part improper, extra sp3 C, N, O.
pub const TORSF: [f64; 8] = [1.00, 1.18, 1.05, 0.0, 0.50, -0.90, 0.70, -2.00];
pub const FBS1: f64 = 0.50;
pub const BATMSCAL: f64 = 0.30;
pub const MCHISHIFT: f64 = -0.09;
pub const RABSHIFT: f64 = -0.110;
pub const RABSHIFTH: f64 = -0.050;
pub const HYPER_SHIFT: f64 = 0.03;
pub const HSHIFT3: f64 = -0.11;
pub const HSHIFT4: f64 = -0.11;
pub const HSHIFT5: f64 = -0.06;
pub const METAL1_SHIFT: f64 = 0.2;
pub const METAL2_SHIFT: f64 = 0.15;
pub const METAL3_SHIFT: f64 = 0.05;
pub const ETA_SHIFT: f64 = 0.040;
/// `qfacbm(0..4)`.
pub const QFACBM: [f64; 5] = [1.0, -0.2, -0.2, 0.70, 0.50];
pub const QFACBM0: f64 = 0.047;
pub const RFGOED1: f64 = 1.175;
pub const HTRIPLE: f64 = 1.45;
pub const HUECKELP2: f64 = 1.00;
pub const HUECKELP3: f64 = -0.24;
pub const HITER: f64 = 0.700;
pub const HUECKELP: f64 = 0.340;
pub const BZREF: f64 = 0.370;
pub const BZREF2: f64 = 0.315;
pub const PILPF: f64 = 0.530;
pub const MAXHITER: usize = 5;
pub const D3A1: f64 = 0.58;
pub const D3A2: f64 = 4.80;
pub const SPLIT0: f64 = 0.670;
pub const SPLIT1: f64 = 1.0 - SPLIT0;
pub const FRINGBO: f64 = 0.020;
pub const AHEAVY3: f64 = 89.0;
pub const AHEAVY4: f64 = 100.0;

/// Hueckel diagonal element relative to C (`hdiag`), by atomic number (0 when not set).
pub fn hdiag(z: u8) -> f64 {
    match z {
        5 => -0.5,
        6 => 0.00,
        7 => 0.14,
        8 => -0.38,
        9 => -0.29,
        16 => -0.30,
        17 => -0.30,
        _ => 0.0,
    }
}

/// Hueckel off-diagonal constant (`hoffdiag`), by atomic number (0 when not set).
pub fn hoffdiag(z: u8) -> f64 {
    match z {
        5 => 0.5,
        6 => 1.00,
        7 => 0.66,
        8 => 1.10,
        9 => 0.23,
        16 => 0.60,
        17 => 1.00,
        _ => 0.0,
    }
}

/// `bsmat(hyb_i, hyb_j)` for hybridizations 0..3 (`None`: -999 in xtb, not defined).
pub fn bsmat(a: usize, b: usize) -> Option<f64> {
    let s = SPLIT0;
    let t = SPLIT1;
    let v = match (a, b) {
        (0, 0) | (3, 0) | (3, 3) => BSTREN[0],
        (2, 2) => BSTREN[1],
        (1, 1) => BSTREN[2],
        (1, 0) | (3, 1) => s * BSTREN[0] + t * BSTREN[2],
        (2, 1) => s * BSTREN[1] + t * BSTREN[2],
        (2, 0) | (3, 2) => s * BSTREN[0] + t * BSTREN[1],
        _ => return None,
    };
    Some(v)
}

// ---- potential parameters (gfnff_set_param) ----
pub const PARAM_CNMAX: f64 = 4.4;
pub const ATCUTA: f64 = 0.595;
pub const ATCUTT: f64 = 0.505;
pub const ATCUTA_NCI: f64 = 0.395;
pub const ATCUTT_NCI: f64 = 0.305;
pub const REPSCALB: f64 = 1.7583;
pub const REPSCALN: f64 = 0.4270;
pub const HBACUT: f64 = 49.0;
pub const HBSCUT: f64 = 22.0;
pub const XBACUT: f64 = 70.0;
pub const XBSCUT: f64 = 5.0;
pub const HBSF: f64 = 1.0;
pub const HBST: f64 = 15.0;
pub const XBSF: f64 = 0.03;
pub const XBST: f64 = 15.0;
pub const HBALP: f64 = 6.0;
pub const HBLONGCUT: f64 = 85.0;
pub const HBLONGCUT_XB: f64 = 70.0;
pub const HBABMIX: f64 = 0.80;
pub const HBNBCUT: f64 = 11.20;
pub const TORS_HB: f64 = 0.94;
pub const BEND_HB: f64 = 0.20;
pub const VBOND_SCALE: f64 = 0.9;
pub const XHACI_GLOBABH: f64 = 0.268;
pub const XHACI_COH: f64 = 0.350;
pub const XHACI_GLOB: f64 = 1.50;

/// HB and XB basicities `xhbas` (0 for elements without one).
pub fn xhbas(z: u8) -> f64 {
    match z {
        6 => 0.80,
        7 => 1.68,
        8 => 0.67,
        9 => 0.52,
        14 => 4.0,
        15 | 33 | 51 => 3.5,
        16 | 34 | 52 => 2.0,
        17 => 1.5,
        35 => 1.5,
        53 => 1.9,
        _ => 0.0,
    }
}

/// HB acidities `xhaci`.
pub fn xhaci(z: u8) -> f64 {
    match z {
        6 => 0.75,
        7 => XHACI_GLOB + 0.1,
        8 | 9 | 15 | 16 => XHACI_GLOB,
        17 | 35 | 53 => XHACI_GLOB + 1.0,
        _ => 0.0,
    }
}

/// XB acidities `xbaci`.
pub fn xbaci(z: u8) -> f64 {
    match z {
        15 | 16 => 1.0,
        17 => 0.5,
        33 | 34 | 51 | 52 | 53 => 1.2,
        35 => 0.9,
        _ => 0.0,
    }
}

/// Bonded three-body (ATM) charge `zb3atm` including the `batmscal^(1/3)` prefactor.
pub fn zb3atm(z: u8) -> f64 {
    if z == 1 {
        -0.25 * BATMSCAL.powf(1.0 / 3.0)
    } else {
        -f64::from(z) * BATMSCAL.powf(1.0 / 3.0)
    }
}

// ---- accuracy thresholds (gfnff_thresholds at xtb's default accuracy 0.1, log10 = -1; squared
// distances in Bohr^2, as xtb prints them) ----
pub const DISPTHR: f64 = 2500.0;
pub const CNTHR: f64 = 150.0;
pub const REPTHR: f64 = 500.0;
pub const HBTHR1: f64 = 250.0;
pub const HBTHR2: f64 = 450.0;
