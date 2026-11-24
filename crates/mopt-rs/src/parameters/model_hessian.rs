//! Parameters of the model Hessians.
//!
//! - Lindh models: R. Lindh, A. Bernhardsson, G. Karlstrom, P.-A. Malmqvist, Chem. Phys. Lett.
//!   1995, 241, 423 (1995 tables) and the 2007 tables, both as distributed with xtb
//!   (`src/param/model_hessian.f90`, LGPL-3.0, Grimme group), together with xtb's D2 data for
//!   model Hessians.
//! - Covalent radii (Pyykko & Atsumi, Chem. Eur. J. 2009, 15, 188) with metals decreased by 10%,
//!   as in xtb (`covalent_radius_2009`), used by the Swart model.
//! - Schlegel stretch parameters: J. M. Wittbrodt, H. B. Schlegel, J. Mol. Struct. (Theochem)
//!   1997, 398-399, 55 (update of H. B. Schlegel, Theor. Chim. Acta 1984, 66, 333).
//!
//! MultiOptPy: `ModelHessian/calc_params.py` and the constants of `ModelHessian/*.py`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use crate::domain::{Element, Units};

/// Lindh 1995 reference distances `r_ref` (Bohr) by row pair (H-He, Li-Ne, beyond).
pub const LINDH_1995_RAV: [[f64; 3]; 3] =
    [[1.35, 2.10, 2.53], [2.10, 2.87, 3.40], [2.53, 3.40, 3.40]];
/// Lindh 1995 decay exponents `alpha` (1/Bohr^2).
pub const LINDH_1995_AAV: [[f64; 3]; 3] = [
    [1.0000, 0.3949, 0.3949],
    [0.3949, 0.2800, 0.2800],
    [0.3949, 0.2800, 0.2800],
];
/// Lindh 2007 reference distances (Bohr).
pub const LINDH_2007_RAV: [[f64; 3]; 3] =
    [[1.35, 2.10, 2.53], [2.10, 2.87, 3.80], [2.53, 3.80, 4.50]];
/// Lindh 2007 decay exponents (1/Bohr^2).
pub const LINDH_2007_AAV: [[f64; 3]; 3] = [
    [1.0000, 0.3949, 0.3949],
    [0.3949, 0.2800, 0.1200],
    [0.3949, 0.1200, 0.0600],
];
/// Lindh 2007 van der Waals distances `d_ref` (Bohr) of the dispersion-like decay term.
pub const LINDH_2007_DAV: [[f64; 3]; 3] = [[0.0, 3.6, 3.6], [3.6, 5.3, 5.3], [3.6, 5.3, 5.3]];

/// Row index of the Lindh tables (xtb `itabrow - 1`: H-He 0, Li-Ne 1, beyond 2).
pub fn lindh_row(el: Element) -> usize {
    match el.atomic_number() {
        0..=2 => 0,
        3..=10 => 1,
        _ => 2,
    }
}

/// Damping steepness of the D2 term in model Hessians (xtb `d2_damping`).
pub const MH_D2_DAMPING: f64 = 20.0;

/// van der Waals radii (Angstrom) of the D2 term in model Hessians, `Z = 1..=86` (xtb
/// `d2_vander`).
const MH_D2_VANDER_ANG: [f64; 86] = [
    0.91, 0.92, 0.75, 1.28, 1.35, 1.32, 1.27, 1.22, 1.17, 1.13, 1.04, 1.24, 1.49, 1.56, 1.55, 1.53,
    1.49, 1.45, 1.35, 1.34, 1.42, 1.42, 1.42, 1.42, 1.42, 1.42, 1.42, 1.42, 1.42, 1.42, 1.50, 1.57,
    1.60, 1.61, 1.59, 1.57, 1.48, 1.46, 1.49, 1.49, 1.49, 1.49, 1.49, 1.49, 1.49, 1.49, 1.49, 1.49,
    1.52, 1.64, 1.71, 1.72, 1.72, 1.71, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00,
    2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00,
    2.00, 2.00, 2.00, 2.00, 2.00, 2.00,
];

/// C6 coefficients of the D2 term in model Hessians (Grimme 2006 values in J nm^6 / mol,
/// used without conversion as in xtb `d2_c6`), `Z = 1..=86`.
const MH_D2_C6: [f64; 86] = [
    0.14, 0.08, 1.61, 1.61, 3.13, 1.75, 1.23, 0.70, 0.75, 0.63, 5.71, 5.71, 10.79, 9.23, 7.84,
    5.57, 5.07, 4.61, 10.80, 10.80, 10.80, 10.80, 10.80, 10.80, 10.80, 10.80, 10.80, 10.80, 10.80,
    10.80, 16.99, 17.10, 16.37, 12.64, 12.47, 12.01, 24.67, 24.67, 24.67, 24.67, 24.67, 24.67,
    24.67, 24.67, 24.67, 24.67, 24.67, 24.67, 37.32, 38.71, 38.44, 31.74, 31.50, 29.99, 50.00,
    50.00, 50.00, 50.00, 50.00, 50.00, 50.00, 50.00, 50.00, 50.00, 50.00, 50.00, 50.00, 50.00,
    50.00, 50.00, 50.00, 50.00, 50.00, 50.00, 50.00, 50.00, 50.00, 50.00, 50.00, 50.00, 50.00,
    50.00, 50.00, 50.00, 50.00, 50.00,
];

fn z_index(el: Element, len: usize) -> usize {
    (el.atomic_number() as usize).clamp(1, len) - 1
}

/// D2 van der Waals radius of the model-Hessian dispersion term (Bohr).
pub fn mh_d2_vdw_radius_bohr(el: Element) -> f64 {
    MH_D2_VANDER_ANG[z_index(el, 86)] * Units::ANGSTROM_TO_BOHR
}

/// D2 C6 coefficient of the model-Hessian dispersion term (xtb units).
pub fn mh_d2_c6(el: Element) -> f64 {
    MH_D2_C6[z_index(el, 86)]
}

/// Pyykko-Atsumi 2009 covalent radii (Angstrom) with metals decreased by 10%, `Z = 1..=118`
/// (xtb `covalent_radius_2009`).
const COV_2009_ANG: [f64; 118] = [
    0.32, 0.46, 1.20, 0.94, 0.77, 0.75, 0.71, 0.63, 0.64, 0.67, 1.40, 1.25, 1.13, 1.04, 1.10, 1.02,
    0.99, 0.96, 1.76, 1.54, 1.33, 1.22, 1.21, 1.10, 1.07, 1.04, 1.00, 0.99, 1.01, 1.09, 1.12, 1.09,
    1.15, 1.10, 1.14, 1.17, 1.89, 1.67, 1.47, 1.39, 1.32, 1.24, 1.15, 1.13, 1.13, 1.08, 1.15, 1.23,
    1.28, 1.26, 1.26, 1.23, 1.32, 1.31, 2.09, 1.76, 1.62, 1.47, 1.58, 1.57, 1.56, 1.55, 1.51, 1.52,
    1.51, 1.50, 1.49, 1.49, 1.48, 1.53, 1.46, 1.37, 1.31, 1.23, 1.18, 1.16, 1.11, 1.12, 1.13, 1.32,
    1.30, 1.30, 1.36, 1.31, 1.38, 1.42, 2.01, 1.81, 1.67, 1.58, 1.52, 1.53, 1.54, 1.55, 1.49, 1.49,
    1.51, 1.51, 1.48, 1.50, 1.56, 1.58, 1.45, 1.41, 1.34, 1.29, 1.27, 1.21, 1.16, 1.15, 1.09, 1.22,
    1.36, 1.43, 1.46, 1.58, 1.48, 1.57,
];

/// Covalent radius of xtb's model Hessians (Bohr).
pub fn covalent_radius_2009_bohr(el: Element) -> f64 {
    COV_2009_ANG[z_index(el, 118)] * Units::ANGSTROM_TO_BOHR
}

/// Wittbrodt-Schlegel stretch parameters `B` (Bohr) by period pair (periods 1-6; heavier
/// elements use period 6).
pub const SCHLEGEL_B: [[f64; 6]; 6] = [
    [-0.2573, 0.3401, 0.6937, 0.7126, 0.8355, 0.9491],
    [0.3401, 0.9652, 1.2843, 1.4725, 1.6549, 1.7190],
    [0.6937, 1.2843, 1.6925, 1.8238, 2.1164, 2.3185],
    [0.7126, 1.4725, 1.8238, 2.0203, 2.2137, 2.5206],
    [0.8355, 1.6549, 2.1164, 2.2137, 2.3718, 2.5110],
    [0.9491, 1.7190, 2.3185, 2.5206, 2.5110, 2.5110],
];

/// Period of an element (1-based; dummy atoms count as period 1).
pub fn period(el: Element) -> usize {
    match el.atomic_number() {
        0..=2 => 1,
        3..=10 => 2,
        11..=18 => 3,
        19..=36 => 4,
        37..=54 => 5,
        55..=86 => 6,
        _ => 7,
    }
}
