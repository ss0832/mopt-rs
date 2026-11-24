//! Scalar root finding.
//!
//! MultiOptPy: no counterpart (new in mopt-rs) (MultiOptPy uses scipy root finding).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use crate::error::{MoptError, MoptResult};
use faer::Mat;

/// Imaginary part (relative to the size of the root) below which a root counts as real.
const REAL_ROOT_TOLERANCE: f64 = 1e-10;

/// The real roots of the polynomial `c[0] + c[1] x + ... + c[d] x^d`, from the eigenvalues
/// of its companion matrix (as `numpy.roots` finds them). Leading zero coefficients are
/// dropped; a constant has no roots.
pub fn polynomial_real_roots(coefficients: &[f64]) -> MoptResult<Vec<f64>> {
    let mut degree = coefficients.len().saturating_sub(1);
    while degree > 0 && coefficients[degree] == 0.0 {
        degree -= 1;
    }
    if degree == 0 {
        return Ok(Vec::new());
    }
    if coefficients[..=degree].iter().any(|c| !c.is_finite()) {
        return Err(MoptError::linalg("polynomial with non-finite coefficients"));
    }
    let lead = coefficients[degree];
    // Companion matrix: ones below the diagonal, minus the scaled coefficients in the
    // last column.
    let companion = Mat::from_fn(degree, degree, |i, j| {
        if j == degree - 1 {
            -coefficients[i] / lead
        } else if i == j + 1 {
            1.0
        } else {
            0.0
        }
    });
    let evd = companion
        .eigen()
        .map_err(|e| MoptError::linalg(format!("roots of a polynomial: {e:?}")))?;
    let values = evd.S();
    let mut roots: Vec<f64> = (0..degree)
        .filter(|&k| values[k].im.abs() <= REAL_ROOT_TOLERANCE * values[k].re.abs().max(1.0))
        .map(|k| values[k].re)
        .collect();
    roots.sort_by(f64::total_cmp);
    Ok(roots)
}

/// Brent's method on a bracketing interval `[a, b]` (`f(a) f(b) < 0`).
///
/// Mirrors `scipy.optimize.brentq`: terminates when the bracket half-width falls below
/// `xtol + rtol * |x|` or `f(x) == 0`; errors after `max_iter` iterations.
pub fn brentq<F: FnMut(f64) -> f64>(
    mut f: F,
    a: f64,
    b: f64,
    xtol: f64,
    rtol: f64,
    max_iter: usize,
) -> MoptResult<f64> {
    let (mut xpre, mut xcur) = (a, b);
    let (mut fpre, mut fcur) = (f(xpre), f(xcur));
    if fpre == 0.0 {
        return Ok(xpre);
    }
    if fcur == 0.0 {
        return Ok(xcur);
    }
    if fpre.signum() == fcur.signum() {
        return Err(MoptError::conv(
            "brentq: f(a) and f(b) must have opposite signs",
        ));
    }
    let (mut xblk, mut fblk, mut spre, mut scur) = (0.0, 0.0, 0.0, 0.0);
    for _ in 0..max_iter {
        if fpre != 0.0 && fcur != 0.0 && fpre.signum() != fcur.signum() {
            xblk = xpre;
            fblk = fpre;
            spre = xcur - xpre;
            scur = spre;
        }
        if fblk.abs() < fcur.abs() {
            xpre = xcur;
            xcur = xblk;
            xblk = xpre;
            fpre = fcur;
            fcur = fblk;
            fblk = fpre;
        }
        let delta = (xtol + rtol * xcur.abs()) / 2.0;
        let sbis = (xblk - xcur) / 2.0;
        if fcur == 0.0 || sbis.abs() < delta {
            return Ok(xcur);
        }
        if spre.abs() > delta && fcur.abs() < fpre.abs() {
            let stry = if xpre == xblk {
                // secant
                -fcur * (xcur - xpre) / (fcur - fpre)
            } else {
                // inverse quadratic interpolation
                let dpre = (fpre - fcur) / (xpre - xcur);
                let dblk = (fblk - fcur) / (xblk - xcur);
                -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre))
            };
            if 2.0 * stry.abs() < spre.abs().min(3.0 * sbis.abs() - delta) {
                spre = scur;
                scur = stry;
            } else {
                spre = sbis;
                scur = sbis;
            }
        } else {
            spre = sbis;
            scur = sbis;
        }
        xpre = xcur;
        fpre = fcur;
        if scur.abs() > delta {
            xcur += scur;
        } else {
            xcur += if sbis > 0.0 { delta } else { -delta };
        }
        fcur = f(xcur);
    }
    Err(MoptError::conv(format!(
        "brentq did not converge in {max_iter} iterations"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn real_roots_of_polynomials() {
        // (x - 1)(x + 2)(x^2 + 1) = x^4 + x^3 - x^2 + x - 2.
        let roots = polynomial_real_roots(&[-2.0, 1.0, -1.0, 1.0, 1.0, 0.0]).unwrap();
        assert_eq!(roots.len(), 2);
        assert!((roots[0] + 2.0).abs() < 1e-12 && (roots[1] - 1.0).abs() < 1e-12);
        assert!(polynomial_real_roots(&[3.0]).unwrap().is_empty());
        assert_eq!(polynomial_real_roots(&[1.0, 2.0]).unwrap(), vec![-0.5]);
        assert!(polynomial_real_roots(&[1.0, f64::NAN]).is_err());
    }

    #[test]
    fn finds_roots() {
        let r = brentq(|x| x * x - 2.0, 0.0, 2.0, 1e-14, 1e-14, 100).unwrap();
        assert!((r - 2f64.sqrt()).abs() < 1e-12);
        let r = brentq(|x: f64| x.cos() - x, 0.0, 1.0, 1e-12, 1e-12, 100).unwrap();
        assert!((r.cos() - r).abs() < 1e-11);
        assert!(brentq(|x| x * x + 1.0, -1.0, 1.0, 1e-12, 1e-12, 100).is_err());
    }
}
