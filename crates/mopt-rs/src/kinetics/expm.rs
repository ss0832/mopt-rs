//! The matrix exponential by scaling and squaring with Pade approximants of degree 3 to 13
//! (Higham, SIAM J. Matrix Anal. Appl. 2005, 26, 1179; Moler, Van Loan, SIAM Rev. 2003, 45, 3),
//! and the block forms built on it: the time integral of `exp(s A)` and the integrals of
//! Van Loan (IEEE Trans. Autom. Control 1978, 23, 395) that give derivatives.
//!
//! For a matrix with non-negative off-diagonal entries (a generator, its transpose, the block
//! matrices of the integrals) [`expm_metzler`] multiplies non-negative numbers only: with
//! `lambda = 2 max |a_ii|` and `P = I + a / lambda >= 0` (diagonal at least 1/2),
//! `exp(tau a) = sum_k exp(-x) x^k / k! P^k` (`x = lambda tau <= 1`, uniformization), then the
//! squarings `exp(2 tau a) = exp(tau a)^2`. For a generator the columns of `exp(tau a)` sum to
//! 1, and a diagonal entry that holds most of its column is rebuilt as 1 minus the sum of the
//! off-diagonal ones after every squaring: the slow outflow of a state is then carried by the
//! off-diagonal entries, which have a relative error of a few hundred rounding errors however
//! small they are. Without that, a diagonal near 1 keeps the outflow only to an absolute
//! rounding error, and the squarings double its error each time. Measured on a network of
//! the golden data with rates from 1e-2 to 1e9 1/s (populations at 1 s and 100 s against a
//! 90-digit reference): Pade 3e-9 and 5e-8, the plain non-negative form 1e-8, with the
//! rebuilt diagonals below 1e-12.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Mat;

use crate::error::{MoptError, MoptResult};
use crate::math::ad::Scalar;

const B3: [f64; 4] = [120.0, 60.0, 12.0, 1.0];
const B5: [f64; 6] = [30240.0, 15120.0, 3360.0, 420.0, 30.0, 1.0];
const B7: [f64; 8] = [
    17_297_280.0,
    8_648_640.0,
    1_995_840.0,
    277_200.0,
    25_200.0,
    1512.0,
    56.0,
    1.0,
];
const B9: [f64; 10] = [
    17_643_225_600.0,
    8_821_612_800.0,
    2_075_673_600.0,
    302_702_400.0,
    30_270_240.0,
    2_162_160.0,
    110_880.0,
    3960.0,
    90.0,
    1.0,
];
const B13: [f64; 14] = [
    64_764_752_532_480_000.0,
    32_382_376_266_240_000.0,
    7_771_770_303_897_600.0,
    1_187_353_796_428_800.0,
    129_060_195_264_000.0,
    10_559_470_521_600.0,
    670_442_572_800.0,
    33_522_128_640.0,
    1_323_241_920.0,
    40_840_800.0,
    960_960.0,
    16380.0,
    182.0,
    1.0,
];
/// Largest 1-norms for which the Pade approximants of degree 3, 5, 7, 9, 13 are accurate to
/// double precision (Higham 2005, Table 2.3).
const THETA: [f64; 5] = [
    1.495_585_217_958_292e-2,
    2.539_398_330_063_23e-1,
    9.504_178_996_162_932e-1,
    2.097_847_961_257_068,
    5.371_920_351_148_152,
];

/// The 1-norm (largest absolute column sum).
pub fn norm1(a: &Mat<f64>) -> f64 {
    (0..a.ncols())
        .map(|j| (0..a.nrows()).map(|i| a[(i, j)].abs()).sum::<f64>())
        .fold(0.0, f64::max)
}

fn add_scaled(target: &mut Mat<f64>, a: &Mat<f64>, s: f64) {
    for j in 0..a.ncols() {
        for i in 0..a.nrows() {
            target[(i, j)] += s * a[(i, j)];
        }
    }
}

fn pade_low(a: &Mat<f64>, b: &[f64]) -> (Mat<f64>, Mat<f64>) {
    let n = a.nrows();
    let a2 = a * a;
    let mut powers = vec![Mat::<f64>::identity(n, n), a2.clone()];
    while powers.len() * 2 < b.len() {
        let next = powers.last().map_or_else(|| a2.clone(), |p| p * &a2);
        powers.push(next);
    }
    let mut u = Mat::<f64>::zeros(n, n);
    let mut v = Mat::<f64>::zeros(n, n);
    for (k, p) in powers.iter().enumerate() {
        if 2 * k + 1 < b.len() {
            add_scaled(&mut u, p, b[2 * k + 1]);
        }
        add_scaled(&mut v, p, b[2 * k]);
    }
    (a * &u, v)
}

fn pade13(a: &Mat<f64>) -> (Mat<f64>, Mat<f64>) {
    let n = a.nrows();
    let b = &B13;
    let a2 = a * a;
    let a4 = &a2 * &a2;
    let a6 = &a4 * &a2;
    let id = Mat::<f64>::identity(n, n);
    let mut inner_u = Mat::<f64>::zeros(n, n);
    add_scaled(&mut inner_u, &a6, b[13]);
    add_scaled(&mut inner_u, &a4, b[11]);
    add_scaled(&mut inner_u, &a2, b[9]);
    let mut u = &a6 * &inner_u;
    add_scaled(&mut u, &a6, b[7]);
    add_scaled(&mut u, &a4, b[5]);
    add_scaled(&mut u, &a2, b[3]);
    add_scaled(&mut u, &id, b[1]);
    let u = a * &u;
    let mut inner_v = Mat::<f64>::zeros(n, n);
    add_scaled(&mut inner_v, &a6, b[12]);
    add_scaled(&mut inner_v, &a4, b[10]);
    add_scaled(&mut inner_v, &a2, b[8]);
    let mut v = &a6 * &inner_v;
    add_scaled(&mut v, &a6, b[6]);
    add_scaled(&mut v, &a4, b[4]);
    add_scaled(&mut v, &a2, b[2]);
    add_scaled(&mut v, &id, b[0]);
    (u, v)
}

/// `exp(a)` of a square matrix.
pub fn expm(a: &Mat<f64>) -> MoptResult<Mat<f64>> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err(MoptError::linalg("expm needs a square matrix"));
    }
    if n == 0 {
        return Ok(Mat::zeros(0, 0));
    }
    let norm = norm1(a);
    if !norm.is_finite() {
        return Err(MoptError::linalg(
            "expm: the matrix has entries that are not finite",
        ));
    }
    let lows: [&[f64]; 4] = [&B3, &B5, &B7, &B9];
    for (k, b) in lows.iter().enumerate() {
        if norm <= THETA[k] {
            let (u, v) = pade_low(a, b);
            return solve_pade(&u, &v);
        }
    }
    let s = if norm > THETA[4] {
        (norm / THETA[4]).log2().ceil().max(0.0) as i32
    } else {
        0
    };
    let scale = 0.5_f64.powi(s);
    let scaled = Mat::from_fn(n, n, |i, j| a[(i, j)] * scale);
    let (u, v) = pade13(&scaled);
    let mut r = solve_pade(&u, &v)?;
    for _ in 0..s {
        r = &r * &r;
    }
    Ok(r)
}

/// `(v - u)^-1 (v + u)`.
fn solve_pade(u: &Mat<f64>, v: &Mat<f64>) -> MoptResult<Mat<f64>> {
    let n = u.nrows();
    let q = Mat::from_fn(n, n, |i, j| v[(i, j)] - u[(i, j)]);
    let p = Mat::from_fn(n, n, |i, j| v[(i, j)] + u[(i, j)]);
    let lu = q.partial_piv_lu();
    let r = faer::linalg::solvers::Solve::solve(&lu, p.as_ref());
    if (0..n).any(|j| (0..n).any(|i| !r[(i, j)].is_finite())) {
        return Err(MoptError::linalg(
            "expm: the denominator of the Pade approximant is singular",
        ));
    }
    Ok(r)
}

/// Terms of the uniformized series (`x <= 1`: the Poisson tail beyond them is below 1e-25).
const SERIES_TERMS: usize = 26;

/// Check that `a` is square with non-negative off-diagonal entries.
fn check_metzler(a: &Mat<f64>) -> MoptResult<f64> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err(MoptError::linalg("expm: the matrix must be square"));
    }
    let mut diag: f64 = 0.0;
    for j in 0..n {
        for i in 0..n {
            let v = a[(i, j)];
            if !v.is_finite() {
                return Err(MoptError::linalg(
                    "expm: the matrix has entries that are not finite",
                ));
            }
            if i == j {
                diag = diag.max(v.abs());
            } else if v < 0.0 {
                return Err(MoptError::linalg(
                    "expm_metzler: an off-diagonal entry is negative",
                ));
            }
        }
    }
    Ok(diag)
}

/// The uniformization of `a` for the time `t`: `(P, x, squarings, lambda)` with
/// `P = I + a / lambda` and `x = lambda t / 2^squarings <= 1`.
fn uniformized(a: &Mat<f64>, t: f64) -> MoptResult<(Mat<f64>, f64, i32, f64)> {
    let n = a.nrows();
    let diag = check_metzler(a)?;
    let lambda = 2.0 * diag.max(f64::MIN_POSITIVE);
    let total = lambda * t;
    let s = if total > 1.0 {
        total.log2().ceil() as i32
    } else {
        0
    };
    let x = total * 0.5_f64.powi(s);
    let p = Mat::from_fn(n, n, |i, j| {
        let v = a[(i, j)] / lambda;
        if i == j {
            1.0 + v
        } else {
            v
        }
    });
    Ok((p, x, s, lambda))
}

/// The Poisson weights `exp(-x) x^k / k!` for `k < SERIES_TERMS`.
fn poisson(x: f64) -> Vec<f64> {
    let mut w = Vec::with_capacity(SERIES_TERMS);
    let mut v = (-x).exp();
    for k in 0..SERIES_TERMS {
        w.push(v);
        v *= x / (k + 1) as f64;
    }
    w
}

/// `sum_k c_k P^k` by Horner's scheme (non-negative coefficients and matrix).
fn series(p: &Mat<f64>, c: &[f64]) -> Mat<f64> {
    let n = p.nrows();
    let mut s = Mat::from_fn(n, n, |i, j| if i == j { c[c.len() - 1] } else { 0.0 });
    for k in (0..c.len() - 1).rev() {
        s = p * &s;
        for i in 0..n {
            s[(i, i)] += c[k];
        }
    }
    s
}

/// Which sums of `exp(t a)` are 1 exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Conserved {
    /// The columns (`a` a generator: zero column sums).
    Columns,
    /// The rows (zero row sums: the transpose of a generator).
    Rows,
    /// Neither.
    Nothing,
}

fn conserved(a: &Mat<f64>) -> Conserved {
    let n = a.nrows();
    let zero = |sum: f64, d: f64| sum.abs() <= 1e-12 * d.abs();
    if (0..n).all(|j| zero((0..n).map(|i| a[(i, j)]).sum(), a[(j, j)])) {
        Conserved::Columns
    } else if (0..n).all(|i| zero((0..n).map(|j| a[(i, j)]).sum(), a[(i, i)])) {
        Conserved::Rows
    } else {
        Conserved::Nothing
    }
}

/// Where a column (row) of a stochastic `e` keeps at least half of its mass on the diagonal,
/// the diagonal is rebuilt as 1 minus the off-diagonal sum: the small outflow then keeps its
/// relative accuracy through the squarings (a diagonal near 1 from a product would carry it
/// only to an absolute rounding error).
fn rebuild(e: &mut Mat<f64>, c: Conserved) {
    let n = e.nrows();
    for j in 0..n {
        let out: f64 = match c {
            Conserved::Columns => (0..n).filter(|&i| i != j).map(|i| e[(i, j)]).sum(),
            Conserved::Rows => (0..n).filter(|&i| i != j).map(|i| e[(j, i)]).sum(),
            Conserved::Nothing => return,
        };
        if out < 0.5 {
            e[(j, j)] = 1.0 - out;
        }
    }
}

fn square(e: &Mat<f64>, c: Conserved) -> Mat<f64> {
    let mut r = e * e;
    rebuild(&mut r, c);
    r
}

/// `exp(t a)` of a matrix with non-negative off-diagonal entries, without subtractions (but
/// the diagonals of a generator, rebuilt from the sums while they hold most of the mass).
pub fn expm_metzler(a: &Mat<f64>, t: f64) -> MoptResult<Mat<f64>> {
    if !(t >= 0.0 && t.is_finite()) {
        return Err(MoptError::linalg("expm: the time must not be negative"));
    }
    let n = a.nrows();
    if n == 0 {
        check_metzler(a)?;
        return Ok(Mat::zeros(0, 0));
    }
    let (p, x, s, _) = uniformized(a, t)?;
    let c = conserved(a);
    let mut e = series(&p, &poisson(x));
    rebuild(&mut e, c);
    for _ in 0..s {
        e = square(&e, c);
    }
    Ok(e)
}

/// `a b` of row-major `n x n` matrices of a scalar.
fn mat_mul_scalar<T: Scalar>(a: &[T], b: &[T], n: usize) -> Vec<T> {
    let zero = T::zero();
    let mut out = vec![zero; n * n];
    for i in 0..n {
        for k in 0..n {
            let aik = a[i * n + k];
            if aik == zero {
                continue;
            }
            for j in 0..n {
                out[i * n + j] += aik * b[k * n + j];
            }
        }
    }
    out
}

fn rebuild_scalar<T: Scalar>(e: &mut [T], n: usize, c: Conserved) {
    for j in 0..n {
        let mut out = T::zero();
        for i in 0..n {
            if i != j {
                out += match c {
                    Conserved::Columns => e[i * n + j],
                    Conserved::Rows => e[j * n + i],
                    Conserved::Nothing => return,
                };
            }
        }
        if out.val() < 0.5 {
            e[j * n + j] = T::one() - out;
        }
    }
}

/// [`expm_metzler`] of a row-major matrix of a scalar (`f64`, or dual numbers for the
/// derivatives): the same arithmetic, with `lambda`, the number of squarings, and which sums
/// are conserved taken from the real parts.
pub fn expm_metzler_scalar<T: Scalar>(a: &[T], n: usize, t: f64) -> MoptResult<Vec<T>> {
    if !(t >= 0.0 && t.is_finite()) || a.len() != n * n {
        return Err(MoptError::linalg(
            "expm: a square matrix and a time that is not negative are needed",
        ));
    }
    let re = Mat::from_fn(n, n, |i, j| a[i * n + j].val());
    let diag = check_metzler(&re)?;
    let c = conserved(&re);
    let lambda = 2.0 * diag.max(f64::MIN_POSITIVE);
    let total = lambda * t;
    let s = if total > 1.0 {
        total.log2().ceil() as i32
    } else {
        0
    };
    let x = total * 0.5_f64.powi(s);
    let inv = T::from_f64(1.0 / lambda);
    let p: Vec<T> = (0..n * n)
        .map(|k| {
            let v = a[k] * inv;
            if k / n == k % n {
                T::one() + v
            } else {
                v
            }
        })
        .collect();
    let w = poisson(x);
    let mut e: Vec<T> = (0..n * n)
        .map(|k| {
            if k / n == k % n {
                T::from_f64(w[w.len() - 1])
            } else {
                T::zero()
            }
        })
        .collect();
    for k in (0..w.len() - 1).rev() {
        e = mat_mul_scalar(&p, &e, n);
        for i in 0..n {
            e[i * n + i] += T::from_f64(w[k]);
        }
    }
    rebuild_scalar(&mut e, n, c);
    for _ in 0..s {
        e = mat_mul_scalar(&e, &e, n);
        rebuild_scalar(&mut e, n, c);
    }
    Ok(e)
}

/// `Pr(N > k)` of a Poisson number with mean `x` over `lambda`, summed from the top so that no
/// difference is formed.
fn tail_coefficients(w: &[f64], lambda: f64) -> Vec<f64> {
    let mut tails = vec![0.0; w.len()];
    let mut acc = 0.0;
    for k in (0..w.len()).rev() {
        tails[k] = acc / lambda;
        acc += w[k];
    }
    tails
}

/// `exp(t a)` and `int_0^t exp(s a) ds` of a matrix with non-negative off-diagonal entries,
/// without subtractions: the integral over the first interval from the uniformized series
/// (`int_0^tau exp(s a) ds = (1 / lambda) sum_k P^k Pr(N > k)`, `N` Poisson with mean `x`),
/// then `I(2 tau) = (1 + E(tau)) I(tau)` with every squaring.
pub fn expm_metzler_integral(a: &Mat<f64>, t: f64) -> MoptResult<(Mat<f64>, Mat<f64>)> {
    if !(t >= 0.0 && t.is_finite()) {
        return Err(MoptError::linalg("expm: the time must not be negative"));
    }
    let n = a.nrows();
    let (p, x, s, lambda) = uniformized(a, t)?;
    let c = conserved(a);
    let w = poisson(x);
    let mut e = series(&p, &w);
    rebuild(&mut e, c);
    let mut integral = if x > 0.0 {
        series(&p, &tail_coefficients(&w, lambda))
    } else {
        Mat::zeros(n, n)
    };
    for _ in 0..s {
        let mut sum = e.clone();
        for i in 0..n {
            sum[(i, i)] += 1.0;
        }
        integral = &sum * &integral;
        e = square(&e, c);
    }
    Ok((e, integral))
}

/// `exp(t a) x` and `int_0^t exp(s a) x ds` for a matrix with non-negative off-diagonal
/// entries.
pub fn exp_and_integral(a: &Mat<f64>, t: f64, x: &[f64]) -> MoptResult<(Vec<f64>, Vec<f64>)> {
    let n = a.nrows();
    let (e, integral) = expm_metzler_integral(a, t)?;
    let apply = |m: &Mat<f64>| -> Vec<f64> {
        (0..n)
            .map(|i| (0..n).map(|j| m[(i, j)] * x[j]).sum())
            .collect()
    };
    Ok((apply(&e), apply(&integral)))
}

/// Van Loan's integral `int_0^t exp((t - s) a) b exp(s c) ds`: the upper-right block of the
/// exponential of `[[a, b], [0, c]] t`. When `a` and `c` have non-negative off-diagonal
/// entries and `b` is not negative, without subtractions: the blocks after the first interval
/// from the uniformized series of the block matrix, then `Z(2 tau) = E_a Z + Z E_c` with
/// every squaring of `E_a`, `E_c`; otherwise by the Pade form.
pub fn van_loan(a: &Mat<f64>, b: &Mat<f64>, c: &Mat<f64>, t: f64) -> MoptResult<Mat<f64>> {
    let n = a.nrows();
    let block = Mat::from_fn(2 * n, 2 * n, |i, j| match (i < n, j < n) {
        (true, true) => a[(i, j)],
        (true, false) => b[(i, j - n)],
        (false, false) => c[(i - n, j - n)],
        (false, true) => 0.0,
    });
    let positive = (0..n).all(|i| (0..n).all(|j| b[(i, j)] >= 0.0));
    let uniform = if positive {
        uniformized(&block, t).ok()
    } else {
        None
    };
    let Some((p, x, s, _)) = uniform else {
        let e = expm(&Mat::from_fn(2 * n, 2 * n, |i, j| block[(i, j)] * t))?;
        return Ok(Mat::from_fn(n, n, |i, j| e[(i, j + n)]));
    };
    let (ca, cc) = (conserved(a), conserved(c));
    let first = series(&p, &poisson(x));
    let mut ea = Mat::from_fn(n, n, |i, j| first[(i, j)]);
    let mut ec = Mat::from_fn(n, n, |i, j| first[(i + n, j + n)]);
    let mut z = Mat::from_fn(n, n, |i, j| first[(i, j + n)]);
    rebuild(&mut ea, ca);
    rebuild(&mut ec, cc);
    for _ in 0..s {
        z = &ea * &z + &z * &ec;
        ea = square(&ea, ca);
        ec = square(&ec, cc);
    }
    Ok(z)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_exponential_of_a_generator_and_of_a_rotation() {
        // A <-> B with rates kf (A -> B) and kr: p_A(t) = pi_A + (1 - pi_A) exp(-(kf + kr) t).
        for (kf, kr, t) in [
            (2.0, 0.5, 0.3),
            (3e8, 1.0, 1e-6),
            (1e-3, 2e-3, 10.0),
            (5.0, 7.0, 100.0),
        ] {
            let k = Mat::from_fn(2, 2, |i, j| match (i, j) {
                (0, 0) => -kf * t,
                (1, 0) => kf * t,
                (0, 1) => kr * t,
                _ => -kr * t,
            });
            let e = expm(&k).unwrap_or_else(|_| Mat::zeros(2, 2));
            let pi_a = kr / (kf + kr);
            let want = pi_a + (1.0 - pi_a) * (-(kf + kr) * t).exp();
            assert!(
                (e[(0, 0)] - want).abs() < 1e-12,
                "{kf} {kr} {t}: {}",
                e[(0, 0)]
            );
            assert!((e[(0, 0)] + e[(1, 0)] - 1.0).abs() < 1e-12);
        }
        // A rotation: exp([[0, -w], [w, 0]]) = [[cos w, -sin w], [sin w, cos w]].
        for w in [1e-3, 0.5, 3.0, 40.0] {
            let a = Mat::from_fn(2, 2, |i, j| match (i, j) {
                (0, 1) => -w,
                (1, 0) => w,
                _ => 0.0,
            });
            let e = expm(&a).unwrap_or_else(|_| Mat::zeros(2, 2));
            assert!((e[(0, 0)] - w.cos()).abs() < 1e-12 * (1.0 + w));
            assert!((e[(1, 0)] - w.sin()).abs() < 1e-12 * (1.0 + w));
        }
        // The integral of exp(-k s) from 0 to t.
        let a = Mat::from_fn(1, 1, |_, _| -2.0);
        let (p, tau) = exp_and_integral(&a, 1.5, &[1.0]).unwrap_or_default();
        assert!((p[0] - (-3.0_f64).exp()).abs() < 1e-14);
        assert!((tau[0] - (1.0 - (-3.0_f64).exp()) / 2.0).abs() < 1e-14);
    }
}
