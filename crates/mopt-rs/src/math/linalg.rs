//! Small dense linear-algebra helpers on top of `faer`.
//!
//! MultiOptPy: no counterpart (new in mopt-rs) (MultiOptPy uses numpy / scipy linear algebra).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::{Accum, Col, Mat, MatRef, Par};

use crate::error::{MoptError, MoptResult};

/// The number of tasks the dense kernels of this module split their work into.
///
/// faer's default parallelism (`Par::rayon(0)`) takes the number of threads of the pool it
/// runs in, and its products, factorizations and eigendecompositions partition the work, and
/// so sum in an order, that depends on it: the same matrix gave other bits in pools of 1 and
/// of 8 threads (measured, `docs/source/benchmarks.md`). The kernels here pass a fixed degree
/// instead, so their results are the same bits in any pool; the tasks run in the ambient pool
/// (one thread runs them one after the other).
pub const KERNEL_TASKS: usize = 8;

/// The parallelism of the dense kernels ([`KERNEL_TASKS`]).
pub fn kernel_par() -> Par {
    Par::rayon(KERNEL_TASKS)
}

/// The products of fewer multiplications than this (`m n k`, about `161^3`) run on one
/// thread: below it the tasks cost more than they save (measured with
/// `examples/bench_kernels.rs`, `docs/source/benchmarks.md`: 8 tasks are slower at `128^3`,
/// 2.7 times faster at `192^3`).
pub const PARALLEL_PRODUCT: usize = 1 << 22;

/// The LU and Cholesky factorizations of matrices of a smaller order than this run on one
/// thread (measured as [`PARALLEL_PRODUCT`]: 8 tasks were slower up to 512, even at 768, and
/// the Cholesky factorization 1.7 times faster at 1536).
pub const PARALLEL_ORDER: usize = 1024;

/// The eigendecompositions of matrices of a smaller order than this run on one thread (split
/// into tasks they were 3 to 4 times slower at every order up to 768, 1.15 times faster at
/// 1536).
pub const PARALLEL_EIGEN_ORDER: usize = 1536;

/// [`kernel_par`] for `work` multiplications, sequential below [`PARALLEL_PRODUCT`] (the choice
/// depends on the sizes alone, so the bits do not depend on the pool).
pub fn par_for(work: usize) -> Par {
    if work < PARALLEL_PRODUCT {
        Par::Seq
    } else {
        kernel_par()
    }
}

/// [`kernel_par`] for an eigendecomposition of order `n`, sequential below
/// [`PARALLEL_EIGEN_ORDER`].
fn par_for_eigen(n: usize) -> Par {
    if n < PARALLEL_EIGEN_ORDER {
        Par::Seq
    } else {
        kernel_par()
    }
}

/// [`kernel_par`] for a factorization of order `n`, sequential below [`PARALLEL_ORDER`].
fn par_for_order(n: usize) -> Par {
    if n < PARALLEL_ORDER {
        Par::Seq
    } else {
        kernel_par()
    }
}

/// `a b` with [`par_for`].
pub fn mul(a: MatRef<'_, f64>, b: MatRef<'_, f64>) -> Mat<f64> {
    mul_with(a, b, par_for(a.nrows() * a.ncols() * b.ncols()))
}

/// `a b` with the parallelism `par` (for measurements of the thresholds; the kernels choose
/// theirs by [`par_for`]).
pub fn mul_with(a: MatRef<'_, f64>, b: MatRef<'_, f64>, par: Par) -> Mat<f64> {
    let mut out = Mat::<f64>::zeros(a.nrows(), b.ncols());
    faer::linalg::matmul::matmul(out.as_mut(), Accum::Replace, a, b, 1.0, par);
    out
}

/// The eigenvalues and eigenvectors of the lower triangle of `a` with the parallelism `par`,
/// in faer's order (for measurements of the thresholds; [`symmetric_eigen`] chooses its own).
pub fn eigen_with(a: MatRef<'_, f64>, par: Par) -> MoptResult<(Col<f64>, Option<Mat<f64>>)> {
    self_adjoint_evd(a, true, par)
}

/// `a b c` with [`kernel_par`] (`(a b) c`).
pub fn mul3(a: MatRef<'_, f64>, b: MatRef<'_, f64>, c: MatRef<'_, f64>) -> Mat<f64> {
    mul(mul(a, b).as_ref(), c)
}

/// An LU factorization with partial pivoting computed with [`kernel_par`] (the same as
/// faer's `partial_piv_lu`, whose parallelism follows the pool).
#[derive(Debug, Clone)]
pub struct DenseLu {
    l: Mat<f64>,
    u: Mat<f64>,
    forward: Vec<usize>,
    inverse: Vec<usize>,
}

impl DenseLu {
    /// Factorize the square matrix `a`.
    pub fn new(a: MatRef<'_, f64>) -> MoptResult<Self> {
        Self::new_with(a, par_for_order(a.nrows()))
    }

    /// [`Self::new`] with the parallelism `par` (for measurements of the thresholds).
    pub fn new_with(a: MatRef<'_, f64>, par: Par) -> MoptResult<Self> {
        let n = a.nrows();
        if a.ncols() != n {
            return Err(MoptError::linalg(format!(
                "LU expects a square matrix, got {}x{}",
                n,
                a.ncols()
            )));
        }
        let mut lu = a.to_owned();
        let mut forward = vec![0usize; n];
        let mut inverse = vec![0usize; n];
        let mut mem = MemBuffer::new(
            faer::linalg::lu::partial_pivoting::factor::lu_in_place_scratch::<usize, f64>(
                n,
                n,
                par,
                Default::default(),
            ),
        );
        faer::linalg::lu::partial_pivoting::factor::lu_in_place(
            lu.as_mut(),
            &mut forward,
            &mut inverse,
            par,
            MemStack::new(&mut mem),
            Default::default(),
        );
        let l = Mat::from_fn(n, n, |i, j| match i.cmp(&j) {
            std::cmp::Ordering::Greater => lu[(i, j)],
            std::cmp::Ordering::Equal => 1.0,
            std::cmp::Ordering::Less => 0.0,
        });
        let u = Mat::from_fn(n, n, |i, j| if i <= j { lu[(i, j)] } else { 0.0 });
        Ok(Self {
            l,
            u,
            forward,
            inverse,
        })
    }

    /// The solution `x` of `a x = rhs` (any number of columns).
    pub fn solve(&self, rhs: MatRef<'_, f64>) -> Mat<f64> {
        let n = self.l.nrows();
        let par = par_for(n * n * rhs.ncols());
        let mut x = rhs.to_owned();
        let perm = faer::perm::PermRef::new_checked(&self.forward, &self.inverse, n);
        let mut mem = MemBuffer::new(
            faer::linalg::lu::partial_pivoting::solve::solve_in_place_scratch::<usize, f64>(
                n,
                x.ncols(),
                par,
            ),
        );
        faer::linalg::lu::partial_pivoting::solve::solve_in_place(
            self.l.as_ref(),
            self.u.as_ref(),
            perm,
            x.as_mut(),
            par,
            MemStack::new(&mut mem),
        );
        x
    }

    /// [`Self::solve`] of one column.
    pub fn solve_col(&self, rhs: &Col<f64>) -> Col<f64> {
        let x = self.solve(rhs.as_mat());
        Col::from_fn(rhs.nrows(), |i| x[(i, 0)])
    }
}

/// A Cholesky factorization `a = L L^T` computed with [`kernel_par`]; `None` when `a` is not
/// positive definite.
#[derive(Debug, Clone)]
pub struct DenseLlt {
    l: Mat<f64>,
}

impl DenseLlt {
    /// Factorize the symmetric matrix `a` (its lower triangle is read).
    pub fn new(a: MatRef<'_, f64>) -> Option<Self> {
        Self::new_with(a, par_for_order(a.nrows()))
    }

    /// [`Self::new`] with the parallelism `par` (for measurements of the thresholds).
    pub fn new_with(a: MatRef<'_, f64>, par: Par) -> Option<Self> {
        let n = a.nrows();
        if a.ncols() != n {
            return None;
        }
        let mut l = Mat::from_fn(n, n, |i, j| if i >= j { a[(i, j)] } else { 0.0 });
        let mut mem = MemBuffer::new(
            faer::linalg::cholesky::llt::factor::cholesky_in_place_scratch::<f64>(
                n,
                par,
                Default::default(),
            ),
        );
        faer::linalg::cholesky::llt::factor::cholesky_in_place(
            l.as_mut(),
            Default::default(),
            par,
            MemStack::new(&mut mem),
            Default::default(),
        )
        .ok()?;
        for j in 0..n {
            for i in 0..j {
                l[(i, j)] = 0.0;
            }
        }
        Some(Self { l })
    }

    /// `L^-1 rhs` (forward substitution alone).
    pub fn solve_lower(&self, rhs: MatRef<'_, f64>) -> Mat<f64> {
        let n = self.l.nrows();
        let mut x = rhs.to_owned();
        faer::linalg::triangular_solve::solve_lower_triangular_in_place(
            self.l.as_ref(),
            x.as_mut(),
            par_for(n * n * rhs.ncols()),
        );
        x
    }

    /// The solution `x` of `a x = rhs` (any number of columns).
    pub fn solve(&self, rhs: MatRef<'_, f64>) -> Mat<f64> {
        let n = self.l.nrows();
        let par = par_for(n * n * rhs.ncols());
        let mut x = rhs.to_owned();
        let mut mem = MemBuffer::new(
            faer::linalg::cholesky::llt::solve::solve_in_place_scratch::<f64>(n, x.ncols(), par),
        );
        faer::linalg::cholesky::llt::solve::solve_in_place(
            self.l.as_ref(),
            x.as_mut(),
            par,
            MemStack::new(&mut mem),
        );
        x
    }
}

/// Dot product `a . b`.
#[inline]
pub fn col_dot(a: &Col<f64>, b: &Col<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Euclidean norm `||a||`.
#[inline]
pub fn col_norm(a: &Col<f64>) -> f64 {
    libm::sqrt(col_dot(a, a))
}

/// Maximum absolute component `max_i |a_i|` (`0` for an empty vector).
#[inline]
pub fn col_max_abs(a: &Col<f64>) -> f64 {
    a.iter().fold(0.0, |m, v| m.max(v.abs()))
}

/// Root mean square `sqrt(sum_i a_i^2 / n)` (`0` for an empty vector).
#[inline]
pub fn col_rms(a: &Col<f64>) -> f64 {
    if a.nrows() == 0 {
        0.0
    } else {
        libm::sqrt(col_dot(a, a) / a.nrows() as f64)
    }
}

/// Symmetrize a square matrix in place: `A <- (A + A^T) / 2`.
pub fn symmetrize(a: &mut Mat<f64>) {
    let n = a.nrows();
    for i in 0..n {
        for j in (i + 1)..n {
            let v = 0.5 * (a[(i, j)] + a[(j, i)]);
            a[(i, j)] = v;
            a[(j, i)] = v;
        }
    }
}

/// Eigen-decomposition of a symmetric matrix with eigenvalues in ascending order.
///
/// Returns `(eigenvalues, eigenvectors)` where column `k` of the matrix is the eigenvector of
/// eigenvalue `k`.
pub fn symmetric_eigen(a: &Mat<f64>) -> MoptResult<(Col<f64>, Mat<f64>)> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err(MoptError::linalg(format!(
            "symmetric_eigen expects a square matrix, got {}x{}",
            n,
            a.ncols()
        )));
    }
    if n == 0 {
        return Ok((Col::zeros(0), Mat::zeros(0, 0)));
    }
    let (s, u) = self_adjoint_evd(a.as_ref(), true, par_for_eigen(n))?;
    let u = u.ok_or_else(|| MoptError::linalg("eigendecomposition without vectors"))?;
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| s[i].total_cmp(&s[j]));
    let vals = Col::from_fn(n, |k| s[order[k]]);
    // faer returns the eigenvalues in ascending order already: copy the vectors as they are.
    let vecs = if order.iter().enumerate().all(|(k, &o)| k == o) {
        u.to_owned()
    } else {
        Mat::from_fn(n, n, |r, k| u[(r, order[k])])
    };
    Ok((vals, vecs))
}

/// The eigenvalues of a symmetric matrix in ascending order, without the eigenvectors (a
/// fraction of the work of [`symmetric_eigen`]).
pub fn symmetric_eigenvalues(a: &Mat<f64>) -> MoptResult<Col<f64>> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err(MoptError::linalg(format!(
            "symmetric_eigenvalues expects a square matrix, got {}x{}",
            n,
            a.ncols()
        )));
    }
    if n == 0 {
        return Ok(Col::zeros(0));
    }
    let (s, _) = self_adjoint_evd(a.as_ref(), false, par_for_eigen(n))?;
    let mut s: Vec<f64> = s.iter().copied().collect();
    s.sort_by(f64::total_cmp);
    Ok(Col::from_fn(n, |k| s[k]))
}

/// faer's eigendecomposition of the lower triangle of `a` with `par`: the
/// eigenvalues, and the eigenvectors when `vectors`.
fn self_adjoint_evd(
    a: MatRef<'_, f64>,
    vectors: bool,
    par: Par,
) -> MoptResult<(Col<f64>, Option<Mat<f64>>)> {
    use faer::linalg::evd::{self, ComputeEigenvectors};
    let n = a.nrows();
    let compute = if vectors {
        ComputeEigenvectors::Yes
    } else {
        ComputeEigenvectors::No
    };
    let mut s = faer::diag::Diag::<f64>::zeros(n);
    let mut u = vectors.then(|| Mat::<f64>::zeros(n, n));
    let mut mem = MemBuffer::new(evd::self_adjoint_evd_scratch::<f64>(
        n,
        compute,
        par,
        Default::default(),
    ));
    evd::self_adjoint_evd(
        a,
        s.as_mut(),
        u.as_mut().map(|u| u.as_mut()),
        par,
        MemStack::new(&mut mem),
        Default::default(),
    )
    .map_err(|e| MoptError::linalg(format!("eigendecomposition failed: {e:?}")))?;
    Ok((Col::from_fn(n, |k| s[k]), u))
}

/// `sum_k scale_k a[:, cols_k] b[:, cols_k]^T`, that is `A_K diag(scale) B_K^T` over the
/// columns `cols`, as one product of matrices (each term `(a_ik scale_k) b_jk`, as a loop over
/// the columns would form it; the sum is taken in the order of the product).
pub fn scaled_columns_product(
    a: &Mat<f64>,
    b: &Mat<f64>,
    cols: &[usize],
    scale: &[f64],
) -> Mat<f64> {
    let left = Mat::from_fn(a.nrows(), cols.len(), |i, k| a[(i, cols[k])] * scale[k]);
    let right = Mat::from_fn(b.nrows(), cols.len(), |j, k| b[(j, cols[k])]);
    mul(left.as_ref(), right.transpose())
}

/// Two-pass modified Gram-Schmidt orthonormalization that drops (near-)dependent columns.
///
/// A candidate is dropped when its norm is `<= tol`, or when the norm left after removing the
/// components along the accepted basis is `<= tol * max(1, original norm)`.
pub fn orthonormalize_columns(nrows: usize, cols: &[Col<f64>], tol: f64) -> Mat<f64> {
    let mut basis: Vec<Col<f64>> = Vec::with_capacity(cols.len());
    for cand in cols {
        let mut v = cand.clone();
        let orig_norm = col_norm(&v);
        if orig_norm <= tol {
            continue;
        }
        for _pass in 0..2 {
            for b in &basis {
                let proj = col_dot(b, &v);
                for i in 0..nrows {
                    v[i] -= proj * b[i];
                }
            }
        }
        let nrm = col_norm(&v);
        if nrm > tol * orig_norm.max(1.0) {
            for i in 0..nrows {
                v[i] /= nrm;
            }
            basis.push(v);
        }
    }
    Mat::from_fn(nrows, basis.len(), |r, c| basis[c][r])
}

/// Inverse of a small square matrix by Gauss-Jordan elimination with partial pivoting.
///
/// Returns `None` when a pivot is exactly zero (the matrix is singular), mirroring
/// `numpy.linalg.inv`.
pub fn inverse_small(a: &Mat<f64>) -> Option<Mat<f64>> {
    let n = a.nrows();
    let mut m = a.clone();
    let mut inv = Mat::<f64>::identity(n, n);
    for col in 0..n {
        let pivot = (col..n).max_by(|&i, &j| m[(i, col)].abs().total_cmp(&m[(j, col)].abs()))?;
        if m[(pivot, col)] == 0.0 || !m[(pivot, col)].is_finite() {
            return None;
        }
        if pivot != col {
            for k in 0..n {
                let (t, u) = (m[(col, k)], inv[(col, k)]);
                m[(col, k)] = m[(pivot, k)];
                inv[(col, k)] = inv[(pivot, k)];
                m[(pivot, k)] = t;
                inv[(pivot, k)] = u;
            }
        }
        let d = m[(col, col)];
        for k in 0..n {
            m[(col, k)] /= d;
            inv[(col, k)] /= d;
        }
        for r in 0..n {
            if r != col {
                let f = m[(r, col)];
                if f != 0.0 {
                    for k in 0..n {
                        m[(r, k)] -= f * m[(col, k)];
                        inv[(r, k)] -= f * inv[(col, k)];
                    }
                }
            }
        }
    }
    Some(inv)
}

/// Pseudo-inverse of the symmetric positive semi-definite `g` (a Gram matrix `J J^T`) from its
/// eigenvalues: those above `relative * max |lambda|` are inverted, the others dropped; with
/// the number kept. Each caller passes its own threshold (the constraint restores `1e-12`, the
/// SHAKE / RATTLE projections `1e-10`). The sum over the kept eigenvectors is taken term by
/// term in the order of the eigenvalues (`v_ak v_bk / lambda_k`).
///
/// MultiOptPy: no counterpart (new in mopt-rs).
pub fn psd_pseudo_inverse(g: &Mat<f64>, relative: f64) -> MoptResult<(Mat<f64>, usize)> {
    let (w, v) = symmetric_eigen(g)?;
    let m = w.nrows();
    let largest = w.iter().fold(0.0_f64, |a, x| a.max(x.abs()));
    let mut inv = Mat::zeros(m, m);
    let mut rank = 0;
    for k in (0..m).filter(|&k| w[k] > relative * largest.max(1e-300)) {
        rank += 1;
        for a in 0..m {
            for b in 0..m {
                inv[(a, b)] += v[(a, k)] * v[(b, k)] / w[k];
            }
        }
    }
    Ok((inv, rank))
}

/// Moore-Penrose pseudo-inverse of a general matrix via SVD (relative cutoff `1e-15`).
pub fn pseudo_inverse(a: &Mat<f64>) -> MoptResult<Mat<f64>> {
    let svd = a
        .thin_svd()
        .map_err(|e| MoptError::linalg(format!("SVD failed: {e:?}")))?;
    let (u, s, v) = (svd.U(), svd.S(), svd.V());
    let k = s.dim();
    let smax = (0..k).fold(0.0_f64, |m, i| m.max(s[i]));
    let cutoff = 1e-15 * smax * (a.nrows().max(a.ncols()) as f64);
    let kept: Vec<usize> = (0..k).filter(|&i| s[i] > cutoff).collect();
    let inv: Vec<f64> = kept.iter().map(|&i| 1.0 / s[i]).collect();
    Ok(scaled_columns_product(
        &v.to_owned(),
        &u.to_owned(),
        &kept,
        &inv,
    ))
}

/// `numpy.linalg.inv` with MultiOptPy's `safe_inv` fallbacks: plain inverse, then inverse of
/// `A + reg I`, then pseudo-inverse of `A + reg I`.
pub fn safe_inverse(a: &Mat<f64>, reg: f64) -> MoptResult<Mat<f64>> {
    if let Some(inv) = inverse_small(a) {
        return Ok(inv);
    }
    let n = a.nrows();
    let areg = a + Mat::<f64>::identity(n, n) * reg;
    match inverse_small(&areg) {
        Some(inv) => Ok(inv),
        None => pseudo_inverse(&areg),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_pseudo_inverse_of_a_gram_matrix_with_its_rank() {
        // J J^T of three rows, the third the sum of the first two: rank 2.
        let j = Mat::from_fn(3, 4, |r, c| match r {
            0 => [1.0, 0.5, -0.2, 0.0][c],
            1 => [0.0, 1.0, 0.3, -0.7][c],
            _ => [1.0, 1.5, 0.1, -0.7][c],
        });
        let g = &j * j.transpose();
        let (inv, rank) = psd_pseudo_inverse(&g, 1e-12).unwrap();
        assert_eq!(rank, 2);
        // G G^+ G = G.
        let back = &g * &inv * &g;
        for a in 0..3 {
            for b in 0..3 {
                assert!((back[(a, b)] - g[(a, b)]).abs() < 1e-12);
            }
        }
        assert_eq!(psd_pseudo_inverse(&g, 2.0).unwrap().1, 0);
    }

    #[test]
    fn eigen_is_sorted_and_reconstructs() {
        let a = Mat::from_fn(3, 3, |i, j| {
            [[2.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 4.0]][i][j]
        });
        let (w, v) = symmetric_eigen(&a).unwrap();
        assert!(w[0] <= w[1] && w[1] <= w[2]);
        let recon = &v * Mat::from_fn(3, 3, |i, j| if i == j { w[i] } else { 0.0 }) * v.transpose();
        for i in 0..3 {
            for j in 0..3 {
                assert!((recon[(i, j)] - a[(i, j)]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn inverses() {
        let a = Mat::from_fn(3, 3, |i, j| {
            [[0.0, 2.0, 1.0], [1.0, 1.0, 0.0], [3.0, 0.0, 1.0]][i][j]
        });
        let inv = inverse_small(&a).unwrap();
        let id = &a * &inv;
        for i in 0..3 {
            for j in 0..3 {
                assert!((id[(i, j)] - f64::from(u8::from(i == j))).abs() < 1e-14);
            }
        }
        let sing = Mat::from_fn(2, 2, |i, _| [1.0, 2.0][i]);
        assert!(inverse_small(&sing).is_none());
        let p = pseudo_inverse(&sing).unwrap();
        let back = &sing * &p * &sing;
        assert!((0..2).all(|i| (0..2).all(|j| (back[(i, j)] - sing[(i, j)]).abs() < 1e-12)));
        assert!(safe_inverse(&sing, 1e-10).is_ok());
    }

    /// A symmetric positive definite test matrix of order `n`.
    fn spd(n: usize) -> Mat<f64> {
        let a = Mat::from_fn(n, n, |i, j| ((i * 7 + j * 13) % 17) as f64 / 17.0 - 0.4);
        let mut m = &a * a.transpose();
        for i in 0..n {
            m[(i, i)] += n as f64;
        }
        m
    }

    fn bits(m: &Mat<f64>) -> Vec<u64> {
        (0..m.ncols())
            .flat_map(|j| (0..m.nrows()).map(move |i| (i, j)))
            .map(|(i, j)| m[(i, j)].to_bits())
            .collect()
    }

    /// The kernels give the same bits in pools of 1, 2, 4 and 8 threads, for orders on both
    /// sides of the thresholds of parallel work (products from about 161, factorizations
    /// from 1024).
    #[test]
    fn kernels_do_not_depend_on_the_number_of_threads() -> MoptResult<()> {
        for n in [40, 360, 1100] {
            let a = spd(n);
            let rhs = Mat::from_fn(n, 3, |i, j| (i + 2 * j) as f64 / n as f64);
            let run = || -> MoptResult<Vec<Vec<u64>>> {
                let (w, v) = symmetric_eigen(&a)?;
                let lu = DenseLu::new(a.as_ref())?;
                let llt = DenseLlt::new(a.as_ref())
                    .ok_or_else(|| MoptError::linalg("not positive definite"))?;
                let w2 = symmetric_eigenvalues(&a)?;
                Ok(vec![
                    w.iter().map(|v| v.to_bits()).collect(),
                    bits(&v),
                    w2.iter().map(|v| v.to_bits()).collect(),
                    bits(&mul(a.as_ref(), a.as_ref())),
                    bits(&lu.solve(rhs.as_ref())),
                    bits(&llt.solve(rhs.as_ref())),
                ])
            };
            let mut first = None;
            for threads in [1, 2, 4, 8] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .map_err(|e| MoptError::config(e.to_string()))?;
                let got = pool.install(run)?;
                match &first {
                    None => first = Some(got),
                    Some(reference) => {
                        for (k, (x, y)) in reference.iter().zip(&got).enumerate() {
                            assert!(x == y, "order {n}, kernel {k}, {threads} threads");
                        }
                    }
                }
            }
            // And they are right.
            let lu = DenseLu::new(a.as_ref())?;
            let x = lu.solve(rhs.as_ref());
            let back = &a * &x;
            for i in 0..n {
                for j in 0..3 {
                    assert!((back[(i, j)] - rhs[(i, j)]).abs() < 1e-10);
                }
            }
            let llt = DenseLlt::new(a.as_ref()).ok_or_else(|| MoptError::linalg("llt"))?;
            let y = llt.solve(rhs.as_ref());
            for i in 0..n {
                for j in 0..3 {
                    assert!((x[(i, j)] - y[(i, j)]).abs() < 1e-12);
                }
            }
            let (w, _) = symmetric_eigen(&a)?;
            let w2 = symmetric_eigenvalues(&a)?;
            for k in 0..n {
                assert!((w[k] - w2[k]).abs() < 1e-9 * w[n - 1]);
            }
        }
        // Not positive definite: no Cholesky factor.
        let mut m = spd(5);
        m[(2, 2)] = -100.0;
        assert!(DenseLlt::new(m.as_ref()).is_none());
        Ok(())
    }

    #[test]
    fn scaled_columns_product_matches_the_loop() {
        let v = Mat::from_fn(6, 4, |i, j| (i as f64 + 1.0) * (j as f64 - 1.5));
        let s = [2.0, -1.0, 0.5];
        let cols = [0, 2, 3];
        let p = scaled_columns_product(&v, &v, &cols, &s);
        for i in 0..6 {
            for j in 0..6 {
                let want: f64 = cols
                    .iter()
                    .zip(&s)
                    .map(|(&k, w)| v[(i, k)] * w * v[(j, k)])
                    .sum();
                assert!((p[(i, j)] - want).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn gram_schmidt_drops_dependent_columns() {
        let a = Col::from_fn(3, |i| [1.0, 0.0, 0.0][i]);
        let b = Col::from_fn(3, |i| [2.0, 0.0, 0.0][i]);
        let c = Col::from_fn(3, |i| [1.0, 1.0, 0.0][i]);
        let q = orthonormalize_columns(3, &[a, b, c], 1e-10);
        assert_eq!(q.ncols(), 2);
        assert!((q[(1, 1)].abs() - 1.0).abs() < 1e-14);
    }
}
