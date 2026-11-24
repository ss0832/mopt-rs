//! The lowest excited states of the Tamm-Dancoff approximation of a tight-binding reference of
//! any spin (unrestricted: the single excitations of each spin), with a kernel between
//! transition charges, by the Davidson method, and their transition dipoles and changes of the
//! atomic populations.
//!
//! The matrix is `A = diag(e_a - e_i) + Q K Q^T` over the excitations `i -> a` of both spins,
//! `Q` the transition charges of the excitations on the sites of the kernel `K` (shells or
//! atoms). A spin-independent kernel (the Coulomb-like response of the charges, as in the TDA
//! and the sTDA of GFN1-xTB and GFN0-xTB) couples the excitations of the two spins equally; for
//! a closed shell the states are then the singlets (`2 q K q`, the spin sum of the two equal
//! blocks) and the triplets (no kernel). The amplitudes are normalized over both spins,
//! `sum X^2 = 1`.
//!
//! Davidson (E. R. Davidson, J. Comput. Phys. 1975, 17, 87): the subspace grows by the
//! residuals preconditioned by `1 / (e_a - e_i - omega)`, orthonormalized; it restarts from the
//! current vectors when it reaches its largest size. `A x` costs `O(n_exc n_sites)` through
//! `Q^T x`, so the matrix itself is never formed.
//!
//! The transition dipole of a state is `sum X_ia <i|r|a>` (atomic orbitals `D`, orbitals `C`);
//! the change of the density from the ground state to the state (unrelaxed) is `sum_s
//! (C_v X_s^T X_s C_v^T - C_o X_s X_s^T C_o^T)`, whose Mulliken populations give the change
//! of the electrons on each atom (for the charge-transfer index of
//! `kinetics::rate::couplings`).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Mat;

use crate::error::{MoptError, MoptResult};
use crate::math::linalg::symmetric_eigen;

/// The excitations of one spin: the orbital energies of its occupied and virtual orbitals and
/// the transition charges of `i -> a` at row `i * n_virtual + a`.
#[derive(Debug, Clone)]
pub struct SpinBlock {
    /// Occupied orbital energies (Hartree).
    pub occupied: Vec<f64>,
    /// Virtual orbital energies (Hartree).
    pub virtuals: Vec<f64>,
    /// Transition charges (excitations x sites).
    pub charges: Mat<f64>,
}

impl SpinBlock {
    fn len(&self) -> usize {
        self.occupied.len() * self.virtuals.len()
    }

    fn gap(&self, k: usize) -> f64 {
        let nv = self.virtuals.len();
        self.virtuals[k % nv] - self.occupied[k / nv]
    }
}

/// The lowest states: excitation energies (Hartree) and amplitudes (one column per state,
/// the excitations of the first spin block, then of the second).
#[derive(Debug, Clone)]
pub struct UnrestrictedStates {
    /// Excitation energies, ascending.
    pub energies: Vec<f64>,
    /// Amplitudes.
    pub amplitudes: Mat<f64>,
}

/// Convergence of the Davidson iterations: the largest norm of a residual (Hartree).
pub const RESIDUAL_TOLERANCE: f64 = 1e-8;
/// Iterations of the Davidson method at most.
const MAX_ITERATIONS: usize = 200;

/// `A x` for the blocks and the kernel.
fn apply(blocks: &[SpinBlock], kernel: &Mat<f64>, x: &Mat<f64>) -> Mat<f64> {
    let sites = kernel.nrows();
    let cols = x.ncols();
    // T = sum_s Q_s^T x_s (sites x cols).
    let mut t = Mat::<f64>::zeros(sites, cols);
    let mut offset = 0;
    for b in blocks {
        let n = b.len();
        if n > 0 {
            t += b.charges.transpose() * x.subrows(offset, n);
        }
        offset += n;
    }
    let kt = kernel * &t;
    let mut out = Mat::<f64>::zeros(x.nrows(), cols);
    let mut offset = 0;
    for b in blocks {
        let n = b.len();
        if n > 0 {
            let part = &b.charges * &kt;
            for c in 0..cols {
                for k in 0..n {
                    out[(offset + k, c)] = b.gap(k) * x[(offset + k, c)] + part[(k, c)];
                }
            }
        }
        offset += n;
    }
    out
}

/// Gram-Schmidt of `v` against the columns of `basis` (twice), normalized; `None` when
/// nothing is left.
fn orthonormalize(basis: &[Vec<f64>], mut v: Vec<f64>) -> Option<Vec<f64>> {
    for _ in 0..2 {
        for b in basis {
            let d: f64 = b.iter().zip(&v).map(|(x, y)| x * y).sum();
            for (vi, bi) in v.iter_mut().zip(b) {
                *vi -= d * bi;
            }
        }
    }
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    (norm > 1e-10).then(|| v.into_iter().map(|x| x / norm).collect())
}

/// The `roots` lowest states of `A` by the Davidson method; the full matrix is diagonalized
/// instead when it is small (at most `4 roots + 50` excitations).
pub fn lowest_states(
    blocks: &[SpinBlock],
    kernel: &Mat<f64>,
    roots: usize,
) -> MoptResult<UnrestrictedStates> {
    let n: usize = blocks.iter().map(SpinBlock::len).sum();
    if n == 0 || roots == 0 {
        return Err(MoptError::calc(
            "unrestricted TDA: no single excitation (or no state asked for)",
        ));
    }
    if kernel.nrows() != kernel.ncols()
        || blocks.iter().any(|b| {
            b.charges.nrows() != b.len() || (b.len() > 0 && b.charges.ncols() != kernel.nrows())
        })
    {
        return Err(MoptError::config(
            "unrestricted TDA: the transition charges do not fit the kernel",
        ));
    }
    let roots = roots.min(n);
    let diagonal: Vec<f64> = blocks
        .iter()
        .flat_map(|b| (0..b.len()).map(move |k| b.gap(k)))
        .collect();
    if n <= 4 * roots + 50 {
        let identity = Mat::<f64>::identity(n, n);
        let full = apply(blocks, kernel, &identity);
        let sym = Mat::from_fn(n, n, |i, j| 0.5 * (full[(i, j)] + full[(j, i)]));
        let (values, vectors) = symmetric_eigen(&sym)?;
        return Ok(UnrestrictedStates {
            energies: values.iter().take(roots).copied().collect(),
            amplitudes: Mat::from_fn(n, roots, |i, j| vectors[(i, j)]),
        });
    }
    // Start from the excitations of the smallest gaps.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| diagonal[a].total_cmp(&diagonal[b]));
    let block = (2 * roots).min(n);
    let max_space = (8 * roots + 20).min(n);
    let mut basis: Vec<Vec<f64>> = Vec::new();
    for &k in order.iter().take(block) {
        let mut v = vec![0.0; n];
        v[k] = 1.0;
        basis.push(v);
    }
    for _ in 0..MAX_ITERATIONS {
        let m = basis.len();
        let b = Mat::from_fn(n, m, |i, j| basis[j][i]);
        let ab = apply(blocks, kernel, &b);
        let small = b.transpose() * &ab;
        let small = Mat::from_fn(m, m, |i, j| 0.5 * (small[(i, j)] + small[(j, i)]));
        let (values, vectors) = symmetric_eigen(&small)?;
        let ritz = &b * Mat::from_fn(m, roots, |i, j| vectors[(i, j)]);
        let a_ritz = &ab * Mat::from_fn(m, roots, |i, j| vectors[(i, j)]);
        let mut worst = 0.0_f64;
        let mut new = Vec::new();
        for r in 0..roots {
            let omega = values[r];
            let residual: Vec<f64> = (0..n)
                .map(|i| a_ritz[(i, r)] - omega * ritz[(i, r)])
                .collect();
            let norm = residual.iter().map(|x| x * x).sum::<f64>().sqrt();
            worst = worst.max(norm);
            if norm > RESIDUAL_TOLERANCE {
                let corrected: Vec<f64> = residual
                    .iter()
                    .zip(&diagonal)
                    .map(|(res, d)| {
                        let den = d - omega;
                        res / if den.abs() < 1e-6 {
                            1e-6f64.copysign(den)
                        } else {
                            den
                        }
                    })
                    .collect();
                new.push(corrected);
            }
        }
        if worst <= RESIDUAL_TOLERANCE {
            return Ok(UnrestrictedStates {
                energies: values.iter().take(roots).copied().collect(),
                amplitudes: ritz,
            });
        }
        if m + new.len() > max_space {
            // Restart from the current vectors.
            basis = (0..roots)
                .map(|r| (0..n).map(|i| ritz[(i, r)]).collect())
                .collect();
        }
        let mut added = 0;
        for v in new {
            if let Some(v) = orthonormalize(&basis, v) {
                basis.push(v);
                added += 1;
            }
        }
        if added == 0 {
            return Err(MoptError::calc(
                "unrestricted TDA: the Davidson subspace stopped growing before convergence",
            ));
        }
    }
    Err(MoptError::calc(format!(
        "unrestricted TDA: the Davidson iterations did not converge in {MAX_ITERATIONS}"
    )))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn blocks(seed: u64, occ: usize, virt: usize, sites: usize) -> SpinBlock {
        let mut state = seed;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64 / (1u64 << 53) as f64) - 0.5
        };
        SpinBlock {
            occupied: (0..occ)
                .map(|i| -0.5 - 0.05 * i as f64 + 0.01 * next())
                .collect(),
            virtuals: (0..virt)
                .map(|a| 0.1 + 0.03 * a as f64 + 0.01 * next())
                .collect(),
            charges: Mat::from_fn(occ * virt, sites, |_, _| 0.3 * next()),
        }
    }

    #[test]
    fn davidson_finds_the_lowest_states_of_the_full_matrix() {
        let sites = 7;
        let kernel = Mat::from_fn(sites, sites, |i, j| {
            if i == j {
                0.4
            } else {
                0.1 / (1.0 + (i as f64 - j as f64).abs())
            }
        });
        let b = vec![blocks(1, 9, 14, sites), blocks(2, 8, 15, sites)];
        let n: usize = b.iter().map(SpinBlock::len).sum();
        let found = lowest_states(&b, &kernel, 4).unwrap();
        let full = apply(&b, &kernel, &Mat::<f64>::identity(n, n));
        let (values, _) = symmetric_eigen(&full).unwrap();
        for r in 0..4 {
            assert!(
                (found.energies[r] - values[r]).abs() < 1e-9,
                "root {r}: {} vs {}",
                found.energies[r],
                values[r]
            );
        }
        // Normalized over both spins.
        let norm: f64 = (0..n).map(|i| found.amplitudes[(i, 0)].powi(2)).sum();
        assert!((norm - 1.0).abs() < 1e-10);
    }
}
