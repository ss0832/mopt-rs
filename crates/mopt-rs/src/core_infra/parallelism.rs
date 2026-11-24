//! The parallelism of the linear algebra outside the explicit kernels: one process-wide
//! setting that makes results independent of the number of threads.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! faer keeps a global parallelism (`faer::set_global_parallelism`) that every operator
//! (`&a * &b`) and every decomposition called without an explicit parallelism argument
//! (`partial_piv_lu`, `full_piv_lu`, `self_adjoint_eigen`, `llt`, `svd`, `qr`, ...) reads at
//! the time of the call. Its default is `Par::rayon(0)`, which faer turns into
//! `Par::Rayon(rayon::current_num_threads())`: the number of tasks, and with it the
//! partition of the work and the order of the sums, follows the pool the call runs in, so
//! the same product or factorization gives other last bits in pools of other sizes.
//! [`deterministic_linear_algebra`] sets it to `Par::Seq` once per process; the dense kernels
//! of [`crate::math::linalg`] pass their own fixed degree and are not affected.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::sync::Once;

static SET: Once = Once::new();

/// Make the linear algebra that does not go through the kernels of [`crate::math::linalg`]
/// independent of the number of threads: set faer's global parallelism to sequential, once
/// per process.
///
/// The commands (`crate::cli`), the construction of a native
/// [`crate::pes::registry::CalculatorRegistry`] and of the calculators it hands out, the
/// MCP server and the Python module call it, so a program that uses the library through any
/// of them is covered; one that calls the optimizers or the engines directly calls it once
/// at its start. Calling it again does nothing, so a program that wants parallel linear algebra
/// sets it with `faer::set_global_parallelism` after the first call, and gives up the
/// guarantee (faer's default, the threads of the pool of each call, cannot be set again
/// once it was changed; a fixed `Par::rayon(n)` can).
pub fn deterministic_linear_algebra() {
    SET.call_once(|| faer::set_global_parallelism(faer::Par::Seq));
}

/// Whether faer's global parallelism is sequential now.
pub fn linear_algebra_is_sequential() -> bool {
    faer::get_global_parallelism() == faer::Par::Seq
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sets_sequential_once() {
        deterministic_linear_algebra();
        assert!(linear_algebra_is_sequential());
        deterministic_linear_algebra();
        assert!(linear_algebra_is_sequential());
    }
}
