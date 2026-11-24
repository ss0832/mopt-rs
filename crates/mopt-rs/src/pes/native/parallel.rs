//! Loops of the engines over the atoms in fixed chunks, so that they can run in parallel
//! with results that do not depend on the number of threads.
//!
//! A loop over the atoms `i` (pair loops over `j < i` inside) is cut into chunks of
//! [`ROWS_PER_CHUNK`] atoms; every chunk adds into its own partial sums (an energy, a
//! gradient, ...), and the partial sums are added up chunk by chunk in the order of the
//! chunks. The partition depends on the number of atoms alone, so the sums are the same bits
//! for any number of threads and whether the chunks run in parallel or one after the other.
//! The chunks run in the ambient rayon pool (inside the pool of a mapper job or of a chain of
//! images) from [`PARALLEL_ATOMS`] atoms on; below, one after the other, where the
//! bookkeeping of the tasks would cost more than the loop.
//!
//! The chunks are taken in waves of [`CHUNKS_PER_WAVE`]: the partial sums of one wave are
//! held at once, so the memory is bounded however many atoms there are.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::ops::Range;

use rayon::prelude::*;

/// Atoms per chunk.
pub const ROWS_PER_CHUNK: usize = 32;

/// Chunks whose partial sums are held at once.
pub const CHUNKS_PER_WAVE: usize = 64;

/// The number of atoms from which the chunks run in parallel (see
/// `docs/source/benchmarks.md` for the measurement).
pub const PARALLEL_ATOMS: usize = 256;

/// Run `work` on the chunks of `0..n` and hand the results to `reduce` in the order of the
/// chunks (see the module documentation).
pub fn for_row_chunks<T, W, R>(n: usize, work: W, mut reduce: R)
where
    T: Send,
    W: Fn(Range<usize>) -> T + Sync,
    R: FnMut(T),
{
    let chunks = n.div_ceil(ROWS_PER_CHUNK);
    let range = |c: usize| c * ROWS_PER_CHUNK..((c + 1) * ROWS_PER_CHUNK).min(n);
    if n < PARALLEL_ATOMS {
        for c in 0..chunks {
            reduce(work(range(c)));
        }
        return;
    }
    let mut first = 0;
    while first < chunks {
        let last = (first + CHUNKS_PER_WAVE).min(chunks);
        let wave: Vec<T> = (first..last)
            .into_par_iter()
            .map(|c| work(range(c)))
            .collect();
        wave.into_iter().for_each(&mut reduce);
        first = last;
    }
}

/// `a += b` element by element.
pub fn add_into(a: &mut [[f64; 3]], b: &[[f64; 3]]) {
    for (x, y) in a.iter_mut().zip(b) {
        for c in 0..3 {
            x[c] += y[c];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A pair sum whose rounding shows the order of the additions.
    fn pair_sum(n: usize) -> f64 {
        let mut total = 0.0;
        for_row_chunks(
            n,
            |rows| {
                let mut s = 0.0;
                for i in rows {
                    for j in 0..i {
                        s += 1.0 / ((i * n + j) as f64 + 0.1);
                    }
                }
                s
            },
            |s| total += s,
        );
        total
    }

    #[test]
    fn the_sums_do_not_depend_on_the_number_of_threads() -> crate::error::MoptResult<()> {
        for n in [5, 300, 3000] {
            let mut first = None;
            for threads in [1, 2, 4, 8] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .map_err(|e| crate::error::MoptError::config(e.to_string()))?;
                let s = pool.install(|| pair_sum(n));
                match first {
                    None => first = Some(s.to_bits()),
                    Some(b) => assert_eq!(b, s.to_bits(), "{n} atoms, {threads} threads"),
                }
            }
        }
        Ok(())
    }
}
