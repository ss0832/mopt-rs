//! The dense kernels of `math::linalg` on one thread against [`KERNEL_TASKS`] tasks, for the
//! thresholds from which the engines split them (`PARALLEL_PRODUCT`, `PARALLEL_ORDER`).
//!
//! ```text
//! cargo run --release -p mopt-rs --example bench_kernels
//! ```
//!
//! For square matrices of each order, inside a rayon pool of 8 threads: the median time of
//! seven products, eigendecompositions (with vectors), LU and Cholesky factorizations, with
//! `Par::Seq` and with the fixed degree of the kernels.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::time::Instant;

use faer::{Mat, Par};
use mopt_rs::error::{MoptError, MoptResult};
use mopt_rs::math::linalg::{eigen_with, kernel_par, mul_with, DenseLlt, DenseLu, KERNEL_TASKS};

fn median_ms(mut f: impl FnMut() -> MoptResult<()>) -> MoptResult<f64> {
    f()?;
    let mut t = Vec::new();
    for _ in 0..7 {
        let start = Instant::now();
        f()?;
        t.push(start.elapsed().as_secs_f64() * 1e3);
    }
    t.sort_by(f64::total_cmp);
    Ok(t[t.len() / 2])
}

fn main() -> MoptResult<()> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build()
        .map_err(|e| MoptError::config(e.to_string()))?;
    println!(
        "| order | product, 1 task (ms) | product, {KERNEL_TASKS} tasks (ms) | eigen, 1 task (ms) \
         | eigen, {KERNEL_TASKS} tasks (ms) | LU, 1 task (ms) | LU, {KERNEL_TASKS} tasks (ms) \
         | Cholesky, 1 task (ms) | Cholesky, {KERNEL_TASKS} tasks (ms) |"
    );
    println!("|---|---|---|---|---|---|---|---|---|");
    pool.install(|| -> MoptResult<()> {
        for n in [64, 96, 128, 192, 256, 320, 384, 512, 768, 1536] {
            let a = Mat::from_fn(n, n, |i, j| {
                let v = ((i * 7 + j * 13) % 17) as f64 / 17.0;
                if i == j {
                    v + n as f64
                } else {
                    v
                }
            });
            let s = Mat::from_fn(n, n, |i, j| a[(i, j)] + a[(j, i)]);
            let product = |par: Par| {
                median_ms(|| {
                    std::hint::black_box(mul_with(a.as_ref(), a.as_ref(), par));
                    Ok(())
                })
            };
            let eigen = |par: Par| {
                median_ms(|| {
                    std::hint::black_box(eigen_with(s.as_ref(), par)?);
                    Ok(())
                })
            };
            let lu = |par: Par| {
                median_ms(|| {
                    std::hint::black_box(DenseLu::new_with(s.as_ref(), par)?);
                    Ok(())
                })
            };
            let llt = |par: Par| {
                median_ms(|| {
                    std::hint::black_box(
                        DenseLlt::new_with(s.as_ref(), par)
                            .ok_or_else(|| MoptError::linalg("not positive definite"))?,
                    );
                    Ok(())
                })
            };
            println!(
                "| {n} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} |",
                product(Par::Seq)?,
                product(kernel_par())?,
                eigen(Par::Seq)?,
                eigen(kernel_par())?,
                lu(Par::Seq)?,
                lu(kernel_par())?,
                llt(Par::Seq)?,
                llt(kernel_par())?
            );
        }
        Ok(())
    })
}
