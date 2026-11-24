//! Interruption and restart through the evaluation journal.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use mopt_rs::cli::optmain::{config_from_cli, run_with, OptmainConfig};
use mopt_rs::domain::{EvalOutput, EvalRequest, MolecularSystem};
use mopt_rs::error::MoptResult;
use mopt_rs::pes::native::LennardJones;
use mopt_rs::pes::registry::CalculatorRegistry;
use mopt_rs::pes::Calculator;
use mopt_rs::workflow::journal::{read_journal, JOURNAL_FILE};
use mopt_rs::workflow::observers::STOP_FILE;

const AR7: [[f64; 3]; 7] = [
    [1.9500, 0.0000, 0.0000],
    [0.6026, 1.8546, 0.0300],
    [-1.5776, 1.1462, 0.0700],
    [-1.5776, -1.1462, 0.1100],
    [0.6026, -1.8546, 0.1500],
    [0.0000, 0.0000, 2.1000],
    [0.2000, 0.0000, -2.0000],
];

struct Counting(Arc<AtomicUsize>);

impl Calculator for Counting {
    fn name(&self) -> &str {
        "counting_lj"
    }
    fn evaluate(&self, sys: &MolecularSystem, req: &EvalRequest) -> MoptResult<EvalOutput> {
        self.0.fetch_add(1, Ordering::SeqCst);
        LennardJones.evaluate(sys, req)
    }
}

fn registry(count: &Arc<AtomicUsize>) -> CalculatorRegistry {
    let mut r = CalculatorRegistry::native();
    let c = Arc::clone(count);
    r.register("counting", move |_| Ok(Box::new(Counting(Arc::clone(&c)))));
    r
}

fn config(xyz: &str) -> OptmainConfig {
    config_from_cli(&[xyz, "-os", "counting", "-opt", "rsirfo_bfgs", "-fc", "4"]).unwrap()
}

fn optimized(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap()
}

#[test]
fn restarts_continue_exactly_where_runs_stopped() {
    let base = std::env::temp_dir().join(format!("mopt_restart_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    std::fs::create_dir_all(&base).unwrap();
    let xyz = base.join("ar7.xyz");
    let mut text = String::from("7\n\n");
    for p in AR7 {
        text.push_str(&format!("Ar {} {} {}\n", p[0], p[1], p[2]));
    }
    std::fs::write(&xyz, text).unwrap();
    let xyz = xyz.to_string_lossy().into_owned();

    // An uninterrupted run.
    let count = Arc::new(AtomicUsize::new(0));
    let full = run_with(&config(&xyz), &base, None, &registry(&count)).unwrap();
    let total = count.load(Ordering::SeqCst);
    assert!(full[0].converged && total > 6, "{total} evaluations");
    let dir = full[0].directory.clone();
    let journal = dir.join(JOURNAL_FILE);
    let (records, _) = read_journal(&journal).unwrap();
    assert_eq!(records.len(), total, "every evaluation is recorded");
    let reference = optimized(&full[0].optimized);

    // Killed after k evaluations, in the middle of writing the next record.
    let k = total / 2;
    let bytes = std::fs::read(&journal).unwrap();
    let (first_k, _) = {
        let mut cut = 8usize;
        for _ in 0..k {
            let len = u32::from_le_bytes(bytes[cut..cut + 4].try_into().unwrap()) as usize;
            cut += 8 + len;
        }
        (cut, ())
    };
    std::fs::write(&journal, &bytes[..first_k + 10]).unwrap();
    std::fs::remove_file(&full[0].optimized).unwrap();

    let mut cfg = config(&xyz);
    cfg.restart_dir = Some(dir.clone());
    count.store(0, Ordering::SeqCst);
    let again = run_with(&cfg, &base, None, &registry(&count)).unwrap();
    assert_eq!(
        again[0].directory, dir,
        "the run continues in its own directory"
    );
    assert_eq!(
        count.load(Ordering::SeqCst),
        total - k,
        "only the evaluations after the interruption are recomputed"
    );
    assert_eq!(
        optimized(&again[0].optimized),
        reference,
        "identical result"
    );
    assert_eq!(read_journal(&journal).unwrap().0.len(), total);

    // A STOP file interrupts cleanly; a restart then finishes without new evaluations.
    std::fs::write(dir.join(STOP_FILE), "").unwrap();
    count.store(0, Ordering::SeqCst);
    let stopped = run_with(&cfg, &base, None, &registry(&count)).unwrap();
    assert!(stopped[0].interrupted && !stopped[0].converged);
    assert!(!dir.join(STOP_FILE).exists(), "the STOP file is consumed");
    let done = run_with(&cfg, &base, None, &registry(&count)).unwrap();
    assert!(done[0].converged && !done[0].interrupted);
    assert_eq!(
        count.load(Ordering::SeqCst),
        0,
        "everything came from the journal"
    );
    assert_eq!(optimized(&done[0].optimized), reference);
    let _ = std::fs::remove_dir_all(&base);
}
