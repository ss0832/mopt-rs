//! Zhu-Nakamura surface hopping in `mdmain` on a two-state model with a known probability:
//! two hydrogen atoms whose adiabatic states come from straight diabatic states along their
//! distance (forces 0.02 and 0.005 Hartree/Bohr, coupling 0.002 Hartree, a sloped crossing
//! at 3 Bohr), started 0.8 Bohr before the crossing on the lower state at 300 K. The run
//! writes the files of every trajectory (`plot_surface_hopping_.png` among them) and the
//! populations of the states over all of them (`surface_hopping_populations.csv` and
//! `plot_surface_hopping_populations_.png` in the directory of the first one).
//!
//! ```text
//! cargo run --release -p mopt-rs --example md_surface_hopping -- OUTPUT_DIR [TRAJECTORIES]
//! ```
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::path::PathBuf;

use faer::Col;
use mopt_rs::cli::mdmain::{config_from_cli, run_with};
use mopt_rs::domain::{EvalOutput, EvalRequest, MolecularSystem, Quantity, Units};
use mopt_rs::error::{MoptError, MoptResult};
use mopt_rs::pes::registry::CalculatorRegistry;
use mopt_rs::pes::Calculator;

/// Two straight diabatic states `V_k = -F_k (r - r_x)` of the distance of two atoms with
/// the coupling `V`; state 0 is the lower adiabatic state, 1 the upper.
struct TwoStateModel;

const FORCES: [f64; 2] = [0.02, 0.005];
const COUPLING: f64 = 0.002;
const CROSSING: f64 = 3.0;

impl Calculator for TwoStateModel {
    fn name(&self) -> &str {
        "two_state_model"
    }

    fn supports_excited_states(&self) -> bool {
        true
    }

    fn evaluate(&self, sys: &MolecularSystem, req: &EvalRequest) -> MoptResult<EvalOutput> {
        if req.excited_state > 1 || sys.n_atoms() != 2 {
            return Err(MoptError::config("the model has two atoms and two states"));
        }
        let x = sys.coords_flat_bohr();
        let d: Vec<f64> = (0..3).map(|c| x[3 + c] - x[c]).collect();
        let r = d.iter().map(|v| v * v).sum::<f64>().sqrt();
        let [f1, f2] = FORCES;
        let s = r - CROSSING;
        let half = -0.5 * (f1 - f2) * s;
        let root = (half * half + COUPLING * COUPLING).sqrt();
        let sign = if req.excited_state == 0 { -1.0 } else { 1.0 };
        let energy = -0.5 * (f1 + f2) * s + sign * root;
        let slope = -0.5 * (f1 + f2) + sign * half * (-0.5 * (f1 - f2)) / root;
        let g = Col::<f64>::from_fn(6, |i| {
            let along = slope * d[i % 3] / r;
            if i < 3 {
                -along
            } else {
                along
            }
        });
        Ok(EvalOutput::energy_gradient(Quantity::new(energy), g))
    }
}

fn main() -> MoptResult<()> {
    let args: Vec<String> = std::env::args().collect();
    let out = PathBuf::from(
        args.get(1)
            .ok_or_else(|| MoptError::config("usage: md_surface_hopping OUTPUT_DIR [N]"))?,
    );
    let trajectories = args.get(2).map_or("100", String::as_str);
    std::fs::create_dir_all(&out)?;
    let xyz = out.join("h2.xyz");
    std::fs::write(
        &xyz,
        format!(
            "2\nthe two-state model\nH 0.0 0.0 0.0\nH {:.10} 0.0 0.0\n",
            (CROSSING - 0.8) * Units::BOHR_TO_ANGSTROM
        ),
    )?;
    let xyz = xyz.to_string_lossy().into_owned();
    let config = config_from_cli(&[
        &xyz,
        "-os",
        "zn_model",
        "-mt",
        "nve",
        "-time",
        "120",
        "-ts",
        "0.25fs",
        "-temp",
        "300",
        "-traj",
        trajectories,
        "-md_opt",
        "seed=1",
        "write_every=5",
        "hopping_states=0,1",
    ])?;
    let mut registry = CalculatorRegistry::native();
    registry.register("zn_model", |_| Ok(Box::new(TwoStateModel)));
    let runs = run_with(&config, &out, None, &registry)?;
    let first = runs
        .first()
        .ok_or_else(|| MoptError::calc("no trajectory was run"))?;
    println!(
        "{} trajectories; the first in {}",
        runs.len(),
        first.directory.display()
    );
    let table = std::fs::read_to_string(first.directory.join("surface_hopping_populations.csv"))?;
    if let Some(last) = table.lines().last() {
        println!("populations at the end (time_fs,state_0,state_1): {last}");
    }
    Ok(())
}
