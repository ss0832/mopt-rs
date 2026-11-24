//! What a molecular dynamics run prints (screen and `mopt.log`), and the analyses of its
//! structures that other commands have (`-gi`, `-cmds`, `-pca`) as observers of the steps.
//!
//! Every `print_every` steps: the step and its time, the energies, MultiOptPy's
//! `hamiltonian :` and `Instantaneous_temperature:` lines, the conserved quantity, the
//! pressure and volume of a cell, the largest residual of the constraints, and the frozen
//! modes when they are computed. At the end: the averages of the run and the drift of the
//! conserved quantity.
//!
//! MultiOptPy: the `print` calls of `moleculardynamics.py:MD` and `MD/thermostat.py`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::path::Path;

use super::md::MdObserver;
use super::md_files::MdLayout;
use super::observers::GeometryInfo;
use super::path_analysis::PathMaps;
use crate::core_infra::log::{python_float, Logger};
use crate::domain::{AtomId, MolecularSystem, Units};
use crate::error::MoptResult;
use crate::md::barostat::gigapascal_per_atomic_unit;
use crate::md::{MdEnd, MdStepReport};

/// The log of a run.
#[derive(Debug)]
pub struct MdLog {
    log: Logger,
    every: usize,
    steps: usize,
    temperatures: Vec<f64>,
    first: Option<(f64, f64)>,
    last: Option<(f64, f64)>,
    largest_deviation: f64,
    state: Option<(usize, MdLayout)>,
    hops: HopTally,
}

/// The candidates for hops of a run of surface hopping, by outcome.
#[derive(Debug, Default)]
struct HopTally {
    seen: bool,
    candidates: usize,
    hops: usize,
    frustrated: usize,
    forbidden: usize,
    state: Option<usize>,
}

impl MdLog {
    /// Print to `log` every `every` steps of a run of `steps` steps (the first and the last
    /// always).
    pub fn new(log: Logger, every: usize, steps: usize) -> Self {
        Self {
            log,
            every: every.max(1),
            steps,
            temperatures: Vec::new(),
            first: None,
            last: None,
            largest_deviation: 0.0,
            state: None,
            hops: HopTally::default(),
        }
    }

    /// Also print the state of the run every `every` steps (and at the last): the structure
    /// (Cartesian coordinates in Angstrom, as MultiOptPy prints them) or the variables, the
    /// velocities, and the energies (see [`state_block`]).
    #[must_use]
    pub fn with_state(mut self, every: usize, layout: MdLayout) -> Self {
        self.state = Some((every.max(1), layout));
        self
    }

    fn state_due(&self, step: usize) -> bool {
        self.state
            .as_ref()
            .is_some_and(|(every, _)| step.is_multiple_of(*every) || step == self.steps)
    }

    fn print(&self, r: &MdStepReport) {
        let mut lines = vec![format!(
            "\n# STEP. {} ({} fs)\n",
            r.step,
            python_float(r.time_fs())
        )];
        lines.push(format!("potential energy : {:.12} hartree", r.energy));
        if r.energy != r.raw_energy {
            lines.push(format!(
                "potential energy without bias : {:.12} hartree",
                r.raw_energy
            ));
        }
        if let Some(state) = r.electronic_state {
            let energies: Vec<String> = r
                .state_energies
                .iter()
                .map(|(s, e)| format!("{s}: {e:.12}"))
                .collect();
            lines.push(format!(
                "electronic state : {state} (energies of the states {} hartree)",
                energies.join(", ")
            ));
        }
        lines.push(format!("kinetic energy : {:.12} hartree", r.kinetic));
        lines.push(format!(
            "hamiltonian : {} hartree",
            python_float(r.total_energy())
        ));
        lines.push(format!("conserved quantity : {:.12} hartree", r.conserved));
        lines.push(format!(
            "Instantaneous_temperature: {:.8} K (target {} K)",
            r.temperature,
            python_float(r.target_temperature)
        ));
        if let Some(p) = r.pressure {
            lines.push(format!(
                "pressure : {:.6e} a.u. ({:.6} GPa)",
                p,
                p * gigapascal_per_atomic_unit()
            ));
        }
        if let Some(v) = r.volume {
            lines.push(format!(
                "volume : {:.6} ang.^3",
                v * Units::BOHR_TO_ANGSTROM.powi(3)
            ));
        }
        if let Some(c) = r.constraint_residual {
            lines.push(format!("largest residual of the constraints : {c:.3e}"));
        }
        if let (true, Some((_, layout))) = (self.state_due(r.step), &self.state) {
            lines.push(state_block(r, layout));
        }
        for line in lines {
            self.log.info(&line);
        }
        for note in &r.notes {
            self.log.debug(note);
        }
    }
}

impl MdObserver for MdLog {
    fn on_step(&mut self, r: &MdStepReport) -> MoptResult<()> {
        if let Some(f) = &r.freeze {
            let values: Vec<String> = f.eigenvalues.iter().map(|v| format!("{v:.6e}")).collect();
            let overlaps: Vec<String> = f.overlaps.iter().map(|v| format!("{v:.4}")).collect();
            self.log.info(&format!(
                "Frozen modes at step {}: {} mode(s), mass-weighted eigenvalues {}{}",
                f.step,
                f.eigenvalues.len().max(f.overlaps.len()),
                if values.is_empty() {
                    "(given directions)".to_string()
                } else {
                    values.join(" ")
                },
                if overlaps.is_empty() {
                    String::new()
                } else {
                    format!(", overlaps with the modes before {}", overlaps.join(" "))
                }
            ));
        }
        for hop in &r.hops {
            self.log.info(&hop.describe());
            self.hops.candidates += 1;
            match hop.outcome {
                crate::md::hopping::HopOutcome::Hopped => self.hops.hops += 1,
                crate::md::hopping::HopOutcome::Frustrated => self.hops.frustrated += 1,
                crate::md::hopping::HopOutcome::Forbidden => self.hops.forbidden += 1,
                _ => {}
            }
        }
        if r.electronic_state.is_some() {
            self.hops.seen = true;
            self.hops.state = r.electronic_state;
        }
        if r.step.is_multiple_of(self.every) || r.step == self.steps || self.state_due(r.step) {
            self.print(r);
        }
        if r.step > 0 {
            self.temperatures.push(r.temperature);
        }
        let point = (r.time_fs(), r.conserved);
        if self.first.is_none() {
            self.first = Some(point);
        }
        if let Some((_, c0)) = self.first {
            self.largest_deviation = self.largest_deviation.max((r.conserved - c0).abs());
        }
        self.last = Some(point);
        Ok(())
    }

    fn on_finish(&mut self, end: Option<MdEnd>) -> MoptResult<()> {
        let n = self.temperatures.len();
        if n > 0 {
            let mean = self.temperatures.iter().sum::<f64>() / n as f64;
            let var = self
                .temperatures
                .iter()
                .map(|t| (t - mean).powi(2))
                .sum::<f64>()
                / n as f64;
            self.log.info(&format!(
                "Temperature over {n} step(s): mean {:.3} K, standard deviation {:.3} K",
                mean,
                libm::sqrt(var)
            ));
        }
        if let (Some((t0, c0)), Some((t1, c1))) = (self.first, self.last) {
            let ps = (t1 - t0) / 1000.0;
            self.log.info(&format!(
                "Conserved quantity: change {:.6e} hartree{}, largest deviation {:.6e} hartree",
                c1 - c0,
                if ps > 0.0 {
                    format!(" ({:.6e} hartree/ps)", (c1 - c0) / ps)
                } else {
                    String::new()
                },
                self.largest_deviation
            ));
        }
        if self.hops.seen {
            let h = &self.hops;
            self.log.info(&format!(
                "Surface hopping: {} candidate(s), {} hop(s), {} frustrated, {} classically \
                 forbidden; final electronic state {}",
                h.candidates,
                h.hops,
                h.frustrated,
                h.forbidden,
                h.state.map_or_else(|| "-".to_string(), |s| s.to_string())
            ));
        }
        if end.is_none() {
            self.log.info("The dynamics was interrupted.");
        }
        Ok(())
    }
}

/// The state of a step for the log: the structure (element and Cartesian coordinates in
/// Angstrom, in MultiOptPy's layout) or the plain variables, the velocities (atomic units,
/// Bohr per atomic unit of time), and the energies in Hartree: the kinetic energy, the energy
/// of the calculator, that energy with the bias potentials, the Hamiltonian (kinetic and
/// potential energy), and the conserved quantity (with the terms of the thermostat and the
/// barostat).
pub fn state_block(r: &MdStepReport, layout: &MdLayout) -> String {
    let mut s = String::new();
    match layout {
        MdLayout::Atoms(template) => {
            let symbols: Vec<&str> = template.elements.iter().map(|e| e.symbol()).collect();
            s.push_str("CARTESIAN COORDINATES (unit ang.):\n");
            for (i, symbol) in symbols.iter().enumerate() {
                let p: [f64; 3] = std::array::from_fn(|c| r.x[3 * i + c] * Units::BOHR_TO_ANGSTROM);
                s.push_str(&super::observers::coordinate_row(symbol, p));
            }
            if let Some(cell) = r
                .cell
                .as_ref()
                .or(template.cell.as_ref())
                .filter(|c| c.is_any_periodic())
            {
                s.push('\n');
                s.push_str(&super::observers::lattice_block(cell));
            }
            s.push_str("\nVELOCITIES (unit a.u.: Bohr per atomic unit of time):\n");
            for (i, symbol) in symbols.iter().enumerate() {
                s.push_str(&super::observers::coordinate_row(
                    symbol,
                    std::array::from_fn(|c| r.v[3 * i + c]),
                ));
            }
        }
        MdLayout::Variables(names) => {
            s.push_str("VARIABLES                VALUE               VELOCITY (unit a.u.)\n");
            for (i, name) in names.iter().enumerate() {
                s.push_str(&format!(
                    "{name:<10} {:>17.12}   {:>17.12}\n",
                    r.x[i], r.v[i]
                ));
            }
        }
    }
    s.push_str("\nENERGIES (unit hartree):\n");
    let line = |label: &str, value: f64| format!("{label:<23}: {value:>15.12} \n");
    s.push_str(&line("KINETIC ENERGY", r.kinetic));
    s.push_str(&line("ENERGY", r.raw_energy));
    s.push_str(&line("BIAS  ENERGY", r.energy));
    s.push_str(&line("HAMILTONIAN", r.total_energy()));
    s.push_str(&line(
        "THERMOSTAT, BAROSTAT",
        r.conserved - r.total_energy(),
    ));
    s.push_str(&line("CONSERVED QUANTITY", r.conserved));
    s
}

/// `-gi` for the steps of a run of atoms.
pub struct MdGeometryInfo {
    info: GeometryInfo,
    template: MolecularSystem,
}

impl MdGeometryInfo {
    /// For the atoms `atoms` of structures like `template`, files in `dir`.
    pub fn new(atoms: Vec<AtomId>, template: MolecularSystem, dir: &Path) -> Self {
        Self {
            info: GeometryInfo::new(atoms, dir),
            template,
        }
    }
}

impl MdObserver for MdGeometryInfo {
    fn on_step(&mut self, r: &MdStepReport) -> MoptResult<()> {
        let sys = self.template.with_flat_coords_bohr(&r.x)?;
        self.info.record(r.step, &sys, &r.raw_gradient, &r.gradient)
    }

    fn on_finish(&mut self, _end: Option<MdEnd>) -> MoptResult<()> {
        self.info.finish()
    }
}

/// Largest number of structures of a trajectory in the maps of `-cmds` and `-pca` (more
/// are thinned evenly: the distance matrix grows with their square).
pub const MAP_FRAMES: usize = 400;

/// `-cmds`, `-pca` for the frames of a trajectory.
pub struct MdMaps {
    maps: PathMaps,
    template: MolecularSystem,
    every: usize,
}

impl MdMaps {
    /// Maps of every `every`-th step of a run of `steps` steps (thinned to at most
    /// [`MAP_FRAMES`] structures), written into `dir`.
    pub fn new(
        cmds: bool,
        pca: bool,
        template: MolecularSystem,
        every: usize,
        steps: usize,
        dir: &Path,
        log: Logger,
    ) -> Self {
        let frames = steps / every.max(1) + 1;
        let thin = frames.div_ceil(MAP_FRAMES).max(1);
        Self {
            maps: PathMaps::new(cmds, pca, dir, log),
            template,
            every: every.max(1) * thin,
        }
    }
}

impl MdObserver for MdMaps {
    fn on_step(&mut self, r: &MdStepReport) -> MoptResult<()> {
        if r.step.is_multiple_of(self.every) {
            let sys = self.template.with_flat_coords_bohr(&r.x)?;
            self.maps.record(&sys, r.raw_energy, r.energy);
        }
        Ok(())
    }

    fn on_finish(&mut self, _end: Option<MdEnd>) -> MoptResult<()> {
        self.maps.finish()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::domain::SpinCharge;
    use crate::io::xyz::parse_xyz_str;
    use faer::Col;

    #[test]
    fn the_state_has_the_structure_the_velocities_and_the_energies() {
        let sys = parse_xyz_str("1\n\nAr 1.0 0.0 0.0\n", SpinCharge::default(), None).unwrap();
        let x = sys.coords_flat_bohr();
        let report = MdStepReport {
            step: 3,
            time: 30.0,
            x: x.clone(),
            v: Col::from_fn(3, |k| [1e-4, 0.0, -2e-4][k]),
            cell: None,
            energy: -0.5,
            raw_energy: -0.75,
            gradient: Col::zeros(3),
            raw_gradient: Col::zeros(3),
            kinetic: 0.125,
            temperature: 300.0,
            target_temperature: 300.0,
            dof: 3.0,
            conserved: -0.25,
            pressure: None,
            volume: None,
            stress: None,
            constraint_residual: None,
            freeze: None,
            notes: Vec::new(),
            dipole: None,
            heat_exchanged: 0.0,
            electronic_state: None,
            state_energies: Vec::new(),
            hops: Vec::new(),
        };
        let block = state_block(&report, &MdLayout::Atoms(Box::new(sys)));
        let lines: Vec<&str> = block.lines().collect();
        assert_eq!(lines[0], "CARTESIAN COORDINATES (unit ang.):");
        assert_eq!(
            lines[1],
            "Ar      1.000000000000      0.000000000000      0.000000000000"
        );
        assert!(lines[4].starts_with("Ar      0.000100000000"));
        assert!(block.contains("ENERGY                 : -0.750000000000 "));
        assert!(block.contains("BIAS  ENERGY           : -0.500000000000 "));
        assert!(block.contains("HAMILTONIAN            : -0.375000000000 "));
        assert!(block.contains("THERMOSTAT, BAROSTAT   :  0.125000000000 "));
        assert!(block.contains("CONSERVED QUANTITY     : -0.250000000000 "));
    }
}
