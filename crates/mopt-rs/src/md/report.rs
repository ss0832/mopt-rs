//! What the molecular dynamics state machine asks for, what it is told, and what it reports
//! after every step.
//!
//! MultiOptPy: the values printed and collected by `moleculardynamics.py:MD.md`
//! (`hamiltonian`, `Instantaneous_temperature`, `tot_energy_list`).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use super::freeze::FreezeUpdate;
use super::hopping::{HopRecord, StateValues, StatesRequest};
use crate::domain::{PeriodicCell, Units};
use crate::pes::stress::Stress;

/// An evaluation the dynamics needs.
#[derive(Debug, Clone)]
pub struct MdRequest {
    /// Positions (Bohr) or values of the plain variables.
    pub x: Col<f64>,
    /// The cell at these positions (periodic structures).
    pub cell: Option<PeriodicCell>,
    /// Whether the Hessian is needed (mode freezing, eigenvector constraints).
    pub need_hessian: bool,
    /// Whether the stress is needed (a barostat).
    pub need_stress: bool,
    /// The step whose positions these are (0: the start).
    pub step: usize,
    /// The simulated time of these positions (atomic units), for time-dependent bias
    /// potentials.
    pub time: f64,
    /// The electronic states of surface hopping (`None`: one surface).
    pub states: Option<StatesRequest>,
}

/// The surface at the positions of an [`MdRequest`].
#[derive(Debug, Clone)]
pub struct MdValue {
    /// Energy of the whole surface (with bias potentials), Hartree.
    pub energy: f64,
    /// Its gradient.
    pub gradient: Col<f64>,
    /// Energy without bias potentials (the energy itself when there are none).
    pub raw_energy: f64,
    /// Gradient without bias potentials (`None`: the same as `gradient`).
    pub raw_gradient: Option<Col<f64>>,
    /// Hessian of the whole surface, when it was asked for.
    pub hessian: Option<Mat<f64>>,
    /// Stress of the whole surface (Hartree / Bohr^3), when the structure is periodic.
    pub stress: Option<Stress>,
    /// Dipole moment (e Bohr) from the atomic charges of the calculator, when it gives them.
    pub dipole: Option<[f64; 3]>,
    /// The states of surface hopping, when they were asked for (the active one is the
    /// surface above).
    pub states: Option<StateValues>,
}

impl MdValue {
    /// A surface without bias potentials.
    pub fn plain(energy: f64, gradient: Col<f64>) -> Self {
        Self {
            energy,
            gradient,
            raw_energy: energy,
            raw_gradient: None,
            hessian: None,
            stress: None,
            dipole: None,
            states: None,
        }
    }
}

/// The state after a step.
#[derive(Debug, Clone)]
pub struct MdStepReport {
    /// Step (0: the start).
    pub step: usize,
    /// Simulated time (atomic units).
    pub time: f64,
    /// Positions (Bohr) or values of the variables.
    pub x: Col<f64>,
    /// Velocities (Bohr per atomic time unit).
    pub v: Col<f64>,
    /// The cell (periodic structures).
    pub cell: Option<PeriodicCell>,
    /// Potential energy of the whole surface (Hartree).
    pub energy: f64,
    /// Potential energy without bias potentials.
    pub raw_energy: f64,
    /// Gradient of the whole surface.
    pub gradient: Col<f64>,
    /// Gradient without bias potentials.
    pub raw_gradient: Col<f64>,
    /// Kinetic energy.
    pub kinetic: f64,
    /// Instantaneous temperature (K).
    pub temperature: f64,
    /// Target temperature (K).
    pub target_temperature: f64,
    /// Degrees of freedom of the temperature.
    pub dof: f64,
    /// The conserved quantity: potential and kinetic energy, the energy of the thermostat
    /// variables or the work of the thermostat, and the energy of the barostat.
    pub conserved: f64,
    /// Instantaneous pressure (Hartree / Bohr^3), when the stress is known.
    pub pressure: Option<f64>,
    /// Volume of the cell (Bohr^3).
    pub volume: Option<f64>,
    /// Stress of the surface.
    pub stress: Option<Stress>,
    /// Largest residual of the constraints.
    pub constraint_residual: Option<f64>,
    /// A computation of the frozen modes in this step.
    pub freeze: Option<FreezeUpdate>,
    /// What the thermostat and barostat report.
    pub notes: Vec<String>,
    /// Dipole moment (e Bohr), when the calculator gives atomic charges.
    pub dipole: Option<[f64; 3]>,
    /// Kinetic energy moved from the cold to the hot slab by the velocity exchanges of a
    /// reverse non-equilibrium run so far (Hartree; 0 without them).
    pub heat_exchanged: f64,
    /// The active electronic state of surface hopping after this step (`None` without it).
    pub electronic_state: Option<usize>,
    /// The energies of the states of surface hopping at this step (Hartree, in the order of
    /// the states; empty without it).
    pub state_energies: Vec<(usize, f64)>,
    /// The candidates for hops decided at this step.
    pub hops: Vec<HopRecord>,
}

impl MdStepReport {
    /// Time in femtoseconds.
    pub fn time_fs(&self) -> f64 {
        self.time * Units::AU_TIME_TO_FS
    }

    /// Total energy (potential and kinetic).
    pub fn total_energy(&self) -> f64 {
        self.energy + self.kinetic
    }
}

/// How the dynamics ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MdEnd {
    /// All steps were made.
    Completed,
}

/// The next action of the dynamics.
// One request per step: boxing the large variant would save nothing.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone)]
pub enum MdAction {
    /// Evaluate the surface here and tell the result.
    Evaluate(MdRequest),
    /// The run has ended.
    Finished(MdEnd),
}
