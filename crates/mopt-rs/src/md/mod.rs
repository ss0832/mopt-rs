//! Molecular dynamics (layer L4: algorithms without I/O).
//!
//! [`MolecularDynamics`] is an ask/tell state machine over the coordinates of atoms or any
//! number of plain variables (unit masses). Its parts are independent extension points:
//! - [`integrator::Integrator`]: the splitting of a step around one evaluation of the forces
//!   (velocity Verlet, BAOAB);
//! - [`thermostat::Thermostat`]: none (NVE), Berendsen, Bussi's velocity rescaling,
//!   Langevin, Andersen, Nose-Hoover chains;
//! - [`barostat::Barostat`]: Berendsen (isotropic and anisotropic), the stochastic cell
//!   rescaling, Martyna-Tobias-Klein (NPT with the cell following the pressure);
//! - [`constraints::Rattle`]: holonomic constraints (SHAKE and RATTLE) for every constraint
//!   of `optmain` (`-pc`, `-gfix`, `-uc`, Python functions), atoms of `-fix` held by zero
//!   inverse masses;
//! - [`freeze::ModeFreezing`]: no motion along chosen modes of the mass-weighted Hessian;
//! - [`cv::Cv`]: collective variables, and [`cv::Umbrella`], the restraint of a window of
//!   umbrella sampling (`workflow::umbrella`, `analysis::free_energy`);
//! - [`hopping::SurfaceHopping`]: trajectory surface hopping among electronic states by the
//!   Zhu-Nakamura theory, without nonadiabatic couplings.
//!
//! [`generalized::GeneralizedDynamics`] is the same kind of machine in the generalized
//! coordinates of a non-redundant coordinate system (the kinetic energy of the metric
//! `G = B M^-1 B^T`, the generalized leapfrog). Both are [`MdMachine`]s for the driver.
//!
//! The surface it moves on is whatever the caller evaluates: a calculator with bias
//! potentials, a combination of electronic states, an objective of plain variables (see
//! `workflow::md`); time-dependent bias potentials receive the simulated time of the request.
//!
//! MultiOptPy: `moleculardynamics.py` and `MD/thermostat.py`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

pub mod barostat;
pub mod constraints;
pub mod cv;
// Collective variables from text (-cv, -cv_bias, -umbrella cv=).
pub mod cv_spec;
pub mod freeze;
// Dynamics in generalized coordinates (mdmain -coord).
pub mod generalized;
// Trajectory surface hopping by the Zhu-Nakamura theory.
pub mod hopping;
pub mod integrator;
// LINCS, an alternative to SHAKE/RATTLE for bond lengths.
pub mod lincs;
pub mod machine;
pub mod report;
pub mod settings;
pub mod space;
pub mod thermostat;
// Initial conditions from the harmonic Wigner distribution.
pub mod wigner;

#[cfg(test)]
mod tests;

pub use barostat::{Barostat, BarostatRegistry, BarostatSettings};
pub use constraints::{LinearRow, Rattle};
pub use cv::{Cv, Umbrella};
pub use freeze::{FreezeSettings, ModeFreezing, ModeSelection};
pub use hopping::{HoppingSettings, SurfaceHopping};
pub use integrator::{Integrator, IntegratorRegistry, Stage};
pub use machine::{MdParts, MolecularDynamics};
pub use report::{MdAction, MdEnd, MdRequest, MdStepReport, MdValue};
pub use settings::{InitialVelocities, MdSettings};
pub use space::{MdSpace, RigidMotion};
pub use thermostat::{Thermostat, ThermostatRegistry, ThermostatSettings};

/// An ask/tell molecular dynamics machine, for the driver (`workflow::md::run_md`).
pub trait MdMachine {
    /// The next action.
    fn ask(&self) -> MdAction;

    /// The surface at the positions of [`Self::ask`]; the report of the step it completes.
    fn tell(&mut self, value: MdValue) -> crate::error::MoptResult<MdStepReport>;

    /// Whether the run has ended.
    fn finished(&self) -> bool;

    /// Evaluations told so far.
    fn evaluations(&self) -> usize;

    /// Degrees of freedom of the temperature.
    fn dof(&self) -> f64;

    /// Number of frozen modes.
    fn frozen_modes(&self) -> usize {
        0
    }
}

impl MdMachine for MolecularDynamics {
    fn ask(&self) -> MdAction {
        MolecularDynamics::ask(self)
    }

    fn tell(&mut self, value: MdValue) -> crate::error::MoptResult<MdStepReport> {
        MolecularDynamics::tell(self, value)
    }

    fn finished(&self) -> bool {
        MolecularDynamics::finished(self)
    }

    fn evaluations(&self) -> usize {
        MolecularDynamics::evaluations(self)
    }

    fn dof(&self) -> f64 {
        MolecularDynamics::dof(self)
    }

    fn frozen_modes(&self) -> usize {
        MolecularDynamics::frozen_modes(self)
    }
}
