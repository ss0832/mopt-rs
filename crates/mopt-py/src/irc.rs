//! The intrinsic reaction coordinate from Python: the ask/tell state machine [`Irc`] over
//! the Cartesian coordinates of a structure or `n` plain variables, and parts of it that
//! can be used alone (the LQA step, the test of the first step, the names of the methods).
//!
//! MultiOptPy: `irc.py:IRC` and `IRC/` (`lqa.py`, `hpc.py`, `dvv.py`, `euler.py`, `rk4.py`,
//! `modekill.py`).
//!
//! This file is part of mopt-py (the Python interface of mopt-rs).
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::sync::Mutex;

use faer::{Col, Mat};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use mopt_rs::cli::constraints::resolve_for_run;
use mopt_rs::error::MoptError;
use mopt_rs::irc::{
    lqa_step, test_first_step, Branch, BranchEnd, IntegratorRegistry, Irc, IrcAction, IrcEvent,
    IrcSettings, IrcSpace, PointRole, PointValue, StartKind,
};
use mopt_rs::pes::constraint::ConstraintSet;

use super::err;

fn column(v: &[f64]) -> Col<f64> {
    Col::from_fn(v.len(), |i| v[i])
}

fn list(c: &Col<f64>) -> Vec<f64> {
    c.iter().copied().collect()
}

fn matrix(flat: &[f64], n: usize, what: &str) -> PyResult<Mat<f64>> {
    if flat.len() != n * n {
        return Err(err(MoptError::config(format!(
            "{what} has {} entries, expected {}",
            flat.len(),
            n * n
        ))));
    }
    Ok(Mat::from_fn(n, n, |i, j| flat[i * n + j]))
}

fn kind_name(kind: StartKind) -> &'static str {
    match kind {
        StartKind::TransitionState => "transition_state",
        StartKind::MetaIrc => "meta_irc",
        StartKind::ModeKill => "modekill",
    }
}

fn end_name(end: BranchEnd) -> &'static str {
    match end {
        BranchEnd::Converged => "converged",
        BranchEnd::Oscillation => "oscillation",
        BranchEnd::MaxSteps => "max_steps",
        BranchEnd::EnergyRise => "energy_rise",
        BranchEnd::ModesRemoved => "modes_removed",
        BranchEnd::NothingToRemove => "nothing_to_remove",
    }
}

fn role_name(role: PointRole) -> &'static str {
    match role {
        PointRole::TransitionState => "start",
        PointRole::Path => "path",
        PointRole::Auxiliary => "auxiliary",
        PointRole::FirstStep => "first_step",
        PointRole::FinalHessian => "final_hessian",
    }
}

fn event_dict<'py>(py: Python<'py>, event: &IrcEvent) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    match event {
        IrcEvent::Start(r) => {
            d.set_item("event", "start")?;
            d.set_item("kind", kind_name(r.kind))?;
            d.set_item("imaginary", r.imaginary)?;
            d.set_item("eigenvalues", r.eigenvalues.clone())?;
            d.set_item("transition_eigenvalue", r.transition_eigenvalue)?;
            d.set_item("transition_wavenumber", r.transition_wavenumber)?;
            d.set_item("transition_vector", r.transition_vector.as_ref().map(list))?;
            d.set_item("stationary", r.stationary)?;
            d.set_item("max_gradient", r.max_gradient)?;
            d.set_item("rms_gradient", r.rms_gradient)?;
            d.set_item("energy", r.energy)?;
            d.set_item("biased_energy", r.biased_energy)?;
        }
        IrcEvent::FirstStep(r) => {
            d.set_item("event", "first_step")?;
            d.set_item("branch", r.branch.label())?;
            d.set_item("length", r.length)?;
            d.set_item("angle_deg", r.test.angle_deg)?;
            d.set_item("rho", r.test.rho)?;
            d.set_item("predicted_change", r.test.predicted_change)?;
            d.set_item("actual_change", r.test.actual_change)?;
            d.set_item("corrected", r.corrected)?;
            d.set_item("iterations", r.iterations)?;
            d.set_item("converged", r.converged)?;
            d.set_item("moved", r.moved)?;
            d.set_item("energy_gain", r.energy_gain)?;
        }
        IrcEvent::Step { branch, record } => {
            d.set_item("event", "step")?;
            d.set_item("branch", branch.label())?;
            d.set_item("step", record.step)?;
            d.set_item("x", list(&record.x))?;
            d.set_item("energy", record.energy)?;
            d.set_item("biased_energy", record.biased_energy)?;
            d.set_item("rms_gradient", record.rms_gradient)?;
            d.set_item("rms_biased_gradient", record.rms_biased_gradient)?;
            d.set_item("arc_length", record.arc_length)?;
            d.set_item("bending_angle", record.bending_angle)?;
            let extras = PyDict::new(py);
            for (name, value) in &record.extras {
                extras.set_item(*name, value)?;
            }
            d.set_item("extras", extras)?;
            d.set_item(
                "curvature",
                record
                    .curvature
                    .as_ref()
                    .map(|c| (c.scalar, c.couplings.clone())),
            )?;
            d.set_item("in_table", record.in_table)?;
            d.set_item("notes", record.notes.clone())?;
        }
        IrcEvent::BranchFinished(s) => {
            d.set_item("event", "side_end")?;
            d.set_item("branch", s.branch.label())?;
            d.set_item("end", end_name(s.end))?;
            d.set_item("arrived", s.end.arrived())?;
            d.set_item("points", s.points)?;
            d.set_item("x", list(&s.x))?;
            d.set_item("dropped_points", s.counts.dropped)?;
            d.set_item("overshot_points", s.counts.overshot)?;
            d.set_item("shortened_steps", s.counts.shortened)?;
            d.set_item("refreshed_hessians", s.counts.refreshed)?;
            d.set_item("approach_steps", s.counts.approach)?;
        }
    }
    Ok(d)
}

/// Ask/tell IRC calculation from a structure (`symbols` given: standard atomic weights or
/// the `masses` of the atoms, rigid-body motion projected out unless `project_rigid=False`)
/// or from a point of `n` plain variables (`masses` of the variables, default one each). `x`
/// is in Bohr for a structure. `update_projection=False` leaves the rigid-body motion in the
/// updated Hessians (`-update_proj off`). `options` are the `KEY=VALUE` settings of `-irc_opt`; `method`
/// may carry options after colons. A structure may be periodic (`cell_bohr`, the rows a, b,
/// c, and `pbc`, as for `Optimizer`) and keep constraints along the path (`fix_atoms`,
/// `constraints`, `user_constraints`, `constraint_functions`, as for `Optimizer`): they are
/// resolved once at `x` (every target that is not given is the value at `x`), as `ircmain`
/// resolves them at its input, and every point of the path holds them to 1e-10.
#[pyclass(name = "Irc")]
pub struct PyIrc {
    inner: Mutex<Irc>,
}

impl PyIrc {
    /// The machine of `space` from `x` with the settings of the keyword arguments.
    fn with_space(
        space: IrcSpace,
        x: Vec<f64>,
        step_size: f64,
        max_steps: usize,
        method: &str,
        exact_hessian_interval: Option<usize>,
        options: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let mut settings =
            IrcSettings::from_multioptpy(&format!("{step_size:e}"), &max_steps.to_string(), method)
                .map_err(err)?;
        settings.exact_hessian_interval = exact_hessian_interval.filter(|&n| n > 0);
        let settings = settings
            .with_options(&options.unwrap_or_default())
            .map_err(err)?;
        let irc = Irc::new(settings, space, column(&x)).map_err(err)?;
        Ok(Self {
            inner: Mutex::new(irc),
        })
    }

    fn lock(&self) -> PyResult<std::sync::MutexGuard<'_, Irc>> {
        self.inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("the state of the IRC is poisoned"))
    }
}

#[pymethods]
impl PyIrc {
    #[new]
    #[pyo3(signature = (
        x,
        *,
        symbols = None,
        masses = None,
        project_rigid = true,
        update_projection = true,
        step_size = 0.1,
        max_steps = 300,
        method = "lqa",
        exact_hessian_interval = None,
        options = None,
        cell_bohr = None,
        pbc = None,
        fix_atoms = None,
        constraints = None,
        user_constraints = None,
        constraint_functions = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        x: Vec<f64>,
        symbols: Option<Vec<String>>,
        masses: Option<Vec<f64>>,
        project_rigid: bool,
        update_projection: bool,
        step_size: f64,
        max_steps: usize,
        method: &str,
        exact_hessian_interval: Option<usize>,
        options: Option<Vec<String>>,
        cell_bohr: Option<Vec<f64>>,
        pbc: Option<[bool; 3]>,
        fix_atoms: Option<Vec<usize>>,
        constraints: Option<Vec<String>>,
        user_constraints: Option<Vec<String>>,
        constraint_functions: Option<Vec<Bound<'_, PyAny>>>,
    ) -> PyResult<Self> {
        let held = super::held_constraints(
            fix_atoms,
            constraints,
            user_constraints,
            constraint_functions,
        )?;
        let cell = super::cell_with_axes(cell_bohr, pbc)?;
        let Some(symbols) = &symbols else {
            if cell.is_some() || !held.is_empty() {
                return Err(err(MoptError::unsupported(
                    "a cell and constraints of an IRC need atoms (symbols); plain variables \
                     have none",
                )));
            }
            let space = match &masses {
                Some(masses) => IrcSpace::with_masses(masses).map_err(err)?,
                None => IrcSpace::variables(x.len()),
            };
            return Self::with_space(
                space,
                x,
                step_size,
                max_steps,
                method,
                exact_hessian_interval,
                options,
            );
        };
        let sys = super::system(symbols, &x, 0, 1, cell)?;
        let space = match &masses {
            Some(masses) => {
                IrcSpace::molecule_with_masses(&sys, masses, project_rigid).map_err(err)?
            }
            None => IrcSpace::molecule(&sys, project_rigid),
        }
        .with_update_projection(update_projection);
        // Resolved once at the start, as ircmain resolves them at its input.
        let space = if held.is_empty() {
            space
        } else {
            let resolved = resolve_for_run(&held, &sys).map_err(err)?;
            let set = ConstraintSet::new(&resolved, &sys, None).map_err(err)?;
            space.with_constraints(set).map_err(err)?
        };
        Self::with_space(
            space,
            x,
            step_size,
            max_steps,
            method,
            exact_hessian_interval,
            options,
        )
    }

    /// The point to evaluate next, or `None` when the calculation has ended: a dict with
    /// `x`, `hessian` (whether the Hessian is asked for), `role` (`start`, `path`,
    /// `auxiliary`, `first_step`, `final_hessian`), `step`, and `branch` (`forward`,
    /// `backward`, `single`, or `None` for the start).
    fn ask<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let request = match self.lock()?.ask().map_err(err)? {
            IrcAction::Finished(_) => return Ok(None),
            IrcAction::Evaluate(request) => request,
        };
        let d = PyDict::new(py);
        d.set_item("x", list(&request.point.x))?;
        d.set_item("hessian", request.point.need_hessian)?;
        d.set_item("role", role_name(request.point.role))?;
        d.set_item("step", request.point.step)?;
        d.set_item("branch", request.branch.map(Branch::label))?;
        Ok(Some(d))
    }

    /// The surface at the point of `ask()`: the energy and gradient that are followed, the
    /// Hessian (flat, row-major) when it was asked for; with bias potentials also the
    /// energy and gradient without them (`raw_energy`, `raw_gradient`) and the Hessian of
    /// the bias potentials (`bias_hessian`), in which case `hessian` is the one without
    /// them.
    #[pyo3(signature = (
        energy,
        gradient,
        hessian = None,
        *,
        raw_energy = None,
        raw_gradient = None,
        bias_hessian = None,
    ))]
    fn tell(
        &self,
        energy: f64,
        gradient: Vec<f64>,
        hessian: Option<Vec<f64>>,
        raw_energy: Option<f64>,
        raw_gradient: Option<Vec<f64>>,
        bias_hessian: Option<Vec<f64>>,
    ) -> PyResult<()> {
        let mut irc = self.lock()?;
        let n = irc.space().dimension();
        let gradient = column(&gradient);
        let value = PointValue {
            energy,
            raw_energy: raw_energy.unwrap_or(energy),
            raw_gradient: raw_gradient
                .as_deref()
                .map_or_else(|| gradient.clone(), column),
            gradient,
            hessian: hessian
                .as_deref()
                .map(|h| matrix(h, n, "the Hessian"))
                .transpose()?,
            bias_hessian: bias_hessian
                .as_deref()
                .map(|h| matrix(h, n, "the Hessian of the bias potentials"))
                .transpose()?,
        };
        irc.tell(value).map_err(err)
    }

    /// What happened since the last call, as dicts with the key `event`: `start` (the
    /// starting structure was examined), `first_step` (the first point of a side was
    /// tested), `step` (a point of the path: `branch`, `step`, `x`, `energy`,
    /// `biased_energy`, `arc_length`, ...), `side_end`. Call it after `ask()` and after
    /// `tell()`.
    fn events<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let events = self.lock()?.drain_events();
        events.iter().map(|e| event_dict(py, e)).collect()
    }

    /// `transition_state`, `meta_irc`, or `modekill` (`None` before the start was told).
    #[getter]
    fn kind(&self) -> PyResult<Option<&'static str>> {
        Ok(self.lock()?.kind().map(kind_name))
    }

    /// The starting structure as the calculation uses it (a structure whose rigid-body
    /// motion is projected out is moved to its centre of mass).
    #[getter]
    fn start(&self) -> PyResult<Vec<f64>> {
        Ok(list(self.lock()?.start()))
    }

    /// Values told so far.
    #[getter]
    fn evaluations(&self) -> PyResult<usize> {
        Ok(self.lock()?.evaluations())
    }

    /// The registered name of the method.
    #[getter]
    fn method(&self) -> PyResult<String> {
        Ok(self.lock()?.settings().method.clone())
    }

    /// Mass-weighted coordinates of `x`.
    fn mass_weighted(&self, x: Vec<f64>) -> PyResult<Vec<f64>> {
        let irc = self.lock()?;
        let x = column(&x);
        irc.space().check(&x).map_err(err)?;
        Ok(list(&irc.space().to_mw(&x)))
    }
}

/// The LQA displacement in mass-weighted coordinates for a mass-weighted Hessian (flat,
/// row-major) and gradient: the steepest-descent path of the quadratic model, followed for
/// the length `step_size`.
#[pyfunction]
#[pyo3(signature = (hessian, gradient, step_size, gradient_floor = 1e-8))]
fn irc_lqa_step(
    hessian: Vec<f64>,
    gradient: Vec<f64>,
    step_size: f64,
    gradient_floor: f64,
) -> PyResult<Vec<f64>> {
    let g = column(&gradient);
    let h = matrix(&hessian, g.nrows(), "the Hessian")?;
    lqa_step(&h, &g, step_size, gradient_floor)
        .map(|s| list(&s))
        .map_err(err)
}

/// The test of a first step: the angle (degrees) between the downhill direction at the
/// trial point and the mass-weighted `displacement` from the transition state, and `rho`,
/// the actual over the predicted energy change. Returns `(angle_deg, rho, predicted_change,
/// actual_change)`.
#[pyfunction]
fn irc_first_step_test(
    displacement: Vec<f64>,
    eigenvalue: f64,
    energy_ts: f64,
    gradient_ts: Vec<f64>,
    energy_trial: f64,
    gradient_trial: Vec<f64>,
) -> (f64, f64, f64, f64) {
    let t = test_first_step(
        &column(&displacement),
        eigenvalue,
        energy_ts,
        &column(&gradient_ts),
        energy_trial,
        &column(&gradient_trial),
    );
    (t.angle_deg, t.rho, t.predicted_change, t.actual_change)
}

/// Names of the IRC methods.
#[pyfunction]
fn irc_methods() -> Vec<String> {
    IntegratorRegistry::builtin().names()
}

/// Add the classes and functions of this module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyIrc>()?;
    m.add_function(wrap_pyfunction!(irc_lqa_step, m)?)?;
    m.add_function(wrap_pyfunction!(irc_first_step_test, m)?)?;
    m.add_function(wrap_pyfunction!(irc_methods, m)?)?;
    Ok(())
}
