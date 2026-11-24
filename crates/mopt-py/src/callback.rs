//! Surface terms implemented in Python.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-py (the Python interface of mopt-rs).
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use mopt_rs::domain::{EvalOutput, MolecularSystem, Quantity};
use mopt_rs::error::{MoptError, MoptResult};
use mopt_rs::pes::{Frame, History, PesTerm, TermContext};

/// Finite-difference step (Bohr) for Hessians the callback does not return.
const FD_STEP: f64 = 1e-5;

/// A term whose energy and derivatives come from a Python callable
/// `f(positions_bohr, cell_bohr, iteration, need_hessian) -> (energy, gradient, hessian | None)`.
///
/// Positions and gradient are flat (3N, atomic units); the cell is the nine components of the
/// lattice vectors (rows) or `None`; the Hessian may be flat (9N^2) or nested. When the
/// callable returns no Hessian although one is needed, it is built from central differences of
/// the callable's gradient.
///
/// What the term depends on besides the structure is read from optional attributes of the
/// callable: `frame` (`"invariant"`: a function of distances and angles; `"translation_only"`;
/// `"absolute"`, the default: it may depend on where the structure is) and `history`
/// (`"static"`, `"iteration_ramp"`, `"time_dependent"`, `"stateful"`, the default). Without
/// them the term is taken to depend on everything, so that optimizers keep rigid-body
/// motions and the term is not shared between jobs.
pub struct PyCallbackTerm {
    label: String,
    callback: Py<PyAny>,
    frame: Frame,
    history: History,
}

/// The value of the optional string attribute `name` of `obj` (`None` when it has none).
fn attribute(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Option<String>> {
    if !obj.hasattr(name)? {
        return Ok(None);
    }
    let value = obj.getattr(name)?;
    if value.is_none() {
        return Ok(None);
    }
    Ok(Some(value.extract::<String>()?.to_ascii_lowercase()))
}

impl PyCallbackTerm {
    /// Term `label` calling `callback`, with the frame and history its attributes declare.
    pub fn new(label: &str, callback: Py<PyAny>) -> PyResult<Self> {
        let (frame, history) = Python::with_gil(|py| -> PyResult<(Frame, History)> {
            let obj = callback.bind(py);
            let bad = |what: &str, value: &str, allowed: &str| {
                PyValueError::new_err(format!(
                    "Python term '{label}': {what} = '{value}' is not one of {allowed}"
                ))
            };
            let frame = match attribute(obj, "frame")?.as_deref() {
                None | Some("absolute") => Frame::Absolute,
                Some("invariant") => Frame::Invariant,
                Some("translation_only") => Frame::TranslationOnly,
                Some(other) => {
                    return Err(bad(
                        "frame",
                        other,
                        "'invariant', 'translation_only', 'absolute'",
                    ))
                }
            };
            let history = match attribute(obj, "history")?.as_deref() {
                None | Some("stateful") => History::Stateful,
                Some("static") => History::Static,
                Some("iteration_ramp") => History::IterationRamp,
                Some("time_dependent") => History::TimeDependent,
                Some(other) => {
                    return Err(bad(
                        "history",
                        other,
                        "'static', 'iteration_ramp', 'time_dependent', 'stateful'",
                    ))
                }
            };
            Ok((frame, history))
        })?;
        Ok(Self {
            label: label.to_string(),
            callback,
            frame,
            history,
        })
    }

    fn call(
        &self,
        sys: &MolecularSystem,
        iteration: usize,
        need_hessian: bool,
    ) -> MoptResult<(f64, Vec<f64>, Option<Vec<f64>>)> {
        let positions: Vec<f64> = sys.coords_flat_bohr().iter().copied().collect();
        let cell: Option<Vec<f64>> = sys
            .active_cell()
            .map(|c| (0..3).flat_map(|k| c.lattice_vector(k).value()).collect());
        let fail = |e: PyErr| MoptError::calc(format!("Python term '{}': {e}", self.label));
        Python::with_gil(|py| {
            let r = self
                .callback
                .call1(py, (positions, cell, iteration, need_hessian))
                .map_err(fail)?;
            let (e, g, h): (f64, Bound<'_, PyAny>, Option<Bound<'_, PyAny>>) =
                r.extract(py).map_err(fail)?;
            let g = flat(&g).map_err(fail)?;
            let h = h.map(|h| flat(&h)).transpose().map_err(fail)?;
            Ok((e, g, h))
        })
    }
}

/// A flat list of floats from a sequence or a sequence of sequences (e.g. a NumPy array).
fn flat(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    if let Ok(v) = obj.extract::<Vec<f64>>() {
        return Ok(v);
    }
    Ok(obj.extract::<Vec<Vec<f64>>>()?.concat())
}

impl PesTerm for PyCallbackTerm {
    fn name(&self) -> &str {
        &self.label
    }

    fn evaluate(
        &self,
        sys: &MolecularSystem,
        ctx: &TermContext<'_>,
        need_hessian: bool,
    ) -> MoptResult<EvalOutput> {
        let n = sys.dof();
        let (e, g, h) = self.call(sys, ctx.iteration, need_hessian)?;
        if g.len() != n {
            return Err(MoptError::calc(format!(
                "Python term '{}' returned {} gradient components for {n} coordinates",
                self.label,
                g.len()
            )));
        }
        let hessian = match (need_hessian, h) {
            (false, _) => None,
            (true, Some(h)) if h.len() == n * n => Some(Mat::from_fn(n, n, |i, j| h[i * n + j])),
            (true, Some(h)) => {
                return Err(MoptError::calc(format!(
                    "Python term '{}' returned a Hessian with {} entries, expected {}",
                    self.label,
                    h.len(),
                    n * n
                )))
            }
            (true, None) => Some(self.fd_hessian(sys, ctx.iteration)?),
        };
        Ok(EvalOutput {
            energy: Quantity::new(e),
            gradient_bohr: Col::from_fn(n, |i| g[i]),
            hessian_bohr: hessian,
            orbitals: None,
            stress: None,
            strain_hessian: None,
        })
    }

    fn frame(&self) -> Frame {
        self.frame
    }

    fn history(&self) -> History {
        self.history
    }
}

impl PyCallbackTerm {
    fn fd_hessian(&self, sys: &MolecularSystem, iteration: usize) -> MoptResult<Mat<f64>> {
        let x = sys.coords_flat_bohr();
        let n = x.nrows();
        let mut h = Mat::zeros(n, n);
        for k in 0..n {
            let mut xp = x.clone();
            let mut xm = x.clone();
            xp[k] += FD_STEP;
            xm[k] -= FD_STEP;
            let gp = self
                .call(&sys.with_flat_coords_bohr(&xp)?, iteration, false)?
                .1;
            let gm = self
                .call(&sys.with_flat_coords_bohr(&xm)?, iteration, false)?
                .1;
            if gp.len() != n || gm.len() != n {
                return Err(MoptError::calc(format!(
                    "Python term '{}' changed its gradient size",
                    self.label
                )));
            }
            for i in 0..n {
                h[(i, k)] = (gp[i] - gm[i]) / (2.0 * FD_STEP);
            }
        }
        // Symmetrize.
        for i in 0..n {
            for j in 0..i {
                let v = 0.5 * (h[(i, j)] + h[(j, i)]);
                h[(i, j)] = v;
                h[(j, i)] = v;
            }
        }
        Ok(h)
    }
}
