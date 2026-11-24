//! Exploration methods of the mapper written in Python (`method=python:MODULE:CLASS`): a
//! Python class behind the [`Proposer`] trait of `mopt_rs::workflow::mapper::plugin`, through
//! the functions of `mopt_rs.mapper_plugins` (which document the class). The commands of the
//! Python package run `run_mapper` with [`parts`], whose method registry resolves the prefix
//! `python:`; the calls take the GIL for their duration only.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-py (the Python interface of mopt-rs).
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::sync::Arc;

use faer::Col;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use mopt_rs::cli::mapper::MapperParts;
use mopt_rs::core_infra::json::JsonValue;
use mopt_rs::domain::{Bohr, PeriodicCell, Units, Vector3};
use mopt_rs::error::{MoptError, MoptResult};
use mopt_rs::workflow::mapper::plugin::{Proposal, ProposalMethod, Proposer, DEFAULT_PATH_METHOD};
use mopt_rs::workflow::mapper::{
    Configuration, ExplorationMethod, Job, MethodSettings, Network, Node, Task,
};

const BRIDGE: &str = "mopt_rs.mapper_plugins";

/// The prefix of the names of the methods of Python classes.
pub const PREFIX: &str = "python:";

fn py_err(context: &str) -> impl Fn(PyErr) -> MoptError + '_ {
    move |e| MoptError::config(format!("{context}: {e}"))
}

/// The parts of the mapper of the commands of the Python package: the built-in ones, and the
/// methods of Python classes.
pub fn parts() -> MapperParts {
    let mut parts = MapperParts::default();
    parts.methods.register_prefix(PREFIX, Arc::new(make_method));
    parts
}

/// The method `python:MODULE:CLASS` (`name`) with the settings of the run: `method_path` and
/// `method_path_KEY=VALUE` choose the chain method of its paths, every other `method_KEY` goes
/// to the class.
fn make_method(name: &str, s: &MethodSettings) -> MoptResult<Box<dyn ExplorationMethod>> {
    let spec = name.strip_prefix(PREFIX).unwrap_or(name);
    let path_method = s
        .extra
        .get("path")
        .cloned()
        .unwrap_or_else(|| DEFAULT_PATH_METHOD.to_string());
    let path_settings: Vec<(String, String)> = s
        .extra
        .iter()
        .filter_map(|(k, v)| k.strip_prefix("path_").map(|k| (k.to_string(), v.clone())))
        .collect();
    let fail = py_err(name);
    let object = Python::with_gil(|py| -> PyResult<Py<PyAny>> {
        let settings = PyDict::new(py);
        for (k, v) in &s.extra {
            if k != "path" && !k.starts_with("path_") {
                settings.set_item(k, v)?;
            }
        }
        let made = py
            .import(BRIDGE)?
            .getattr("make")?
            .call1((spec, settings))?;
        Ok(made.unbind())
    })
    .map_err(fail)?;
    Ok(Box::new(ProposalMethod::new(
        Box::new(PyProposer {
            name: name.to_string(),
            object,
        }),
        &path_method,
        path_settings,
        s.top_n,
    )))
}

/// A Python class of `mopt_rs.mapper_plugins` as a proposer.
struct PyProposer {
    name: String,
    object: Py<PyAny>,
}

/// A point of the network as the Python side sees it.
fn point_dict<'py>(
    py: Python<'py>,
    point: &Configuration,
    id: Option<usize>,
    label: &str,
    energy: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("id", id)?;
    d.set_item("label", label)?;
    d.set_item("energy", energy)?;
    match point {
        Configuration::Structure(sys) => {
            let symbols: Vec<&str> = sys.elements.iter().map(|e| e.symbol()).collect();
            let x = sys.coords_flat_bohr();
            let rows: Vec<[f64; 3]> = (0..sys.n_atoms())
                .map(|a| std::array::from_fn(|c| x[3 * a + c] * Units::BOHR_TO_ANGSTROM))
                .collect();
            let cell: Option<Vec<[f64; 3]>> = sys.active_cell().map(|c| {
                (0..3)
                    .map(|k| {
                        c.lattice_vector(k)
                            .value()
                            .map(|v| v * Units::BOHR_TO_ANGSTROM)
                    })
                    .collect()
            });
            d.set_item("symbols", symbols)?;
            d.set_item("positions", rows)?;
            d.set_item("charge", sys.spin_charge.charge)?;
            d.set_item("multiplicity", sys.spin_charge.multiplicity)?;
            d.set_item("cell", cell)?;
            d.set_item("variables", py.None())?;
        }
        Configuration::Variables(p) => {
            let values: Vec<f64> = p.values.iter().copied().collect();
            d.set_item("variables", values)?;
            d.set_item("names", p.names.as_ref().clone())?;
            d.set_item("symbols", py.None())?;
        }
    }
    Ok(d)
}

/// The point of a proposal: flat positions in Bohr (or the values of the variables), and the
/// lattice vectors in Bohr where it has a cell of its own.
type PyPoint = (Vec<f64>, Option<Vec<f64>>);

/// The configuration of a point of a proposal, like `start`.
fn configuration(start: &Configuration, (flat, cell): &PyPoint) -> MoptResult<Configuration> {
    let mut point = start.with_flat(&Col::from_fn(flat.len(), |i| flat[i]))?;
    if let (Some(cell), Configuration::Structure(sys)) = (cell, &mut point) {
        let periodic = sys.cell.as_ref().map_or([true; 3], |c| c.periodic_axes);
        let v = |k: usize| Vector3::<Bohr>::new([cell[3 * k], cell[3 * k + 1], cell[3 * k + 2]]);
        sys.cell = Some(PeriodicCell::from_lattice_vectors(
            v(0),
            v(1),
            v(2),
            periodic,
        )?);
    }
    Ok(point)
}

impl Proposer for PyProposer {
    fn name(&self) -> &str {
        &self.name
    }

    fn tasks(&self, node: &Node, _network: &Network) -> MoptResult<Vec<Task>> {
        let found: Vec<(String, String, String)> = Python::with_gil(|py| {
            let d = point_dict(py, &node.system, Some(node.id), &node.label(), node.energy)?;
            py.import(BRIDGE)?
                .getattr("tasks")?
                .call1((self.object.clone_ref(py), d))?
                .extract()
        })
        .map_err(py_err(&self.name))?;
        found
            .into_iter()
            .map(|(key, description, parameters)| {
                Ok(Task {
                    key,
                    description,
                    parameters: JsonValue::parse(&parameters)?,
                })
            })
            .collect()
    }

    fn propose(&self, job: &Job) -> MoptResult<Proposal> {
        type Found = (Vec<PyPoint>, Vec<Vec<PyPoint>>, Vec<String>);
        let (guesses, paths, notes): Found = Python::with_gil(|py| {
            let task = PyDict::new(py);
            task.set_item("key", &job.task.key)?;
            task.set_item("description", &job.task.description)?;
            let json = py.import("json")?;
            task.set_item(
                "parameters",
                json.getattr("loads")?
                    .call1((job.task.parameters.to_string_compact(),))?,
            )?;
            let d = PyDict::new(py);
            d.set_item("id", job.id)?;
            d.set_item("node", job.node)?;
            d.set_item("task", task)?;
            d.set_item(
                "start",
                point_dict(py, &job.start, Some(job.node), "start", 0.0)?,
            )?;
            py.import(BRIDGE)?
                .getattr("propose")?
                .call1((self.object.clone_ref(py), d))?
                .extract()
        })
        .map_err(py_err(&self.name))?;
        Ok(Proposal {
            guesses: guesses
                .iter()
                .map(|p| configuration(&job.start, p))
                .collect::<MoptResult<_>>()?,
            paths: paths
                .iter()
                .map(|path| {
                    path.iter()
                        .map(|p| configuration(&job.start, p))
                        .collect::<MoptResult<Vec<_>>>()
                })
                .collect::<MoptResult<_>>()?,
            notes,
        })
    }
}
