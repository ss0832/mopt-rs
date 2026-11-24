//! The orientation search of `orientsearch` from Python: the ask/tell search over placements
//! of the parts of a complex (`OrientationSearch`), the comparison of two structures with
//! equivalent atoms and identical molecules exchanged (`compare_structures`), and the parts
//! of a complex (`complex_parts`).
//!
//! Positions are flat lists in Bohr, energies in Hartree; atoms and parts are numbered from 1.
//!
//! MultiOptPy: `Entrypoints/orientation_search.py` (the samples); the comparison and the pool
//!  of minima are new.
//!
//! This file is part of mopt-py (the Python interface of mopt-rs).
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::sync::Mutex;

use faer::Col;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use mopt_rs::domain::{MolecularSystem, Quantity, Units};
use mopt_rs::workflow::explore::orient::{
    ComplexParts, OrientAction, OrientSettings, OrientationSearch, SampleOrigin, SampleOutcome,
    Scheme,
};
use mopt_rs::workflow::explore::{PriorityRegistry, StructureMatcher};

use super::{cell_from, err, system};

fn list(c: &Col<f64>) -> Vec<f64> {
    c.iter().copied().collect()
}

fn one_based(parts: &[Vec<usize>]) -> Vec<Vec<usize>> {
    parts
        .iter()
        .map(|p| p.iter().map(|i| i + 1).collect())
        .collect()
}

fn lists(parts: &[Vec<usize>]) -> Vec<String> {
    parts
        .iter()
        .map(|p| {
            p.iter()
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
                .join(",")
        })
        .collect()
}

fn parts_of(
    sys: &MolecularSystem,
    parts: Option<Vec<Vec<usize>>>,
    fixed: &[usize],
    bond_scale: f64,
) -> PyResult<ComplexParts> {
    match parts {
        Some(p) => ComplexParts::from_lists(sys, &lists(&p), fixed, bond_scale),
        None => ComplexParts::molecules(sys, fixed, bond_scale),
    }
    .map_err(err)
}

/// The parts of a complex: `parts` (lists of atoms from 1, the rest one more part; `None`: the
/// molecules of the bond graph) as `atoms`, `fixed`, and `periodic` (bonded to their own
/// periodic images).
#[pyfunction]
#[pyo3(signature = (symbols, positions_bohr, parts = None, *, fixed = None, cell_bohr = None, bond_scale = 1.2))]
fn complex_parts<'py>(
    py: Python<'py>,
    symbols: Vec<String>,
    positions_bohr: Vec<f64>,
    parts: Option<Vec<Vec<usize>>>,
    fixed: Option<Vec<usize>>,
    cell_bohr: Option<Vec<f64>>,
    bond_scale: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let sys = system(&symbols, &positions_bohr, 0, 1, cell_from(cell_bohr)?)?;
    let p = parts_of(&sys, parts, &fixed.unwrap_or_default(), bond_scale)?;
    let d = PyDict::new(py);
    d.set_item("atoms", one_based(&p.atoms))?;
    d.set_item("fixed", p.fixed.clone())?;
    d.set_item("periodic", p.periodic.clone())?;
    Ok(d)
}

/// Compare two structures of the same atoms: `same_bonds` (their bond graphs agree up to a
/// permutation), and when they do the `rmsd` (Bohr) after matching equivalent atoms and
/// identical molecules and superposing, the matching `permutation` (atom `i` of `a` is atom
/// `permutation[i - 1]` of `b`, from 1), and whether `b` was `mirrored`.
#[pyfunction]
#[pyo3(signature = (symbols, a_bohr, b_bohr, *, cell_bohr = None, bond_scale = 1.2, mirror = true))]
fn compare_structures<'py>(
    py: Python<'py>,
    symbols: Vec<String>,
    a_bohr: Vec<f64>,
    b_bohr: Vec<f64>,
    cell_bohr: Option<Vec<f64>>,
    bond_scale: f64,
    mirror: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let cell = cell_from(cell_bohr)?;
    let a = system(&symbols, &a_bohr, 0, 1, cell.clone())?;
    let b = system(&symbols, &b_bohr, 0, 1, cell)?;
    let matcher = StructureMatcher {
        bond_scale,
        mirror_images: mirror,
        ..StructureMatcher::default()
    };
    let found = matcher.best_match(&a, &b).map_err(err)?;
    let d = PyDict::new(py);
    d.set_item("same_bonds", found.is_some())?;
    d.set_item("rmsd", found.as_ref().map(|m| m.rmsd.value()))?;
    d.set_item(
        "permutation",
        found
            .as_ref()
            .map(|m| m.permutation.iter().map(|j| j + 1).collect::<Vec<_>>()),
    )?;
    d.set_item("mirrored", found.as_ref().is_some_and(|m| m.mirrored))?;
    Ok(d)
}

struct SearchState {
    search: OrientationSearch,
    template: MolecularSystem,
    parts: ComplexParts,
}

/// Ask/tell orientation search: `ask()` gives the start of the next sample (a dict with
/// `index`, `positions_bohr`, `origin`) or `None` at the end; optimize it with any optimizer
/// and calculator and `tell(positions_bohr, energy, converged)` the result (`None` positions:
/// the optimization failed). `minima()` lists the distinct minima, lowest first.
#[pyclass(name = "OrientationSearch")]
pub struct PyOrientationSearch {
    inner: Mutex<SearchState>,
}

impl PyOrientationSearch {
    fn lock(&self) -> PyResult<std::sync::MutexGuard<'_, SearchState>> {
        self.inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("the state of the search is poisoned"))
    }
}

#[pymethods]
impl PyOrientationSearch {
    #[new]
    #[pyo3(signature = (symbols, positions_bohr, parts = None, *, samples = 5, scheme = "ring",
        seed = 1, distance_angstrom = 5.0, contact = (1.0, 1.2), clash = 0.5, refine = 0.0,
        priority = "tabu", rmsd_angstrom = 0.25, energy_tolerance = 1e-4, mirror = true,
        bond_scale = 1.2, fixed = None, cell_bohr = None, charge = 0, multiplicity = 1))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        symbols: Vec<String>,
        positions_bohr: Vec<f64>,
        parts: Option<Vec<Vec<usize>>>,
        samples: usize,
        scheme: &str,
        seed: u64,
        distance_angstrom: f64,
        contact: (f64, f64),
        clash: f64,
        refine: f64,
        priority: &str,
        rmsd_angstrom: f64,
        energy_tolerance: f64,
        mirror: bool,
        bond_scale: f64,
        fixed: Option<Vec<usize>>,
        cell_bohr: Option<Vec<f64>>,
        charge: i64,
        multiplicity: i64,
    ) -> PyResult<Self> {
        let template = system(
            &symbols,
            &positions_bohr,
            charge,
            multiplicity,
            cell_from(cell_bohr)?,
        )?;
        let p = parts_of(&template, parts, &fixed.unwrap_or_default(), bond_scale)?;
        let mut settings = OrientSettings {
            samples,
            refine,
            priority: priority.to_string(),
            seed,
            ..OrientSettings::default()
        };
        settings.placement.scheme = Scheme::parse(scheme).map_err(err)?;
        settings.placement.distance = Quantity::new(distance_angstrom * Units::ANGSTROM_TO_BOHR);
        settings.placement.contact = [contact.0, contact.1];
        settings.placement.clash = clash;
        settings.matcher = StructureMatcher {
            rmsd_tolerance: Quantity::new(rmsd_angstrom * Units::ANGSTROM_TO_BOHR),
            energy_tolerance: Quantity::new(energy_tolerance),
            bond_scale,
            mirror_images: mirror,
        };
        let search = OrientationSearch::new(&template, &p, settings, &PriorityRegistry::default())
            .map_err(err)?;
        Ok(Self {
            inner: Mutex::new(SearchState {
                search,
                template,
                parts: p,
            }),
        })
    }

    /// The start of the next sample, or `None` when all are done.
    fn ask<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let mut s = self.lock()?;
        let request = match s.search.ask().map_err(err)? {
            OrientAction::Finished => return Ok(None),
            OrientAction::Optimize(r) => r,
        };
        let d = PyDict::new(py);
        d.set_item("index", request.index)?;
        d.set_item("positions_bohr", list(&request.start.coords_flat_bohr()))?;
        d.set_item(
            "origin",
            match request.origin {
                SampleOrigin::Random { .. } => "random".to_string(),
                SampleOrigin::Perturbed { minimum } => format!("minimum {}", minimum + 1),
            },
        )?;
        Ok(Some(d))
    }

    /// The result of the sample asked for last; returns `minimum` (number of the distinct
    /// minimum it gave, from 1, in the order they were found; `None` when it did not
    /// converge), `new`, and `rmsd` (Bohr, to the known minimum it matched).
    #[pyo3(signature = (positions_bohr, energy, converged = true))]
    fn tell<'py>(
        &self,
        py: Python<'py>,
        positions_bohr: Option<Vec<f64>>,
        energy: f64,
        converged: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let mut s = self.lock()?;
        let outcome = match positions_bohr {
            None => None,
            Some(x) => {
                if x.len() != s.template.dof() {
                    return Err(PyValueError::new_err(format!(
                        "positions_bohr has {} numbers, expected {}",
                        x.len(),
                        s.template.dof()
                    )));
                }
                let flat = Col::from_fn(x.len(), |i| x[i]);
                Some(SampleOutcome {
                    system: s.template.with_flat_coords_bohr(&flat).map_err(err)?,
                    energy: Quantity::new(energy),
                    converged,
                })
            }
        };
        let record = s.search.tell(outcome).map_err(err)?;
        let d = PyDict::new(py);
        d.set_item("minimum", record.insertion.map(|i| i.index + 1))?;
        d.set_item("new", record.insertion.is_some_and(|i| i.new))?;
        d.set_item(
            "rmsd",
            record.insertion.and_then(|i| i.rmsd).map(|r| r.value()),
        )?;
        Ok(d)
    }

    /// The distinct minima, lowest first: `positions_bohr`, `energy`, `weight` (Boltzmann at
    /// `temperature` K), `hits`, `first_sample`, `sample`, `number` (from 1, in the order
    /// found).
    #[pyo3(signature = (temperature = 298.15))]
    fn minima<'py>(&self, py: Python<'py>, temperature: f64) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let s = self.lock()?;
        let pool = s.search.pool();
        let weights = pool.boltzmann_weights(temperature).map_err(err)?;
        pool.ranking()
            .into_iter()
            .map(|k| {
                let m = &pool.minima()[k];
                let d = PyDict::new(py);
                d.set_item("number", k + 1)?;
                d.set_item("positions_bohr", list(&m.system.coords_flat_bohr()))?;
                d.set_item("energy", m.energy.value())?;
                d.set_item("weight", weights[k])?;
                d.set_item("hits", m.hits)?;
                d.set_item("first_sample", m.first_trial)?;
                d.set_item("sample", m.trial)?;
                Ok(d)
            })
            .collect()
    }

    /// The parts (atoms from 1).
    #[getter]
    fn parts(&self) -> PyResult<Vec<Vec<usize>>> {
        Ok(one_based(&self.lock()?.parts.atoms))
    }
}

/// Add the classes and functions of this module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyOrientationSearch>()?;
    m.add_function(wrap_pyfunction!(complex_parts, m)?)?;
    m.add_function(wrap_pyfunction!(compare_structures, m)?)?;
    Ok(())
}
