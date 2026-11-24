//! The parts of the conformation search one by one from Python: bond graphs (classes of
//! equivalent atoms, isomorphism), the RMSD that does not depend on the numbering of equivalent
//! atoms, the conformer ensemble (duplicates, rotamers, populations, conformational entropy),
//! Z-matrices and the crossing of conformers, and the RMSD metadynamics bias. The command
//! itself runs through `run_cli("confsearch", ...)`.
//!
//! Positions are flat lists in Bohr, energies in Hartree; atoms are numbered from 1.
//!
//! MultiOptPy: no counterpart (new in mopt-rs); MultiOptPy's conformation search cannot be
//! taken apart.
//!
//! This file is part of mopt-py (the Python interface of mopt-rs).
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::sync::Mutex;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use mopt_rs::coord::graph::{MolecularGraph, MULTIOPTPY_BOND_SCALE};
use mopt_rs::coord::invariant_rmsd::{invariant_rmsd as rmsd_match, RmsdOptions};
use mopt_rs::coord::zmatrix::ZMatrix;
use mopt_rs::core_infra::rng::Rng;
use mopt_rs::domain::{MolecularSystem, Units};
use mopt_rs::error::{MoptError, MoptResult};
use mopt_rs::pes::bias::RmsdMetadynamics;
use mopt_rs::pes::{PesTerm, TermContext};
use mopt_rs::workflow::confsearch::moves::{ConformerMove, MoveContext, MoveRegistry, Proposal};
use mopt_rs::workflow::explore::{
    crossover_candidates, CompareSettings, ConformerEnsemble, CrossoverSettings, Verdict,
};

use super::{cell_from, err, system};

fn flat(sys: &MolecularSystem) -> Vec<f64> {
    sys.coords_flat_bohr().iter().copied().collect()
}

fn one_based(list: &[usize]) -> Vec<usize> {
    list.iter().map(|a| a + 1).collect()
}

/// The bond graph of a structure: `bonds` (pairs from 1), `classes` (one per atom, equal for
/// atoms equivalent by colour refinement), and the bonded `fragments`; two atoms are bonded
/// within `scale` times the sum of their covalent radii (MultiOptPy's 1.1).
#[pyfunction]
#[pyo3(signature = (symbols, positions_bohr, scale = MULTIOPTPY_BOND_SCALE, cell_bohr = None))]
fn bond_graph<'py>(
    py: Python<'py>,
    symbols: Vec<String>,
    positions_bohr: Vec<f64>,
    scale: f64,
    cell_bohr: Option<Vec<f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let sys = system(&symbols, &positions_bohr, 0, 1, cell_from(cell_bohr)?)?;
    let g = MolecularGraph::from_system(&sys, scale);
    let d = PyDict::new(py);
    d.set_item(
        "bonds",
        g.bonds().iter().map(|b| one_based(b)).collect::<Vec<_>>(),
    )?;
    d.set_item("classes", g.classes())?;
    d.set_item(
        "fragments",
        g.fragments()
            .iter()
            .map(|f| one_based(f))
            .collect::<Vec<_>>(),
    )?;
    Ok(d)
}

/// A renumbering under which the bond graphs of two structures of the same atoms agree:
/// atom `i` of the first is atom `map[i - 1]` of the second (from 1); `None` when they differ.
#[pyfunction]
#[pyo3(signature = (symbols, first_bohr, second_bohr, scale = MULTIOPTPY_BOND_SCALE))]
fn graph_isomorphism(
    symbols: Vec<String>,
    first_bohr: Vec<f64>,
    second_bohr: Vec<f64>,
    scale: f64,
) -> PyResult<Option<Vec<usize>>> {
    let a = system(&symbols, &first_bohr, 0, 1, None)?;
    let b = system(&symbols, &second_bohr, 0, 1, None)?;
    let (ga, gb) = (
        MolecularGraph::from_system(&a, scale),
        MolecularGraph::from_system(&b, scale),
    );
    Ok(ga.isomorphism(&gb).map(|m| one_based(&m)))
}

/// RMSD (Bohr) of two structures after the best rotation and, with `permutations`, the best
/// renumbering of equivalent atoms (their bond-graph classes), with `mirror` also of the
/// mirror image of the first: `(rmsd, permutation from 1, mirrored)`.
#[pyfunction]
#[pyo3(signature = (symbols, first_bohr, second_bohr, permutations = true, mirror = false))]
fn invariant_rmsd(
    symbols: Vec<String>,
    first_bohr: Vec<f64>,
    second_bohr: Vec<f64>,
    permutations: bool,
    mirror: bool,
) -> PyResult<(f64, Vec<usize>, bool)> {
    let a = system(&symbols, &first_bohr, 0, 1, None)?;
    let b = system(&symbols, &second_bohr, 0, 1, None)?;
    let ca = MolecularGraph::from_system(&a, MULTIOPTPY_BOND_SCALE).classes();
    let cb = MolecularGraph::from_system(&b, MULTIOPTPY_BOND_SCALE).classes();
    let options = RmsdOptions {
        permutations,
        mirror,
        rotation: true,
    };
    let m = rmsd_match(&a, &b, &ca, &cb, &options).map_err(err)?;
    Ok((m.rmsd, one_based(&m.permutation), m.mirrored))
}

/// The rows of the Z-matrix of a structure along its bonds, `(atom, bond, angle, dihedral)`
/// from 1 (0 where a row has no such atom), and their values (Bohr, rad).
#[pyfunction]
#[allow(clippy::type_complexity)]
fn zmatrix(
    symbols: Vec<String>,
    positions_bohr: Vec<f64>,
) -> PyResult<(Vec<[usize; 4]>, Vec<[f64; 3]>)> {
    let sys = system(&symbols, &positions_bohr, 0, 1, None)?;
    let g = MolecularGraph::from_system(&sys, MULTIOPTPY_BOND_SCALE);
    let z = ZMatrix::from_graph(&g, &sys).map_err(err)?;
    let rows = z
        .rows()
        .iter()
        .map(|r| {
            let at = |x: Option<usize>| x.map_or(0, |i| i + 1);
            [r.atom + 1, at(r.bond), at(r.angle), at(r.dihedral)]
        })
        .collect();
    Ok((rows, z.values(&sys)))
}

/// New candidates `Z_ref + Z_i - Z_j` from the reference conformer and the parents (the
/// genetic crossing of iMTD-GC), each with the bonds of the reference and apart from the
/// others by `rmsd_bohr`; at most `max` pairs, shuffled by `seed`.
#[pyfunction]
#[pyo3(signature = (symbols, reference_bohr, parents_bohr, max = 300, rmsd_bohr = None, seed = 1))]
fn crossover_conformers(
    symbols: Vec<String>,
    reference_bohr: Vec<f64>,
    parents_bohr: Vec<Vec<f64>>,
    max: usize,
    rmsd_bohr: Option<f64>,
    seed: u64,
) -> PyResult<Vec<Vec<f64>>> {
    let reference = system(&symbols, &reference_bohr, 0, 1, None)?;
    let parents = parents_bohr
        .iter()
        .map(|p| system(&symbols, p, 0, 1, None))
        .collect::<PyResult<Vec<_>>>()?;
    let mut settings = CrossoverSettings {
        max,
        ..CrossoverSettings::default()
    };
    if let Some(r) = rmsd_bohr {
        settings.rmsd = r;
    }
    let found = crossover_candidates(&reference, &parents, &settings, &mut Rng::from_seed(seed))
        .map_err(err)?;
    Ok(found.iter().map(flat).collect())
}

/// An ensemble of minima: duplicates removed, rotamers counted with their conformers,
/// Boltzmann populations, and the conformational entropy (thresholds in Hartree, Bohr, and
/// a share; CREGEN's 0.05 kcal/mol, 0.125 Angstrom, and 1 % without them).
#[pyclass(name = "ConformerEnsemble")]
pub struct PyConformerEnsemble {
    inner: Mutex<ConformerEnsemble>,
}

#[pymethods]
impl PyConformerEnsemble {
    #[new]
    #[pyo3(signature = (energy = None, rmsd = None, rotational = None, permutations = true, mirror = true))]
    fn new(
        energy: Option<f64>,
        rmsd: Option<f64>,
        rotational: Option<f64>,
        permutations: bool,
        mirror: bool,
    ) -> Self {
        let d = CompareSettings::default();
        Self {
            inner: Mutex::new(ConformerEnsemble::new(CompareSettings {
                energy: energy.unwrap_or(d.energy),
                rmsd: rmsd.unwrap_or(d.rmsd),
                rotational: rotational.unwrap_or(d.rotational),
                permutations,
                mirror,
                ..d
            })),
        }
    }

    /// Add a minimum: `("new", index)`, `("duplicate", of)`, or `("rotamer", of)` (from 0).
    #[pyo3(signature = (symbols, positions_bohr, energy, origin = ""))]
    fn add(
        &self,
        symbols: Vec<String>,
        positions_bohr: Vec<f64>,
        energy: f64,
        origin: &str,
    ) -> PyResult<(String, usize)> {
        let sys = system(&symbols, &positions_bohr, 0, 1, None)?;
        let mut e = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("the ensemble is poisoned"))?;
        Ok(match e.add(&sys, energy, origin).map_err(err)? {
            Verdict::New(i) => ("new".to_string(), i),
            Verdict::Duplicate { of, .. } => ("duplicate".to_string(), of),
            Verdict::Rotamer { of, .. } => ("rotamer".to_string(), of),
        })
    }

    /// Energies of the members in the order they were found.
    fn energies(&self) -> PyResult<Vec<f64>> {
        let e = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("the ensemble is poisoned"))?;
        Ok(e.members().iter().map(|m| m.energy).collect())
    }

    /// Populations, average energy and free energy above the lowest (Hartree), and the
    /// conformational entropy (J / (mol K)) at `temperature` (K).
    #[pyo3(signature = (temperature = 298.15))]
    fn thermo<'py>(&self, py: Python<'py>, temperature: f64) -> PyResult<Bound<'py, PyDict>> {
        let e = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("the ensemble is poisoned"))?;
        let t = e.thermo(temperature).map_err(err)?;
        let d = PyDict::new(py);
        d.set_item("populations", t.populations)?;
        d.set_item("average_energy", t.average_energy)?;
        d.set_item("entropy", t.entropy)?;
        d.set_item("free_energy", t.free_energy)?;
        d.set_item(
            "free_energy_kcal_mol",
            t.free_energy * Units::HARTREE_TO_KCAL_MOL,
        )?;
        Ok(d)
    }

    fn __len__(&self) -> PyResult<usize> {
        Ok(self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("the ensemble is poisoned"))?
            .len())
    }
}

/// The RMSD metadynamics bias `sum_i k exp(-alpha RMSD_i^2)` over reference structures
/// (`height` in Hartree, `alpha` in Bohr^-2; `atoms` from 1, all without them).
#[pyclass(name = "RmsdMetadynamics")]
pub struct PyRmsdMetadynamics {
    inner: RmsdMetadynamics,
}

#[pymethods]
impl PyRmsdMetadynamics {
    #[new]
    #[pyo3(signature = (height, alpha, atoms = None))]
    fn new(height: f64, alpha: f64, atoms: Option<Vec<usize>>) -> PyResult<Self> {
        let mut inner = RmsdMetadynamics::new(height, alpha).map_err(err)?;
        if let Some(list) = atoms {
            let ids = list
                .iter()
                .map(|&a| mopt_rs::domain::AtomId::from_one_based(a))
                .collect::<Result<Vec<_>, _>>()
                .map_err(err)?;
            inner = inner.on_atoms(&ids);
        }
        Ok(Self { inner })
    }

    /// Add a reference structure.
    fn add_reference(&self, symbols: Vec<String>, positions_bohr: Vec<f64>) -> PyResult<()> {
        let sys = system(&symbols, &positions_bohr, 0, 1, None)?;
        self.inner.add_reference(&sys).map_err(err)
    }

    /// Number of references.
    #[getter]
    fn references(&self) -> usize {
        self.inner.references()
    }

    /// Energy (Hartree) and gradient (Hartree/Bohr) of the bias at `positions_bohr`.
    fn evaluate(
        &self,
        symbols: Vec<String>,
        positions_bohr: Vec<f64>,
    ) -> PyResult<(f64, Vec<f64>)> {
        let sys = system(&symbols, &positions_bohr, 0, 1, None)?;
        let out = self
            .inner
            .evaluate(&sys, &TermContext::at_iteration(0, &sys), false)
            .map_err(err)?;
        Ok((
            out.energy.value(),
            out.gradient_bohr.iter().copied().collect(),
        ))
    }
}

/// A move of a conformer search made by a Python callable
/// `f(symbols, positions_bohr, trial, conformer, seed)`: the symbols of the atoms, the flat
/// positions of the conformer the trial starts from (3N, Bohr), the number of the trial (from
/// 0), the number of that conformer in the order of discovery, and a seed drawn from the
/// random numbers of the search (the same search gives the same seeds). It returns the
/// positions to minimize (3N, flat or nested), or `(positions, text)` with a description for
/// the log, or `None` when there is nothing more to try (the search ends then).
struct PyMove {
    name: String,
    callback: Py<PyAny>,
}

impl ConformerMove for PyMove {
    fn name(&self) -> &str {
        &self.name
    }

    fn propose(
        &mut self,
        current: &MolecularSystem,
        context: &MoveContext<'_>,
        rng: &mut Rng,
    ) -> MoptResult<Option<Proposal>> {
        let symbols: Vec<String> = current
            .elements
            .iter()
            .map(|e| e.symbol().to_string())
            .collect();
        let positions = flat(current);
        let seed = rng.next_u64() >> 1;
        let fail =
            |e: PyErr| MoptError::calc(format!("Python conformer move '{}': {e}", self.name));
        let found = Python::with_gil(|py| -> MoptResult<(Vec<f64>, Option<String>)> {
            let r = self
                .callback
                .call1(
                    py,
                    (symbols, positions, context.trial, context.conformer, seed),
                )
                .map_err(fail)?;
            let r = r.bind(py);
            if r.is_none() {
                return Ok((Vec::new(), None));
            }
            let coordinates = |obj: &Bound<'_, PyAny>| -> PyResult<Vec<f64>> {
                if let Ok(v) = obj.extract::<Vec<f64>>() {
                    return Ok(v);
                }
                Ok(obj.extract::<Vec<Vec<f64>>>()?.concat())
            };
            if let Ok((x, text)) = r.extract::<(Bound<'_, PyAny>, String)>() {
                return Ok((coordinates(&x).map_err(fail)?, Some(text)));
            }
            Ok((coordinates(r).map_err(fail)?, None))
        })?;
        let (moved, what) = found;
        if moved.is_empty() {
            return Ok(None);
        }
        if moved.len() != current.dof() {
            return Err(MoptError::calc(format!(
                "Python conformer move '{}': {} coordinates for {} atoms",
                self.name,
                moved.len(),
                current.n_atoms()
            )));
        }
        let x = faer::Col::from_fn(moved.len(), |i| moved[i]);
        Ok(Some(Proposal {
            start: current.with_flat_coords_bohr(&x)?,
            what: what.unwrap_or_else(|| format!("the move {}", self.name)),
        }))
    }
}

/// Run `confsearch` with `args` (without the program name) and the moves `moves`, a dict of
/// names and callables (`-cs_opt method=NAME` chooses one; see [`PyMove`] for the callable):
/// the search minimizes what the callable returns, judges it, and goes on as with the
/// built-in moves. Returns the directory of the run, the number of conformers, and their
/// energies (Hartree, ascending). The GIL is released while the command runs and taken for
/// every call of a move.
#[pyfunction]
#[pyo3(signature = (args, moves = None))]
fn run_confsearch(
    py: Python<'_>,
    args: Vec<String>,
    moves: Option<Bound<'_, PyDict>>,
) -> PyResult<(String, usize, Vec<f64>)> {
    let mut registry = MoveRegistry::builtin();
    if let Some(moves) = moves {
        for (name, callable) in moves.iter() {
            let name: String = name.extract()?;
            if !callable.is_callable() {
                return Err(PyRuntimeError::new_err(format!(
                    "the conformer move '{name}' is not callable"
                )));
            }
            let callback: Py<PyAny> = callable.unbind();
            let label = name.clone();
            registry
                .register(
                    &name,
                    std::sync::Arc::new(move |options| {
                        // A Python move has no settings of the command line: it is an object
                        // of the caller, who sets it up.
                        options.only(&label, &[])?;
                        Ok(Box::new(PyMove {
                            name: label.clone(),
                            callback: Python::with_gil(|py| callback.clone_ref(py)),
                        }) as Box<dyn ConformerMove>)
                    }),
                )
                .map_err(err)?;
        }
    }
    let texts: Vec<&str> = args.iter().map(String::as_str).collect();
    let config = mopt_rs::cli::confsearch::config_from_cli(&texts).map_err(err)?;
    let calculators = super::pycalc::python_registry();
    let out = py
        .allow_threads(|| {
            mopt_rs::cli::confsearch::run_with_moves(
                &config,
                std::path::Path::new("."),
                Some(mopt_rs::core_infra::log::Level::Info),
                &calculators,
                &registry,
            )
        })
        .map_err(err)?;
    Ok((
        out.directory.to_string_lossy().into_owned(),
        out.count,
        out.energies,
    ))
}

/// The names of the built-in moves of `confsearch` (`-cs_opt method=`).
#[pyfunction]
fn conformer_move_names() -> Vec<String> {
    MoveRegistry::builtin().names()
}

/// Register the functions and classes of this module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_confsearch, m)?)?;
    m.add_function(wrap_pyfunction!(conformer_move_names, m)?)?;
    m.add_class::<PyConformerEnsemble>()?;
    m.add_class::<PyRmsdMetadynamics>()?;
    m.add_function(wrap_pyfunction!(bond_graph, m)?)?;
    m.add_function(wrap_pyfunction!(graph_isomorphism, m)?)?;
    m.add_function(wrap_pyfunction!(invariant_rmsd, m)?)?;
    m.add_function(wrap_pyfunction!(zmatrix, m)?)?;
    m.add_function(wrap_pyfunction!(crossover_conformers, m)?)?;
    Ok(())
}
