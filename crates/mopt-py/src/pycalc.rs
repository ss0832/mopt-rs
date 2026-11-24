//! Python engines as Rust calculators, so that every command run from the Python package
//! (`optmain`, `mopt-py optmain`) can use PySCF, Psi4, tblite, dxtb, and any ASE calculator.
//!
//! The registry of [`python_registry`] has the Rust potentials plus a fallback that asks
//! `mopt_rs.calculators` for any other name; evaluations take the GIL for the duration of the
//! Python call only.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-py (the Python interface of mopt-rs).
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Mat;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use mopt_rs::domain::{EvalOutput, EvalRequest, MolecularSystem, OrbitalSet, Orbitals};
use mopt_rs::error::{MoptError, MoptResult};
use mopt_rs::pes::registry::{CalculatorOptions, CalculatorRegistry, CalculatorRequest};
use mopt_rs::pes::states::StateCharacters;
use mopt_rs::pes::Calculator;

use super::{output, with_spin_square};

/// A Python engine evaluated through the state-following memory of the Rust calculator.
#[pyclass(name = "FollowedEvaluator")]
pub struct PyFollowedEvaluator {
    inner: mopt_rs::pes::followed::FollowedStates,
}

#[pymethods]
impl PyFollowedEvaluator {
    #[new]
    fn new(py: Python<'_>, calculator: Py<PyAny>) -> PyResult<Self> {
        let calc = PyEngineCalculator::from_instance(py, calculator)?;
        if !calc.supports_excited_states() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "following requires an excited-state engine",
            ));
        }
        Ok(Self {
            inner: mopt_rs::pes::followed::FollowedStates::new(Box::new(calc)),
        })
    }

    #[pyo3(signature=(symbols, positions_bohr, charge, multiplicity, excited_state, need_hessian=false, cell_bohr=None, pbc=None))]
    #[allow(clippy::too_many_arguments)]
    fn evaluate(
        &self,
        py: Python<'_>,
        symbols: Vec<String>,
        positions_bohr: Vec<f64>,
        charge: i64,
        multiplicity: i64,
        excited_state: usize,
        need_hessian: bool,
        cell_bohr: Option<Vec<f64>>,
        pbc: Option<[bool; 3]>,
    ) -> PyResult<super::NativeResult> {
        let sys = super::system(
            &symbols,
            &positions_bohr,
            charge,
            multiplicity,
            super::cell_with_axes(cell_bohr, pbc)?,
        )?;
        super::evaluate_flat(py, &self.inner, &sys, need_hessian, excited_state)
    }

    fn take_warnings(&self) -> Vec<String> {
        self.inner.take_warnings()
    }
}

const BRIDGE: &str = "mopt_rs.calculators._bridge";

/// A `mopt_rs.calculators.EngineCalculator` behind the [`Calculator`] trait.
pub struct PyEngineCalculator {
    name: String,
    description: String,
    /// The papers to cite, as the engine names them.
    citations: Vec<String>,
    calc: Py<PyAny>,
    pbc: bool,
    excited_states: bool,
    electrons: bool,
    rigid_body_invariant: bool,
    /// Whether the engine tests its SCF solutions (`-stability`).
    stability: bool,
    point_charges: bool,
    /// What the calculator said about its evaluations (the tests of the SCF solution),
    /// until the command takes them as warnings.
    notes: std::sync::Mutex<Vec<String>>,
}

fn py_err(context: &str) -> impl Fn(PyErr) -> MoptError + '_ {
    move |e| MoptError::calc(format!("{context}: {e}"))
}

fn options_dict<'py>(py: Python<'py>, o: &CalculatorOptions) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("method", &o.method)?;
    d.set_item("basis", &o.basis)?;
    d.set_item("sub_basis", o.sub_basis.clone())?;
    d.set_item("ecp", o.ecp.clone())?;
    d.set_item("unrestricted", o.unrestricted)?;
    d.set_item("grid_level", o.grid_level)?;
    d.set_item("threads", o.threads)?;
    d.set_item("memory", &o.memory)?;
    d.set_item("solvation", o.solvation.clone())?;
    d.set_item("software_paths", o.software_paths.clone())?;
    d.set_item("software_path_file", &o.software_path_file)?;
    d.set_item("extra", o.extra.clone())?;
    d.set_item("stability", o.stability)?;
    d.set_item("parallel", o.structures_at_once())?;
    if let Some(model) = &o.model_of {
        d.set_item("model_atoms", model.atoms.clone())?;
        d.set_item("model_links", model.links.clone())?;
        d.set_item(
            "model_link_placement",
            if model.at_distance {
                "distance"
            } else {
                "scaled"
            },
        )?;
    }
    d.set_item(
        "stability_external",
        if o.stability_external.is_empty() {
            "report"
        } else {
            o.stability_external.as_str()
        },
    )?;
    Ok(d)
}

impl PyEngineCalculator {
    /// The Python calculator for `req`.
    pub fn create(req: &CalculatorRequest<'_>) -> MoptResult<Self> {
        let context = format!("calculator '{}'", req.name);
        let fail = py_err(&context);
        Python::with_gil(|py| {
            let make = py
                .import(BRIDGE)
                .map_err(&fail)?
                .getattr("make")
                .map_err(&fail)?;
            let (calc, info): (Py<PyAny>, Bound<'_, PyDict>) = make
                .call1((req.name, options_dict(py, req.options).map_err(&fail)?))
                .map_err(&fail)?
                .extract()
                .map_err(&fail)?;
            Self::from_parts(req.name, calc, &info)
        })
    }

    pub(super) fn from_instance(py: Python<'_>, calc: Py<PyAny>) -> PyResult<Self> {
        let info = py
            .import(BRIDGE)?
            .getattr("calculator_info")?
            .call1((calc.bind(py),))?;
        Self::from_parts("Python calculator", calc, info.downcast::<PyDict>()?).map_err(super::err)
    }

    fn from_parts(fallback: &str, calc: Py<PyAny>, info: &Bound<'_, PyDict>) -> MoptResult<Self> {
        let fail = py_err(fallback);
        let get_or = |k: &str, default: bool| -> MoptResult<bool> {
            info.get_item(k)
                .map_err(&fail)?
                .map_or(Ok(default), |v| v.extract::<bool>().map_err(&fail))
        };
        let get = |k: &str| get_or(k, false);
        let name = info
            .get_item("name")
            .map_err(&fail)?
            .map(|v| v.extract::<String>())
            .transpose()
            .map_err(&fail)?
            .unwrap_or_else(|| fallback.to_string());
        let description = info
            .get_item("description")
            .map_err(&fail)?
            .map(|v| v.extract::<String>())
            .transpose()
            .map_err(&fail)?
            .unwrap_or_else(|| name.clone());
        let citations = info
            .get_item("citations")
            .map_err(&fail)?
            .map(|v| v.extract::<Vec<String>>())
            .transpose()
            .map_err(&fail)?
            .unwrap_or_default();
        Ok(Self {
            name,
            description,
            citations,
            calc,
            pbc: get("pbc")?,
            excited_states: get("excited_states")?,
            electrons: get("electrons")?,
            rigid_body_invariant: get_or("rigid_body_invariant", true)?,
            stability: get("stability")?,
            point_charges: get("point_charges")?,
            notes: std::sync::Mutex::new(Vec::new()),
        })
    }
}

impl Calculator for PyEngineCalculator {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> String {
        self.description.clone()
    }

    fn citations(&self) -> Vec<String> {
        self.citations.clone()
    }

    fn evaluate(&self, sys: &MolecularSystem, req: &EvalRequest) -> MoptResult<EvalOutput> {
        let fail = py_err(&self.name);
        let symbols: Vec<&str> = sys.elements.iter().map(|e| e.symbol()).collect();
        let positions: Vec<f64> = sys.coords_flat_bohr().iter().copied().collect();
        let cell: Option<Vec<f64>> = sys
            .cell
            .as_ref()
            .map(|c| (0..3).flat_map(|k| c.lattice_vector(k).value()).collect());
        let pbc = sys.cell.as_ref().map_or([false; 3], |c| c.periodic_axes);
        type Evaluated = (
            f64,
            Vec<f64>,
            Option<Vec<f64>>,
            Option<f64>,
            Option<String>,
            Option<Vec<f64>>,
            Vec<String>,
        );
        let (e, g, h, spin_square, printed, stress, notes): Evaluated = Python::with_gil(|py| {
            let evaluate = py.import(BRIDGE)?.getattr("evaluate")?;
            evaluate
                .call1((
                    self.calc.clone_ref(py),
                    symbols,
                    positions,
                    sys.spin_charge.charge,
                    sys.spin_charge.multiplicity,
                    cell,
                    pbc.to_vec(),
                    req.need_hessian,
                    req.excited_state,
                    req.need_stress,
                    req.need_gradient,
                ))?
                .extract()
        })
        .map_err(fail)?;
        if !notes.is_empty() {
            if let Ok(mut kept) = self.notes.lock() {
                kept.extend(notes);
            }
        }
        if stress.as_ref().is_some_and(|s| s.len() != 9) {
            return Err(MoptError::calc(format!(
                "{}: a stress that is not 3 x 3",
                self.name
            )));
        }
        output(e, g, h)
            .map(|out| with_spin_square(out, spin_square))
            .map(|mut out| {
                // Hartree/Bohr^3, row by row, with the sign of ASE.
                out.stress =
                    stress.map(|s| std::array::from_fn(|a| std::array::from_fn(|b| s[3 * a + b])));
                if let Some(text) = printed.filter(|t| !t.trim().is_empty()) {
                    out.orbitals
                        .get_or_insert_with(Default::default)
                        .program_output = Some(text);
                }
                out
            })
            .map_err(|e| MoptError::calc(format!("{}: {e}", self.name)))
    }

    fn supports_point_charges(&self) -> bool {
        self.point_charges
    }

    fn evaluate_in_charges(
        &self,
        sys: &MolecularSystem,
        req: &EvalRequest,
        charges: &mopt_rs::pes::point_charges::PointCharges,
    ) -> MoptResult<mopt_rs::pes::point_charges::EmbeddedOutput> {
        let symbols: Vec<_> = sys.elements.iter().map(|e| e.symbol()).collect();
        let positions: Vec<_> = sys.coords_flat_bohr().iter().copied().collect();
        let elements: Vec<_> = charges
            .elements
            .iter()
            .map(|e| e.map(|x| x.symbol()))
            .collect();
        Python::with_gil(|py| {
            let (e, g, h, q): (f64, Vec<f64>, Option<Vec<f64>>, Vec<[f64; 3]>) = py
                .import(BRIDGE)?
                .getattr("evaluate_embedded")?
                .call1((
                    self.calc.bind(py),
                    symbols,
                    positions,
                    sys.spin_charge.charge,
                    sys.spin_charge.multiplicity,
                    req.excited_state,
                    req.need_hessian,
                    &charges.positions_bohr,
                    &charges.charges,
                    elements,
                ))?
                .extract()?;
            if g.len() != sys.dof() || q.len() != charges.len() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "invalid embedded gradient dimensions",
                ));
            }
            Ok(mopt_rs::pes::point_charges::EmbeddedOutput {
                output: output(e, g, h)?,
                charge_gradient_bohr: q,
            })
        })
        .map_err(py_err(&self.name))
    }

    fn orbitals(&self, sys: &MolecularSystem) -> MoptResult<Option<Orbitals>> {
        let fail = py_err(&self.name);
        let symbols: Vec<&str> = sys.elements.iter().map(|e| e.symbol()).collect();
        let positions: Vec<f64> = sys.coords_flat_bohr().iter().copied().collect();
        let cell: Option<Vec<f64>> = sys
            .cell
            .as_ref()
            .map(|c| (0..3).flat_map(|k| c.lattice_vector(k).value()).collect());
        let pbc = sys.cell.as_ref().map_or([false; 3], |c| c.periodic_axes);
        // Coefficients (row by row), number of orbitals, energies, occupations.
        type Set = (Vec<f64>, usize, Vec<f64>, Vec<f64>);
        type Found = Option<(Vec<f64>, usize, Vec<Set>, Option<Vec<usize>>)>;
        let found: Found = Python::with_gil(|py| {
            let orbitals = py.import(BRIDGE)?.getattr("orbitals")?;
            orbitals
                .call1((
                    self.calc.clone_ref(py),
                    symbols,
                    positions,
                    sys.spin_charge.charge,
                    sys.spin_charge.multiplicity,
                    cell,
                    pbc.to_vec(),
                ))?
                .extract()
        })
        .map_err(fail)?;
        let Some((overlap, n, sets, atom_of_ao)) = found else {
            return Ok(None);
        };
        let wrong = || MoptError::calc(format!("{}: orbitals of the wrong size", self.name));
        if overlap.len() != n * n || sets.is_empty() || sets.len() > 2 {
            return Err(wrong());
        }
        let overlap = Mat::from_fn(n, n, |i, j| overlap[i * n + j]);
        let mut built = Vec::with_capacity(sets.len());
        for (coefficients, m, energies, occupations) in sets {
            if coefficients.len() != n * m {
                return Err(wrong());
            }
            let c = Mat::from_fn(n, m, |i, k| coefficients[i * m + k]);
            built.push(OrbitalSet::new(c, energies, occupations)?);
        }
        let beta = if built.len() == 2 { built.pop() } else { None };
        let alpha = built.pop().ok_or_else(wrong)?;
        let orbitals = Orbitals::new(overlap, alpha, beta)?;
        match atom_of_ao {
            Some(atoms) => orbitals.with_atoms(atoms).map(Some),
            None => Ok(Some(orbitals)),
        }
    }

    fn state_characters(
        &self,
        sys: &MolecularSystem,
        count: usize,
    ) -> MoptResult<Option<StateCharacters>> {
        let fail = py_err(&self.name);
        let symbols: Vec<&str> = sys.elements.iter().map(|e| e.symbol()).collect();
        let positions: Vec<f64> = sys.coords_flat_bohr().iter().copied().collect();
        let cell: Option<Vec<f64>> = sys
            .cell
            .as_ref()
            .map(|c| (0..3).flat_map(|k| c.lattice_vector(k).value()).collect());
        let pbc = sys.cell.as_ref().map_or([false; 3], |c| c.periodic_axes);
        // Orbitals (row by row), their number, the occupied ones, energies, amplitudes.
        type Found = Option<(Vec<f64>, usize, usize, Vec<f64>, Vec<f64>)>;
        let found: Found = Python::with_gil(|py| {
            let states = py.import(BRIDGE)?.getattr("state_characters")?;
            states
                .call1((
                    self.calc.clone_ref(py),
                    symbols,
                    positions,
                    sys.spin_charge.charge,
                    sys.spin_charge.multiplicity,
                    cell,
                    pbc.to_vec(),
                    count,
                ))?
                .extract()
        })
        .map_err(fail)?;
        let Some((orbitals, n, occupied, energies, amplitudes)) = found else {
            return Ok(None);
        };
        let states = energies.len();
        if orbitals.len() != n * n
            || occupied > n
            || amplitudes.len() != occupied * (n - occupied) * states
        {
            return Err(MoptError::calc(format!(
                "{}: excited states of the wrong size",
                self.name
            )));
        }
        StateCharacters::new(
            Mat::from_fn(n, n, |i, j| orbitals[i * n + j]),
            occupied,
            energies,
            Mat::from_fn(occupied * (n - occupied), states, |p, k| {
                amplitudes[p * states + k]
            }),
        )
        .map(Some)
    }

    fn transitions(
        &self,
        sys: &MolecularSystem,
        count: usize,
    ) -> MoptResult<Option<Vec<mopt_rs::pes::states::Transition>>> {
        let fail = py_err(&self.name);
        let symbols: Vec<&str> = sys.elements.iter().map(|e| e.symbol()).collect();
        let positions: Vec<f64> = sys.coords_flat_bohr().iter().copied().collect();
        let cell: Option<Vec<f64>> = sys
            .cell
            .as_ref()
            .map(|c| (0..3).flat_map(|k| c.lattice_vector(k).value()).collect());
        let pbc = sys.cell.as_ref().map_or([false; 3], |c| c.periodic_axes);
        // Excitation, transition dipole, population changes, and (when the engine gives them)
        // the change of the dipole and the transition populations.
        type One = (f64, Vec<f64>, Vec<f64>, Option<Vec<f64>>, Option<Vec<f64>>);
        type Found = Option<Vec<One>>;
        let found: Found = Python::with_gil(|py| {
            let transitions = py.import(BRIDGE)?.getattr("transitions")?;
            transitions
                .call1((
                    self.calc.clone_ref(py),
                    symbols,
                    positions,
                    sys.spin_charge.charge,
                    sys.spin_charge.multiplicity,
                    cell,
                    pbc.to_vec(),
                    count,
                ))?
                .extract()
        })
        .map_err(fail)?;
        let Some(found) = found else {
            return Ok(None);
        };
        let n = sys.n_atoms();
        found
            .into_iter()
            .enumerate()
            .map(
                |(k, (excitation, dipole, change, dipole_change, transition))| {
                    let wrong_extra = dipole_change.as_ref().is_some_and(|d| d.len() != 3)
                        || transition.as_ref().is_some_and(|t| t.len() != n);
                    if dipole.len() != 3 || change.len() != n || wrong_extra {
                        return Err(MoptError::calc(format!(
                            "{}: transition {} has {} dipole components and {} population changes \
                         for {n} atoms",
                            self.name,
                            k + 1,
                            dipole.len(),
                            change.len()
                        )));
                    }
                    Ok(mopt_rs::pes::states::Transition {
                        excitation,
                        transition_dipole: [dipole[0], dipole[1], dipole[2]],
                        population_change: change,
                        dipole_change: dipole_change.map(|d| [d[0], d[1], d[2]]),
                        transition_population: transition,
                    })
                },
            )
            .collect::<MoptResult<Vec<_>>>()
            .map(Some)
    }

    fn supports_pbc(&self) -> bool {
        self.pbc
    }

    fn supports_excited_states(&self) -> bool {
        self.excited_states
    }

    fn uses_electrons(&self) -> bool {
        self.electrons
    }

    fn is_rigid_body_invariant(&self) -> bool {
        self.rigid_body_invariant
    }

    fn checks_stability(&self) -> bool {
        self.stability
    }

    /// Asked of the engine every time: an ASE engine whose Hessian method fails goes over to
    /// differences in the middle of a run. A failing question answers `false`.
    fn analytic_hessian(&self) -> bool {
        Python::with_gil(|py| {
            py.import(BRIDGE)
                .and_then(|m| m.getattr("analytic_hessian"))
                .and_then(|f| f.call1((self.calc.clone_ref(py),)))
                .and_then(|v| v.extract::<bool>())
                .unwrap_or(false)
        })
    }

    fn take_warnings(&self) -> Vec<String> {
        self.notes
            .lock()
            .map(|mut kept| std::mem::take(&mut *kept))
            .unwrap_or_default()
    }
}

/// The Rust potentials plus every engine of `mopt_rs.calculators`.
pub fn python_registry() -> CalculatorRegistry {
    let mut r = CalculatorRegistry::native();
    r.set_fallback(|q| Ok(Box::new(PyEngineCalculator::create(q)?)));
    r
}
