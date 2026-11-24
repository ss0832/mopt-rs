//! The ONIOM calculator from Python: the layers, link atoms, embedding, charges and
//! multiplicities of the model systems, and the settings of the microiterations, given as
//! keyword arguments and turned into the options `-os`, `-oniom`, `-oniom_middle`,
//! `-oniom_opt`, and `-calc_opt` of the commands, which the one parser of the commands
//! (`cli::oniom::oniom_spec`, through `cli::calculator::calculator_options`) reads. The
//! surface is then made as the commands make it (`cli::oniom_micro::wrap_constrained`):
//! [`crate::PyOptimizer`] with `calculator=` an ONIOM calculator moves the core with the
//! environment relaxed at every evaluation, as `optmain -oniom` does.
//!
//! MultiOptPy: `interface.py` (`-oniom`), `ONIOM/` (the calculation of the layers).
//!
//! This file is part of mopt-py (the Python interface of mopt-rs).
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::sync::Mutex;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use mopt_rs::cli::calculator::{calculator_name, calculator_options};
use mopt_rs::cli::spec::OPTMAIN;
use mopt_rs::domain::Element;
use mopt_rs::pes::registry::CalculatorOptions;
use mopt_rs::pes::Calculator;

use super::{cell_with_axes, err, evaluate_flat, multiplicity_of, system, NativeResult};

/// Atoms as the commands take them: a text such as `"1-6,9"` or numbers from 1.
fn atom_text(atoms: &Bound<'_, PyAny>, what: &str) -> PyResult<String> {
    if let Ok(text) = atoms.extract::<String>() {
        return Ok(text);
    }
    let numbers: Vec<usize> = atoms.extract().map_err(|_| {
        PyValueError::new_err(format!(
            "{what}: atoms from 1 as a list of numbers or a text such as '1-6,9'"
        ))
    })?;
    if numbers.is_empty() {
        return Err(PyValueError::new_err(format!("{what}: no atoms")));
    }
    Ok(numbers
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(","))
}

/// The link atoms of a layer: `"auto"` (also when not given), `"none"`, or atoms.
fn link_text(links: Option<&Bound<'_, PyAny>>, what: &str) -> PyResult<String> {
    match links {
        None => Ok("auto".to_string()),
        Some(links) => atom_text(links, what),
    }
}

/// `KEY=VALUE` texts of a mapping or of a sequence of pairs, the values as the command line
/// writes them.
fn settings_text(settings: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<String>> {
    let Some(settings) = settings else {
        return Ok(Vec::new());
    };
    let pairs: Vec<(String, Bound<'_, PyAny>)> = match settings.downcast::<PyDict>() {
        Ok(d) => d
            .iter()
            .map(|(k, v)| Ok((k.str()?.to_string(), v)))
            .collect::<PyResult<_>>()?,
        Err(_) => settings.extract()?,
    };
    pairs
        .iter()
        .map(|(k, v)| setting_text(v).map(|v| format!("{k}={v}")))
        .collect()
}

/// The value of a setting as the command line writes it (`True` and `False` as `on`, `off`).
fn setting_text(value: &Bound<'_, PyAny>) -> PyResult<String> {
    if let Ok(flag) = value.extract::<bool>() {
        return Ok(if flag { "on" } else { "off" }.to_string());
    }
    Ok(value.str()?.to_string())
}

/// An ONIOM calculator: the options of the commands it stands for (`options`), the name of
/// the calculator of the highest level, and its settings as the parser of the commands made
/// them. Give it to `Optimizer(..., calculator=...)` (the optimizer moves the core, the
/// environment is relaxed at every evaluation), to `relax_band` (as the calculator of the
/// chain), or its `options` to `run_optmain` and `run_cli`.
#[pyclass(name = "OniomCalculator")]
pub struct PyOniomCalculator {
    pub(crate) name: String,
    pub(crate) options: CalculatorOptions,
    tokens: Vec<String>,
    /// The ONIOM surface of whole structures (`micro=off` only), made at the first
    /// evaluation for the elements of that structure.
    whole: Mutex<Option<(Vec<Element>, Box<dyn Calculator>)>>,
}

#[pymethods]
impl PyOniomCalculator {
    /// `high_atoms` (the model system, `-oniom`; a text such as `"1-6"` or atoms from 1), `low`
    /// (the calculator of the lowest level),
    /// `high_calculator` (that of the highest level, `-os`), `links` (`"auto"`, `"none"`, or
    /// the atoms whose bonds are cut), the middle layer of a three-layer model
    /// (`middle_atoms`, `middle_links`, `middle_calculator`, `-oniom_middle`), the placement
    /// of the link atoms (`link_placement`: `"scaled"`, the g-factor of the bond, or
    /// `"distance"`, a fixed distance), the embedding (`"mechanical"` or `"electronic"`), the
    /// charge and multiplicity of the model systems, the settings of `-oniom_opt` (`settings`,
    /// a mapping such as `{"micro_tolerance": 1e-5, "micro_replicas": 4,
    /// "micro_hessian_solve": "cg", "low.cutoff": 6.0}`), and the settings of the calculator
    /// of the highest level (`calculator_settings`, `-calc_opt`).
    #[new]
    #[pyo3(signature = (
        high_atoms,
        low,
        *,
        high_calculator,
        links = None,
        middle_atoms = None,
        middle_links = None,
        middle_calculator = None,
        link_placement = None,
        embedding = None,
        high_charge = None,
        high_multiplicity = None,
        middle_charge = None,
        middle_multiplicity = None,
        settings = None,
        calculator_settings = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        high_atoms: Bound<'_, PyAny>,
        low: &str,
        high_calculator: &str,
        links: Option<Bound<'_, PyAny>>,
        middle_atoms: Option<Bound<'_, PyAny>>,
        middle_links: Option<Bound<'_, PyAny>>,
        middle_calculator: Option<String>,
        link_placement: Option<String>,
        embedding: Option<String>,
        high_charge: Option<i64>,
        high_multiplicity: Option<i64>,
        middle_charge: Option<i64>,
        middle_multiplicity: Option<i64>,
        settings: Option<Bound<'_, PyAny>>,
        calculator_settings: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let mut tokens = vec![
            "-os".to_string(),
            high_calculator.to_string(),
            "-oniom".to_string(),
            atom_text(&high_atoms, "high_atoms")?,
            link_text(links.as_ref(), "links")?,
            low.to_string(),
        ];
        match (&middle_atoms, &middle_calculator) {
            (Some(atoms), Some(calculator)) => tokens.extend([
                "-oniom_middle".to_string(),
                atom_text(atoms, "middle_atoms")?,
                link_text(middle_links.as_ref(), "middle_links")?,
                calculator.clone(),
            ]),
            (None, None) => {}
            _ => {
                return Err(PyValueError::new_err(
                    "a middle layer needs middle_atoms and middle_calculator (-oniom_middle \
                     ATOMS LINK CALCULATOR)",
                ))
            }
        }
        let mut opt: Vec<String> = Vec::new();
        if let Some(p) = link_placement {
            opt.push(format!("link={p}"));
        }
        if let Some(e) = embedding {
            opt.push(format!("embedding={e}"));
        }
        for (key, value) in [
            ("high_charge", high_charge),
            ("high_multiplicity", high_multiplicity),
            ("middle_charge", middle_charge),
            ("middle_multiplicity", middle_multiplicity),
        ] {
            if let Some(v) = value {
                opt.push(format!("{key}={v}"));
            }
        }
        opt.extend(settings_text(settings.as_ref())?);
        if !opt.is_empty() {
            tokens.push("-oniom_opt".to_string());
            tokens.extend(opt);
        }
        let calc = settings_text(calculator_settings.as_ref())?;
        if !calc.is_empty() {
            tokens.push("-calc_opt".to_string());
            tokens.extend(calc);
        }
        Self::from_tokens(tokens)
    }

    /// The options of the commands this calculator stands for (`-os ... -oniom ...`).
    #[getter]
    fn options(&self) -> Vec<String> {
        self.tokens.clone()
    }

    /// The calculator of the highest level.
    #[getter]
    fn high_calculator(&self) -> String {
        self.name.clone()
    }

    /// Whether the environment is relaxed at every evaluation (`micro=off` not given).
    #[getter]
    fn microiterations(&self) -> bool {
        self.options
            .oniom
            .as_ref()
            .is_some_and(|s| s.micro.is_some())
    }

    /// `(energy, gradient, hessian or None)` of the ONIOM surface of a whole structure, with
    /// every atom moving: only with `micro=off`, since with microiterations the surface is
    /// that of the core, which `Optimizer(..., calculator=...)` sets up.
    #[pyo3(signature = (symbols, positions_bohr, charge = 0, multiplicity = None, need_hessian = false, cell_bohr = None, unpaired_electrons = None, *, pbc = None, excited_state = 0))]
    #[allow(clippy::too_many_arguments)]
    fn evaluate(
        &self,
        py: Python<'_>,
        symbols: Vec<String>,
        positions_bohr: Vec<f64>,
        charge: i64,
        multiplicity: Option<i64>,
        need_hessian: bool,
        cell_bohr: Option<Vec<f64>>,
        unpaired_electrons: Option<i64>,
        pbc: Option<[bool; 3]>,
        excited_state: usize,
    ) -> PyResult<NativeResult> {
        let spec = self
            .options
            .oniom
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("no ONIOM settings"))?;
        if spec.micro.is_some() {
            return Err(PyValueError::new_err(
                "ONIOM with microiterations: the surface is that of the core with the \
                 environment relaxed; give this calculator to Optimizer(..., calculator=...) \
                 (or settings={'micro': False} to move every atom)",
            ));
        }
        let multiplicity = multiplicity_of(multiplicity, unpaired_electrons)?;
        let sys = system(
            &symbols,
            &positions_bohr,
            charge,
            multiplicity,
            cell_with_axes(cell_bohr, pbc)?,
        )?;
        let mut guard = self
            .whole
            .lock()
            .map_err(|_| PyValueError::new_err("the ONIOM calculator is poisoned"))?;
        if guard.as_ref().is_none_or(|(el, _)| *el != sys.elements) {
            let registry = crate::pycalc::python_registry();
            let calc = registry
                .create_oniom(&self.name, &self.options, &sys)
                .and_then(|w| w.with_every_atom_moving(spec))
                .map_err(err)?;
            *guard = Some((sys.elements.clone(), calc));
        }
        let (_, calc) = guard
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("the ONIOM calculator is missing"))?;
        evaluate_flat(py, calc.as_ref(), &sys, need_hessian, excited_state)
    }
}

impl PyOniomCalculator {
    /// The options of the commands this calculator stands for.
    pub(crate) fn options_tokens(&self) -> Vec<String> {
        self.tokens.clone()
    }

    /// The calculator of the options `tokens` of the commands, read by their parser.
    fn from_tokens(tokens: Vec<String>) -> PyResult<Self> {
        let refs: Vec<&str> = tokens.iter().map(String::as_str).collect();
        let parsed = mopt_rs::cli::option::parser("optmain", OPTMAIN)
            .parse(&refs)
            .map_err(err)?;
        let name = calculator_name(&parsed).map_err(err)?;
        let options = calculator_options(&parsed).map_err(err)?;
        if options.oniom.is_none() {
            return Err(PyValueError::new_err("no -oniom in the options"));
        }
        Ok(Self {
            name,
            options,
            tokens,
            whole: Mutex::new(None),
        })
    }
}

/// Add the class of this module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyOniomCalculator>()
}
