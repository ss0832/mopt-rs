//! Rust components callable one by one from Python: units, elements, native calculators,
//! model Hessians, D3/D4 dispersion, and constraint functions.
//!
//! Matrices cross the boundary as flat row-major lists in atomic units.
//!
//! MultiOptPy: no counterpart (new in mopt-rs); MultiOptPy's parts cannot be called one by one.
//!
//! This file is part of mopt-py (the Python interface of mopt-rs).
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use mopt_rs::cli::constraints::user_constraint;
use mopt_rs::cli::optmain::NATIVE_CALCULATORS;
use mopt_rs::domain::{Element, EvalOutput, Units};
use mopt_rs::model_hessian::{base_names, resolve as resolve_model, ModelSpec};
use mopt_rs::pes::constraint::ConstraintSet;
use mopt_rs::pes::dispersion::{D3Bj, D4};

use super::{cell_from, err, system};

fn flat(m: &Mat<f64>) -> Vec<f64> {
    let (r, c) = (m.nrows(), m.ncols());
    (0..r * c).map(|k| m[(k / c, k % c)]).collect()
}

fn triple(out: EvalOutput) -> (f64, Vec<f64>, Option<Vec<f64>>) {
    (
        out.energy.value(),
        out.gradient_bohr.iter().copied().collect(),
        out.hessian_bohr.as_ref().map(flat),
    )
}

/// Unit conversion factors used everywhere in mopt-rs (CODATA 2018, `mopt_rs::domain::Units`).
#[pyfunction]
pub fn units(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("bohr_to_angstrom", Units::BOHR_TO_ANGSTROM)?;
    d.set_item("angstrom_to_bohr", Units::ANGSTROM_TO_BOHR)?;
    d.set_item("hartree_to_ev", Units::HARTREE_TO_EV)?;
    d.set_item("hartree_to_kcalmol", Units::HARTREE_TO_KCAL_MOL)?;
    d.set_item("hartree_to_kjmol", Units::HARTREE_TO_KJ_MOL)?;
    Ok(d)
}

/// Atomic numbers of element symbols (`"Fe"`, `"fe"`, `"26"`).
#[pyfunction]
pub fn atomic_numbers(symbols: Vec<String>) -> PyResult<Vec<u32>> {
    symbols
        .iter()
        .map(|s| Element::parse_token(s).map(|e| u32::from(e.atomic_number())))
        .collect::<Result<_, _>>()
        .map_err(err)
}

/// Names of the potentials implemented in Rust.
#[pyfunction]
pub fn native_calculators() -> Vec<&'static str> {
    NATIVE_CALCULATORS.to_vec()
}

/// Model Hessian names (bases; prefixes `legacy_` and modifiers as in `-modelhess`).
#[pyfunction]
pub fn model_hessian_names() -> Vec<&'static str> {
    base_names().to_vec()
}

/// A model Hessian (flat 3N x 3N, Hartree/Bohr^2). `name` as for `-modelhess`
/// (`fischerd3`, `lindh2007d4`, `schlegel`, ...); `gradient` is used by modifiers that need it.
#[pyfunction]
#[pyo3(signature = (name, symbols, positions_bohr, gradient = None, charge = 0, multiplicity = 1, cell_bohr = None))]
#[allow(clippy::too_many_arguments)]
pub fn model_hessian(
    py: Python<'_>,
    name: &str,
    symbols: Vec<String>,
    positions_bohr: Vec<f64>,
    gradient: Option<Vec<f64>>,
    charge: i64,
    multiplicity: i64,
    cell_bohr: Option<Vec<f64>>,
) -> PyResult<Vec<f64>> {
    let sys = system(
        &symbols,
        &positions_bohr,
        charge,
        multiplicity,
        cell_from(cell_bohr)?,
    )?;
    ModelSpec::parse(name).map_err(err)?;
    let model = resolve_model(name).map_err(err)?;
    let g = gradient.unwrap_or_else(|| vec![0.0; positions_bohr.len()]);
    if g.len() != positions_bohr.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "gradient and positions differ in length",
        ));
    }
    let g = Col::from_fn(g.len(), |i| g[i]);
    let h = py.allow_threads(|| model.hessian(&sys, &g)).map_err(err)?;
    Ok(flat(&h))
}

/// Genuine D3(BJ) or D4 dispersion (PBE parameters): `(energy, gradient, hessian or None)`.
#[pyfunction]
#[pyo3(signature = (model, symbols, positions_bohr, charge = 0, need_hessian = false, cell_bohr = None))]
pub fn dispersion(
    py: Python<'_>,
    model: &str,
    symbols: Vec<String>,
    positions_bohr: Vec<f64>,
    charge: i64,
    need_hessian: bool,
    cell_bohr: Option<Vec<f64>>,
) -> PyResult<(f64, Vec<f64>, Option<Vec<f64>>)> {
    let sys = system(&symbols, &positions_bohr, charge, 1, cell_from(cell_bohr)?)?;
    let out = match model.to_ascii_lowercase().as_str() {
        "d3" | "d3bj" => py.allow_threads(|| D3Bj::PBE.evaluate(&sys, need_hessian)),
        "d4" => py.allow_threads(|| D4::PBE.evaluate(&sys, need_hessian)),
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "dispersion model must be d3 or d4, got '{other}'"
            )))
        }
    }
    .map_err(err)?;
    Ok(triple(out))
}

/// D3 coordination numbers and D4 EEQ partial charges.
#[pyfunction]
#[pyo3(signature = (symbols, positions_bohr, charge = 0, cell_bohr = None))]
pub fn dispersion_properties(
    py: Python<'_>,
    symbols: Vec<String>,
    positions_bohr: Vec<f64>,
    charge: i64,
    cell_bohr: Option<Vec<f64>>,
) -> PyResult<Bound<'_, PyDict>> {
    let sys = system(&symbols, &positions_bohr, charge, 1, cell_from(cell_bohr)?)?;
    let d = PyDict::new(py);
    d.set_item(
        "d3_coordination_numbers",
        D3Bj::PBE.coordination_numbers(&sys).map_err(err)?,
    )?;
    if sys.active_cell().is_none() {
        d.set_item("d4_charges", D4::PBE.charges(&sys).map_err(err)?)?;
    }
    Ok(d)
}

/// The definitions of the component tools of the MCP servers that compute from structures,
/// and their structure arguments, as JSON text (`mopt_rs::mcp::compute`).
#[pyfunction]
pub fn mcp_component_tools() -> String {
    mopt_rs::mcp::compute::tools_json()
}

/// Compute the MCP component tool `name` with the arguments `arguments` (JSON text) and the
/// texts of its structures `structures` (JSON text: `{property: xyz text}`, read by the
/// server through its sandbox); the result as JSON text, or `ValueError` with the message.
#[pyfunction]
pub fn mcp_component(
    py: Python<'_>,
    name: &str,
    arguments: &str,
    structures: &str,
) -> PyResult<String> {
    let invalid = |e: String| pyo3::exceptions::PyValueError::new_err(e);
    let args = mopt_rs::core_infra::json::JsonValue::parse(arguments)
        .map_err(|e| invalid(format!("arguments: {e}")))?;
    let texts = mopt_rs::core_infra::json::JsonValue::parse(structures)
        .map_err(|e| invalid(format!("structures: {e}")))?;
    let texts: std::collections::BTreeMap<String, String> = texts
        .as_object()
        .ok_or_else(|| invalid("structures is an object".to_string()))?
        .iter()
        .filter_map(|(k, v)| Some((k.clone(), v.as_str()?.to_string())))
        .collect();
    let name = name.to_string();
    py.allow_threads(|| mopt_rs::mcp::compute::compute(&name, &args, &texts))
        .map(|v| v.to_string_compact())
        .map_err(invalid)
}

/// Values, residuals, and Jacobian (flat m x 3N) of user constraints (`-uc` expressions).
#[pyfunction]
#[pyo3(signature = (expressions, symbols, positions_bohr, cell_bohr = None))]
pub fn constraint_values(
    py: Python<'_>,
    expressions: Vec<String>,
    symbols: Vec<String>,
    positions_bohr: Vec<f64>,
    cell_bohr: Option<Vec<f64>>,
) -> PyResult<Bound<'_, PyDict>> {
    let sys = system(&symbols, &positions_bohr, 0, 1, cell_from(cell_bohr)?)?;
    let constraints = expressions
        .iter()
        .map(|e| user_constraint(e))
        .collect::<Result<Vec<_>, _>>()
        .map_err(err)?;
    let set = ConstraintSet::new(&constraints, &sys, None).map_err(err)?;
    let r = set.residuals(&sys).map_err(err)?;
    let zero = Col::zeros(sys.dof());
    let frame = set.frame(&sys, &zero).map_err(err)?;
    let d = PyDict::new(py);
    d.set_item("residuals", r.iter().copied().collect::<Vec<f64>>())?;
    d.set_item("jacobian", flat(&frame.jacobian))?;
    Ok(d)
}

/// Register the functions of this module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(units, m)?)?;
    m.add_function(wrap_pyfunction!(atomic_numbers, m)?)?;
    m.add_function(wrap_pyfunction!(native_calculators, m)?)?;
    m.add_function(wrap_pyfunction!(model_hessian_names, m)?)?;
    m.add_function(wrap_pyfunction!(mcp_component_tools, m)?)?;
    m.add_function(wrap_pyfunction!(mcp_component, m)?)?;
    m.add_function(wrap_pyfunction!(model_hessian, m)?)?;
    m.add_function(wrap_pyfunction!(dispersion, m)?)?;
    m.add_function(wrap_pyfunction!(dispersion_properties, m)?)?;
    m.add_function(wrap_pyfunction!(constraint_values, m)?)?;
    Ok(())
}
