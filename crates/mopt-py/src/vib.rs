//! Vibrational analysis from Python: normal modes of a structure with its Hessian, its
//! point group, rotational constants, and thermochemistry.
//!
//! Positions are flat lists in Bohr, Hessians flat row-major lists in Hartree/Bohr^2.
//!
//! MultiOptPy: `Thermo/normal_mode_analyzer.py:MolecularVibrations`,
//! `Utils/symmetry_analyzer.py`.
//!
//! This file is part of mopt-py (the Python interface of mopt-rs).
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use mopt_rs::domain::MolecularSystem;
use mopt_rs::vib::{
    self, Direction, EnthalpyTreatment, FrequencySettings, NormalModes, Parts, QuasiHarmonic,
    RotorMoment, ThermoSettings,
};

use super::{cell_from, err, system};

fn square(flat: &[f64], n: usize) -> PyResult<Mat<f64>> {
    if flat.len() != n * n {
        return Err(PyValueError::new_err(format!(
            "the Hessian has {} entries, expected {n} x {n}",
            flat.len()
        )));
    }
    Ok(Mat::from_fn(n, n, |i, j| flat[i * n + j]))
}

fn directions(n: usize, gradient: Option<Vec<f64>>) -> PyResult<Vec<Direction>> {
    match gradient {
        None => Ok(Vec::new()),
        Some(g) if g.len() == n => Ok(vec![Direction::Gradient(Col::from_fn(n, |k| g[k]))]),
        Some(g) => Err(PyValueError::new_err(format!(
            "the gradient has {} components, expected {n}",
            g.len()
        ))),
    }
}

fn modes_dict<'py>(py: Python<'py>, modes: &NormalModes) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("wavenumbers", modes.wavenumbers.clone())?;
    d.set_item("curvatures", modes.curvatures.clone())?;
    d.set_item(
        "displacements",
        modes
            .displacements
            .iter()
            .map(|c| c.iter().copied().collect::<Vec<f64>>())
            .collect::<Vec<_>>(),
    )?;
    d.set_item("reduced_masses", modes.reduced_masses.clone())?;
    d.set_item("force_constants", modes.force_constants.clone())?;
    d.set_item("temperatures", modes.temperatures.clone())?;
    Ok(d)
}

fn parts<'py>(py: Python<'py>, p: &Parts) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("total", p.total)?;
    d.set_item("electronic", p.electronic)?;
    d.set_item("translational", p.translational)?;
    d.set_item("rotational", p.rotational)?;
    d.set_item("vibrational", p.vibrational)?;
    Ok(d)
}

/// Normal modes of a structure with its Cartesian Hessian: a dict with `wavenumbers`
/// (cm^-1, negative for imaginary frequencies), `curvatures` (eigenvalues of the
/// mass-weighted Hessian), `displacements` (one Cartesian vector per mode),
/// `reduced_masses` (amu), `force_constants` (mdyne/Angstrom), and `temperatures` (K), in
/// the order of the eigenvalues. With `project`, translations and rotations are removed
/// (translations only in a cell); `gradient` removes the direction of a gradient as well
/// (the tangent of a reaction path through the structure).
#[pyfunction]
#[pyo3(signature = (symbols, positions_bohr, hessian, project = true, cell_bohr = None, gradient = None))]
fn normal_modes<'py>(
    py: Python<'py>,
    symbols: Vec<String>,
    positions_bohr: Vec<f64>,
    hessian: Vec<f64>,
    project: bool,
    cell_bohr: Option<Vec<f64>>,
    gradient: Option<Vec<f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let sys = system(&symbols, &positions_bohr, 0, 1, cell_from(cell_bohr)?)?;
    let n = positions_bohr.len();
    let h = square(&hessian, n)?;
    let left_out = directions(n, gradient)?;
    let modes = py
        .allow_threads(|| vib::normal_modes_without(&sys, &h, project, &left_out))
        .map_err(err)?;
    modes_dict(py, &modes)
}

/// The point group of a structure: `(Schoenflies symbol, rotational symmetry number,
/// number of operations found, tolerance in Bohr at which they form a group)`.
#[pyfunction]
#[pyo3(signature = (symbols, positions_bohr, tolerance = 0.01))]
fn point_group(
    symbols: Vec<String>,
    positions_bohr: Vec<f64>,
    tolerance: f64,
) -> PyResult<(String, u32, usize, f64)> {
    let sys = system(&symbols, &positions_bohr, 0, 1, None)?;
    let group = vib::point_group(&sys, tolerance);
    Ok((
        group.label,
        group.symmetry_number,
        group.operations,
        group.tolerance,
    ))
}

/// Rotational constants `A >= B >= C` of a structure in GHz (infinite for an axis without
/// a moment of inertia).
#[pyfunction]
fn rotational_constants(symbols: Vec<String>, positions_bohr: Vec<f64>) -> PyResult<[f64; 3]> {
    let sys = system(&symbols, &positions_bohr, 0, 1, None)?;
    vib::rotational_constants_ghz(&sys).map_err(err)
}

fn settings(
    sys: &MolecularSystem,
    options: Option<&Bound<'_, PyDict>>,
) -> PyResult<FrequencySettings> {
    let mut s = FrequencySettings::default();
    let mut thermo = ThermoSettings {
        multiplicity: sys.spin_charge.multiplicity,
        ..ThermoSettings::default()
    };
    let Some(options) = options else {
        s.thermo = thermo;
        return Ok(s);
    };
    for (key, value) in options.iter() {
        let key: String = key.extract()?;
        match key.as_str() {
            "temperature" => thermo.temperature = value.extract()?,
            "pressure" => thermo.pressure = value.extract()?,
            "quasi_harmonic" => {
                thermo.quasi_harmonic =
                    QuasiHarmonic::parse(&value.extract::<String>()?).map_err(err)?;
            }
            "cutoff" => thermo.cutoff = value.extract()?,
            "lowest" => thermo.lowest = value.extract()?,
            "enthalpy" => {
                thermo.enthalpy = value
                    .extract::<Option<String>>()?
                    .map(|name| EnthalpyTreatment::parse(&name))
                    .transpose()
                    .map_err(err)?;
            }
            "enthalpy_cutoff" => thermo.enthalpy_cutoff = value.extract()?,
            "concentration" => thermo.concentration = value.extract()?,
            "frequency_scale" => thermo.frequency_scale = value.extract()?,
            "zero_point_scale" => thermo.zero_point_scale = value.extract()?,
            "invert_below" => thermo.invert_below = value.extract()?,
            "rotor_moment" => {
                thermo.rotor_moment =
                    RotorMoment::parse(&value.extract::<String>()?).map_err(err)?;
            }
            "symmetry_number" => {
                // A symmetry number that is given is used as it is: no point-group search.
                thermo.symmetry_number = value.extract()?;
                s.symmetry_tolerance = None;
            }
            "symmetry_tolerance" => s.symmetry_tolerance = value.extract()?,
            "project" => s.project_rigid = value.extract()?,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown setting '{other}' (temperature, pressure, quasi_harmonic, cutoff, \
                     lowest, enthalpy, enthalpy_cutoff, concentration, frequency_scale, \
                     zero_point_scale, invert_below, rotor_moment, symmetry_number, \
                     symmetry_tolerance, project)"
                )))
            }
        }
    }
    s.thermo = thermo;
    Ok(s)
}

/// Normal modes, point group, and thermochemistry of a molecule with its Cartesian Hessian
/// and electronic energy (Hartree): a dict with `modes` (as [`normal_modes`]),
/// `point_group`, `symmetry_number`, `thermo` (temperature, pressure, zero-point energy,
/// `energy_0k`, rotational constants, and entropy, heat capacities, internal energy,
/// enthalpy, and Gibbs free energy with their electronic, translational, rotational, and
/// vibrational parts; Hartree and Hartree/K per molecule), and `text` (the report that
/// `optmain -freq` writes).
///
/// `options` are the choices of `-freq`: `temperature` (K), `pressure` (Pa),
/// `quasi_harmonic` (`grimme`, `truhlar`, `none`), `cutoff` and `lowest` (cm^-1),
/// `enthalpy` (`head-gordon`, `truhlar`, `none`), `enthalpy_cutoff`, `concentration`
/// (mol/L), `frequency_scale`, `zero_point_scale`, `invert_below` (cm^-1), `rotor_moment`
/// (`global`, `molecule`), `symmetry_number` (used as given, without a point-group search),
/// `symmetry_tolerance` (Bohr; `None`: no search, `C1`), `project` (remove translations
/// and rotations). `gradient` removes the direction of a gradient from the modes.
#[pyfunction]
#[pyo3(signature = (symbols, positions_bohr, hessian, energy, charge = 0, multiplicity = 1, options = None, gradient = None))]
#[allow(clippy::too_many_arguments)]
fn vibrational_analysis<'py>(
    py: Python<'py>,
    symbols: Vec<String>,
    positions_bohr: Vec<f64>,
    hessian: Vec<f64>,
    energy: f64,
    charge: i64,
    multiplicity: i64,
    options: Option<Bound<'py, PyDict>>,
    gradient: Option<Vec<f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let sys = system(&symbols, &positions_bohr, charge, multiplicity, None)?;
    let n = positions_bohr.len();
    let h = square(&hessian, n)?;
    let left_out = directions(n, gradient)?;
    let chosen = settings(&sys, options.as_ref())?;
    let analysis = py
        .allow_threads(|| vib::analyze_without(&sys, &h, energy, &chosen, &left_out))
        .map_err(err)?;
    let t = &analysis.thermo;
    let thermo = PyDict::new(py);
    thermo.set_item("temperature", t.temperature)?;
    thermo.set_item("pressure", t.pressure)?;
    thermo.set_item("electronic_energy", t.electronic_energy)?;
    thermo.set_item("zero_point_energy", t.zero_point_energy)?;
    thermo.set_item("energy_0k", t.energy_0k)?;
    thermo.set_item(
        "rotational_constants_ghz",
        t.rotational_constants_ghz.to_vec(),
    )?;
    thermo.set_item("entropy", parts(py, &t.entropy)?)?;
    thermo.set_item("cv", parts(py, &t.cv)?)?;
    thermo.set_item("cp", parts(py, &t.cp)?)?;
    thermo.set_item("internal_energy", parts(py, &t.internal_energy)?)?;
    thermo.set_item("enthalpy", parts(py, &t.enthalpy)?)?;
    thermo.set_item("gibbs", parts(py, &t.gibbs)?)?;
    thermo.set_item("vibrations", t.vibrations)?;
    thermo.set_item("imaginary", t.imaginary)?;
    thermo.set_item("inverted", t.inverted)?;
    let d = PyDict::new(py);
    d.set_item("modes", modes_dict(py, &analysis.modes)?)?;
    d.set_item("point_group", &analysis.point_group.label)?;
    d.set_item("symmetry_number", t.symmetry_number)?;
    d.set_item("thermo", thermo)?;
    d.set_item("text", vib::thermochemistry_text(&analysis))?;
    Ok(d)
}

fn flat(c: &Col<f64>) -> Vec<f64> {
    c.iter().copied().collect()
}

fn columns(
    name: &str,
    rows: Option<Vec<Vec<f64>>>,
    count: usize,
    n: usize,
) -> PyResult<Option<Vec<Col<f64>>>> {
    let Some(rows) = rows else {
        return Ok(None);
    };
    if rows.len() != count {
        return Err(PyValueError::new_err(format!(
            "{} {name} for {count} path points",
            rows.len()
        )));
    }
    rows.iter()
        .enumerate()
        .map(|(k, r)| {
            if r.len() == n {
                Ok(Col::from_fn(n, |i| r[i]))
            } else {
                Err(PyValueError::new_err(format!(
                    "the {name} of path point {k} has {} components, expected {n}",
                    r.len()
                )))
            }
        })
        .collect::<PyResult<Vec<_>>>()
        .map(Some)
}

/// The vibrations transverse to a reaction path and where the path bifurcates (the analysis
/// of `freqmain -path`, `mopt_rs::vib::transverse_profile`).
///
/// `positions_bohr` and `hessians` have one flat list per point of the path (Bohr,
/// Hartree/Bohr^2, row-major), in a common orientation. The tangent of the path at a point
/// is chosen as `freqmain -path_tangent` does: with `tangent="gradient"` (the default, a
/// steepest-descent path) the gradient where its RMS is above `stationary_rms`
/// (Hartree/Bohr) and the line through the neighbouring points elsewhere; with
/// `tangent="points"` that line everywhere; with `tangent="given"` the displacements of
/// `tangents` (one flat list per point). `project` removes translations and rotations.
///
/// The result is a dict: `arc_length` (sqrt(amu) Bohr), `tangent` (unit tangents in
/// mass-weighted coordinates), `tangents_from_points` (Cartesian, `path_tangents`; for two
/// points or more), `gradient_alignment` (overlap of each gradient with that tangent, `None`
/// without gradients; NaN where a gradient vanishes), `curvature` (of the path) and
/// `curvature_vector`, `wavenumbers` and `curvatures` (of the transverse modes at every point,
/// ascending; cm^-1 and Hartree / (Bohr^2 amu)), `curvature_couplings`, `zero_point_energy`
/// (Hartree), `lowest_wavenumber`, `lowest_curvature`, `lowest_mode` (Cartesian), `overlap`
/// (of the lowest mode with that of the point before), `tracks`, `mode_couplings` (flat,
/// row-major), `tangent_source` (`"gradient"` or `"points"` per point), and `bifurcations`: the
/// valley-ridge transitions, each a dict with `after` (index of the last point with a positive
/// lowest curvature), `arc_length` (where it is zero by linear interpolation),
/// `curvature_before` and `curvature_after` (the lowest curvatures of the two points), and
/// `mode` (Cartesian displacement of the mode that turns unstable).
#[pyfunction]
#[pyo3(signature = (symbols, positions_bohr, hessians, gradients = None, tangent = "gradient", tangents = None, project = true, stationary_rms = 3e-4))]
#[allow(clippy::too_many_arguments)]
fn transverse_profile<'py>(
    py: Python<'py>,
    symbols: Vec<String>,
    positions_bohr: Vec<Vec<f64>>,
    hessians: Vec<Vec<f64>>,
    gradients: Option<Vec<Vec<f64>>>,
    tangent: &str,
    tangents: Option<Vec<Vec<f64>>>,
    project: bool,
    stationary_rms: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let count = positions_bohr.len();
    if count == 0 {
        return Err(PyValueError::new_err("a path needs at least one point"));
    }
    if hessians.len() != count {
        return Err(PyValueError::new_err(format!(
            "{} Hessians for {count} path points",
            hessians.len()
        )));
    }
    let systems: Vec<MolecularSystem> = positions_bohr
        .iter()
        .map(|x| system(&symbols, x, 0, 1, None))
        .collect::<PyResult<Vec<_>>>()?;
    let n = 3 * symbols.len();
    let gradients = columns("gradients", gradients, count, n)?;
    let given = columns("tangents", tangents, count, n)?;
    let from_points = if count > 1 {
        Some(vib::path_tangents(&systems).map_err(err)?)
    } else {
        None
    };
    let alignment = match (&gradients, count > 1) {
        (Some(g), true) => Some(vib::gradient_alignment(&systems, g).map_err(err)?),
        _ => None,
    };
    let mut sources = Vec::with_capacity(count);
    let mut points = Vec::with_capacity(count);
    for (k, sys) in systems.iter().enumerate() {
        let direction = match tangent {
            "given" => {
                let Some(t) = &given else {
                    return Err(PyValueError::new_err(
                        "tangent='given' needs the tangents of the points",
                    ));
                };
                sources.push("given");
                Direction::Displacement(t[k].clone())
            }
            "gradient" | "points" => {
                let moving = gradients.as_ref().and_then(|g| {
                    let rms = (g[k].iter().map(|v| v * v).sum::<f64>() / n as f64).sqrt();
                    (tangent == "gradient" && rms > stationary_rms).then(|| g[k].clone())
                });
                match (moving, &from_points) {
                    (Some(g), _) => {
                        sources.push("gradient");
                        Direction::Gradient(g)
                    }
                    (None, Some(t)) => {
                        sources.push("points");
                        t[k].clone()
                    }
                    (None, None) => {
                        return Err(PyValueError::new_err(
                            "a path of one point needs a gradient that does not vanish or \
                             tangent='given'",
                        ))
                    }
                }
            }
            other => {
                return Err(PyValueError::new_err(format!(
                    "tangent must be 'gradient', 'points', or 'given', got '{other}'"
                )))
            }
        };
        points.push(vib::PathPoint {
            system: sys.clone(),
            tangent: direction,
            hessian: square(&hessians[k], n)?,
        });
    }
    let profile = py
        .allow_threads(|| vib::transverse_profile(&points, project))
        .map_err(err)?;
    let d = PyDict::new(py);
    d.set_item("arc_length", profile.arc_length.clone())?;
    d.set_item(
        "tangent",
        profile.tangent.iter().map(flat).collect::<Vec<_>>(),
    )?;
    d.set_item(
        "tangents_from_points",
        from_points.map(|t| {
            t.iter()
                .map(|dir| match dir {
                    Direction::Gradient(v) | Direction::Displacement(v) => flat(v),
                })
                .collect::<Vec<_>>()
        }),
    )?;
    d.set_item("gradient_alignment", alignment)?;
    d.set_item("tangent_source", sources)?;
    d.set_item("curvature", profile.curvature.clone())?;
    d.set_item(
        "curvature_vector",
        profile
            .curvature_vector
            .iter()
            .map(flat)
            .collect::<Vec<_>>(),
    )?;
    d.set_item("wavenumbers", profile.wavenumbers.clone())?;
    d.set_item(
        "curvatures",
        profile
            .modes
            .iter()
            .map(|m| m.curvatures.clone())
            .collect::<Vec<_>>(),
    )?;
    d.set_item("curvature_couplings", profile.curvature_couplings.clone())?;
    d.set_item("zero_point_energy", profile.zero_point_energy.clone())?;
    d.set_item("lowest_wavenumber", profile.lowest_wavenumber.clone())?;
    d.set_item("lowest_curvature", profile.lowest_curvature.clone())?;
    d.set_item(
        "lowest_mode",
        profile.lowest_mode.iter().map(flat).collect::<Vec<_>>(),
    )?;
    d.set_item("overlap", profile.overlap.clone())?;
    d.set_item("tracks", profile.tracks.clone())?;
    d.set_item(
        "mode_couplings",
        profile
            .mode_couplings
            .iter()
            .map(|m| {
                (0..m.nrows())
                    .flat_map(|i| (0..m.ncols()).map(move |j| m[(i, j)]))
                    .collect::<Vec<f64>>()
            })
            .collect::<Vec<_>>(),
    )?;
    let found = pyo3::types::PyList::empty(py);
    for b in &profile.bifurcations {
        let one = PyDict::new(py);
        one.set_item("after", b.after)?;
        one.set_item("arc_length", b.arc_length)?;
        one.set_item("curvature_before", profile.lowest_curvature[b.after])?;
        one.set_item("curvature_after", profile.lowest_curvature[b.after + 1])?;
        one.set_item("mode", flat(&b.mode))?;
        found.append(one)?;
    }
    d.set_item("bifurcations", found)?;
    Ok(d)
}

/// Add the functions of this module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(transverse_profile, m)?)?;
    m.add_function(wrap_pyfunction!(normal_modes, m)?)?;
    m.add_function(wrap_pyfunction!(point_group, m)?)?;
    m.add_function(wrap_pyfunction!(rotational_constants, m)?)?;
    m.add_function(wrap_pyfunction!(vibrational_analysis, m)?)?;
    Ok(())
}
