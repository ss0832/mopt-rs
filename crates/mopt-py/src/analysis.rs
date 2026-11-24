//! Analyses of a series of structures from Python: distances between structures, classical
//! multidimensional scaling, principal components, and the dynamic modes of an
//! optimization.
//!
//! Matrices cross the boundary as lists of rows.
//!
//! MultiOptPy: `PESAnalyzer/cmds_analysis.py`, `pca_analysis.py`, `koopman_analysis.py`.
//!
//! This file is part of mopt-py (the Python interface of mopt-rs).
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Mat;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use mopt_rs::analysis::diabatic;
use mopt_rs::analysis::koopman;
use mopt_rs::analysis::orbitals::{nro, roet, tracking};
use mopt_rs::analysis::path_map;
use mopt_rs::domain::{MolecularSystem, OrbitalSet};
use mopt_rs::path::interp::{self, FirstPath, PathFrame, PathSettings};
use mopt_rs::pes::states::StateCharacters;

use super::{cell_from, err, system};

fn rows(m: &Mat<f64>) -> Vec<Vec<f64>> {
    (0..m.nrows())
        .map(|i| (0..m.ncols()).map(|j| m[(i, j)]).collect())
        .collect()
}

fn matrix(data: &[Vec<f64>]) -> PyResult<Mat<f64>> {
    let columns = data.first().map_or(0, Vec::len);
    if data.iter().any(|row| row.len() != columns) {
        return Err(PyValueError::new_err("the rows differ in length"));
    }
    Ok(Mat::from_fn(data.len(), columns, |i, j| data[i][j]))
}

fn series(
    symbols: &[String],
    frames_bohr: &[Vec<f64>],
    cell_bohr: Option<Vec<f64>>,
) -> PyResult<Vec<MolecularSystem>> {
    let cell = cell_from(cell_bohr)?;
    frames_bohr
        .iter()
        .map(|frame| system(symbols, frame, 0, 1, cell.clone()))
        .collect()
}

/// The distances between all pairs of structures of the same atoms: for each pair
/// `sqrt(sum_i m_i |r_i - r'_i|^2)` after the mass-weighted superposition of one on the
/// other, in Angstrom amu^(1/2). `frames_bohr` are flat lists of positions in Bohr.
#[pyfunction]
#[pyo3(signature = (symbols, frames_bohr, cell_bohr = None))]
fn structure_distances(
    py: Python<'_>,
    symbols: Vec<String>,
    frames_bohr: Vec<Vec<f64>>,
    cell_bohr: Option<Vec<f64>>,
) -> PyResult<Vec<Vec<f64>>> {
    let frames = series(&symbols, &frames_bohr, cell_bohr)?;
    let d = py
        .allow_threads(|| path_map::distance_matrix(&frames))
        .map_err(err)?;
    Ok(rows(&d))
}

/// Classical multidimensional scaling of a symmetric matrix of distances: a dict with
/// `coordinates` (a row per point), `eigenvalues` (all, largest first), `contributions`
/// (the share of each coordinate in the sum of the positive eigenvalues), and
/// `reproducibility` (the sum of all eigenvalues over that of the positive ones).
#[pyfunction]
#[pyo3(signature = (distances, components = 2))]
fn classical_mds<'py>(
    py: Python<'py>,
    distances: Vec<Vec<f64>>,
    components: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let map = path_map::classical_mds(&matrix(&distances)?, components).map_err(err)?;
    let d = PyDict::new(py);
    d.set_item("coordinates", rows(&map.coordinates))?;
    d.set_item("eigenvalues", map.eigenvalues)?;
    d.set_item("contributions", map.contributions)?;
    d.set_item("reproducibility", map.reproducibility)?;
    Ok(d)
}

/// Principal components of `data` (a row per structure, a column per variable): a dict
/// with `variances`, `contributions`, `vectors` (one unit vector per component), `scores`
/// (a row per structure), `means`, and `scales`. With `standardize` the variables are
/// divided by their standard deviations.
#[pyfunction]
#[pyo3(signature = (data, standardize = false, components = None))]
fn principal_components<'py>(
    py: Python<'py>,
    data: Vec<Vec<f64>>,
    standardize: bool,
    components: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    let x = matrix(&data)?;
    let pca = path_map::principal_components(&x, standardize, components.unwrap_or(x.nrows()))
        .map_err(err)?;
    let d = PyDict::new(py);
    d.set_item("variances", pca.variances)?;
    d.set_item("contributions", pca.contributions)?;
    d.set_item("vectors", rows(&pca.vectors.transpose().to_owned()))?;
    d.set_item("scores", rows(&pca.scores))?;
    d.set_item("means", pca.means)?;
    d.set_item("scales", pca.scales)?;
    Ok(d)
}

/// Variables of a series of structures for [`principal_components`]: `(values, names)`
/// with a row per structure. `kind` is `cartesian` (mass-weighted coordinates after
/// superposition on the first structure, Angstrom amu^(1/2)), `distances` (all interatomic
/// distances, Angstrom), or `z_matrix` (a chain of distances, angles, and dihedral angles
/// in the order of the atoms; Angstrom and degrees).
#[pyfunction]
#[pyo3(signature = (kind, symbols, frames_bohr, cell_bohr = None))]
fn path_variables(
    kind: &str,
    symbols: Vec<String>,
    frames_bohr: Vec<Vec<f64>>,
    cell_bohr: Option<Vec<f64>>,
) -> PyResult<(Vec<Vec<f64>>, Vec<String>)> {
    let frames = series(&symbols, &frames_bohr, cell_bohr)?;
    let variables = match kind.to_ascii_lowercase().as_str() {
        "cartesian" => path_map::cartesian_variables(&frames),
        "distances" => path_map::distance_variables(&frames),
        "z_matrix" => path_map::z_matrix_variables(&frames),
        other => {
            return Err(PyValueError::new_err(format!(
                "kind must be cartesian, distances, or z_matrix; got '{other}'"
            )))
        }
    }
    .map_err(err)?;
    Ok((rows(&variables.values), variables.names))
}

/// The observables of one iteration for [`dynamic_modes`]: the coordinates, their powers up
/// to `degree`, the distances between all pairs of atoms, and the gradient when given.
#[pyfunction]
#[pyo3(signature = (positions, gradient = None, degree = koopman::DEFAULT_DEGREE))]
fn koopman_observables(positions: Vec<f64>, gradient: Option<Vec<f64>>, degree: usize) -> Vec<f64> {
    koopman::observables(&positions, gradient.as_deref(), degree)
}

/// The dynamic modes of consecutive iterations (`snapshots`: the observables of each): a
/// dict with `rank`, `eigenvalues` (pairs of real and imaginary part, largest modulus
/// first), `modes` (for each eigenvalue, pairs of real and imaginary part per observable),
/// and `behavior` (what each eigenvalue says about the iterations); `None` when the
/// observables do not change.
#[pyfunction]
#[pyo3(signature = (snapshots, rank_threshold = koopman::DEFAULT_RANK_THRESHOLD))]
fn dynamic_modes<'py>(
    py: Python<'py>,
    snapshots: Vec<Vec<f64>>,
    rank_threshold: f64,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    let Some(modes) = koopman::dynamic_modes(&snapshots, rank_threshold).map_err(err)? else {
        return Ok(None);
    };
    let d = PyDict::new(py);
    d.set_item("rank", modes.rank)?;
    d.set_item(
        "behavior",
        modes
            .eigenvalues
            .iter()
            .map(|e| koopman::behavior(*e))
            .collect::<Vec<_>>(),
    )?;
    d.set_item(
        "eigenvalues",
        modes
            .eigenvalues
            .iter()
            .map(|e| (e[0], e[1]))
            .collect::<Vec<_>>(),
    )?;
    d.set_item(
        "modes",
        modes
            .modes
            .iter()
            .map(|mode| mode.iter().map(|c| (c[0], c[1])).collect::<Vec<_>>())
            .collect::<Vec<_>>(),
    )?;
    Ok(Some(d))
}

/// One set of orbitals from Python: coefficients (a row per atomic orbital), energies,
/// occupations.
type OrbitalInput = (Vec<Vec<f64>>, Vec<f64>, Vec<f64>);

fn orbital_set(input: OrbitalInput) -> PyResult<OrbitalSet> {
    let (coefficients, energies, occupations) = input;
    OrbitalSet::new(matrix(&coefficients)?, energies, occupations).map_err(err)
}

/// `other` brought into the order, rotation (among degenerate orbitals), and sign of
/// `reference` by the overlaps of the orbitals: a dict with `coefficients`, `energies`,
/// `occupations`, `source` (the index of each orbital's partner in `other`), and `overlaps`
/// (the cosine between each orbital and what follows it). Sets are `(coefficients,
/// energies, occupations)` with a row of coefficients per atomic orbital; `overlap` is that
/// of the atomic orbitals at the reference structure.
#[pyfunction]
#[pyo3(signature = (overlap, reference, other, degeneracy = tracking::DEFAULT_DEGENERACY))]
fn follow_orbitals<'py>(
    py: Python<'py>,
    overlap: Vec<Vec<f64>>,
    reference: OrbitalInput,
    other: OrbitalInput,
    degeneracy: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let followed = tracking::follow(
        &orbital_set(reference)?,
        &orbital_set(other)?,
        &matrix(&overlap)?,
        degeneracy,
    )
    .map_err(err)?;
    let d = PyDict::new(py);
    d.set_item("coefficients", rows(&followed.orbitals.coefficients))?;
    d.set_item("energies", followed.orbitals.energies)?;
    d.set_item("occupations", followed.orbitals.occupations)?;
    d.set_item("source", followed.source)?;
    d.set_item("overlaps", followed.overlaps)?;
    Ok(d)
}

/// The natural reaction orbitals of one set of orbitals from the orbitals at the structure
/// and at structures displaced by `+step` and `-step` along a direction (Ebisawa et al.,
/// Phys. Chem. Chem. Phys. 2022, 24, 3532): a dict with `singular_values`, `shares`,
/// `sum_of_squares`, `occupied` and `virtuals` (the orbitals of the pairs in the atomic
/// orbitals, a row per atomic orbital), `occupied_weights` and `virtual_weights` (in the
/// orbitals of the calculation), `occupied_indices`, `virtual_indices`, and
/// `weakest_overlap` (of an occupied orbital with what followed it).
#[pyfunction]
#[pyo3(signature = (overlap, reference, plus, minus, step, degeneracy = tracking::DEFAULT_DEGENERACY))]
fn natural_reaction_orbitals<'py>(
    py: Python<'py>,
    overlap: Vec<Vec<f64>>,
    reference: OrbitalInput,
    plus: OrbitalInput,
    minus: OrbitalInput,
    step: f64,
    degeneracy: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let reference = orbital_set(reference)?;
    let response = nro::response_matrix(
        &reference,
        &orbital_set(plus)?,
        &orbital_set(minus)?,
        &matrix(&overlap)?,
        step,
        degeneracy,
    )
    .map_err(err)?;
    let found = nro::natural_reaction_orbitals(&reference, &response.u).map_err(err)?;
    let d = PyDict::new(py);
    d.set_item(
        "weakest_overlap",
        response.weakest_overlap(&found.occupied_indices),
    )?;
    d.set_item("singular_values", found.singular_values)?;
    d.set_item("shares", found.shares)?;
    d.set_item("sum_of_squares", found.sum_of_squares)?;
    d.set_item("occupied", rows(&found.occupied))?;
    d.set_item("virtuals", rows(&found.virtuals))?;
    d.set_item("occupied_weights", rows(&found.occupied_weights))?;
    d.set_item("virtual_weights", rows(&found.virtual_weights))?;
    d.set_item("occupied_indices", found.occupied_indices)?;
    d.set_item("virtual_indices", found.virtual_indices)?;
    Ok(d)
}

/// The displacement of Cartesian coordinates that is a step of length `step` (amu^(1/2)
/// Bohr) in mass-weighted coordinates along the Cartesian vector `direction`.
#[pyfunction]
fn mass_weighted_displacement(
    masses_amu: Vec<f64>,
    direction: Vec<f64>,
    step: f64,
) -> PyResult<Vec<f64>> {
    let d = faer::Col::from_fn(direction.len(), |k| direction[k]);
    let dx = nro::mass_weighted_displacement(&masses_amu, &d, step).map_err(err)?;
    Ok(dx.iter().copied().collect())
}

/// The orbitals whose energies change most between the two ends of a path: `(relative
/// changes, occupied reactive orbital or None, unoccupied reactive orbital or None)` from
/// the energies of the followed orbitals at the first and the last point.
#[pyfunction]
fn reactive_orbitals(
    first: Vec<f64>,
    last: Vec<f64>,
    occupations: Vec<f64>,
) -> PyResult<(Vec<f64>, Option<usize>, Option<usize>)> {
    let found = roet::reactive_orbitals(&first, &last, &occupations).map_err(err)?;
    Ok((found.changes, found.occupied, found.unoccupied))
}

/// `count` images between two structures of the same atoms, made without a calculation by
/// `method` (`linear`, `lst`, `idpp`, `sidpp`, `geodesic`): the first chain of a
/// relaxation. Positions are flat lists in Bohr; `symbols` gives the elements (the
/// geodesic uses their covalent radii), `cell_bohr` the lattice vectors of a periodic
/// cell, `fixed_atoms` the atoms (from zero) that stay where they are. Returns the images,
/// the iterations of the relaxations, and whether all of them converged.
#[pyfunction]
#[pyo3(signature = (method, first_bohr, last_bohr, count, symbols = None, cell_bohr = None, fixed_atoms = None, max_iterations = 500))]
#[allow(clippy::too_many_arguments)]
fn images_between(
    py: Python<'_>,
    method: &str,
    first_bohr: Vec<f64>,
    last_bohr: Vec<f64>,
    count: usize,
    symbols: Option<Vec<String>>,
    cell_bohr: Option<Vec<f64>>,
    fixed_atoms: Option<Vec<usize>>,
    max_iterations: usize,
) -> PyResult<(Vec<Vec<f64>>, usize, bool)> {
    let method = FirstPath::from_name(method).map_err(err)?;
    let cell = cell_from(cell_bohr)?;
    let elements = match &symbols {
        Some(symbols) => system(symbols, &first_bohr, 0, 1, cell.clone())?.elements,
        None => Vec::new(),
    };
    let frame = PathFrame {
        elements,
        cell,
        fixed_components: fixed_atoms
            .unwrap_or_default()
            .into_iter()
            .flat_map(|atom| (0..3).map(move |k| 3 * atom + k))
            .collect(),
    };
    let settings = PathSettings {
        max_iterations,
        ..PathSettings::default()
    };
    let column = |x: &[f64]| faer::Col::from_fn(x.len(), |k| x[k]);
    let (first, last) = (column(&first_bohr), column(&last_bohr));
    let made = py
        .allow_threads(|| interp::images_between(method, &first, &last, count, &frame, &settings))
        .map_err(err)?;
    Ok((
        made.images
            .iter()
            .map(|image| image.iter().copied().collect())
            .collect(),
        made.iterations,
        made.converged,
    ))
}

/// Add the functions of this module.
/// Excited states as Python hands them over: orbitals (a row per basis function, the
/// occupied ones first), the number of occupied ones, and amplitudes (a row per single
/// excitation `(i, a)` at `i * n_virtual + a`, a column per state).
type StatesInput = (Vec<Vec<f64>>, usize, Vec<Vec<f64>>);

fn state_characters(input: StatesInput) -> PyResult<StateCharacters> {
    let (orbitals, occupied, amplitudes) = input;
    let amplitudes = matrix(&amplitudes)?;
    let energies = vec![0.0; amplitudes.ncols()];
    StateCharacters::new(matrix(&orbitals)?, occupied, energies, amplitudes).map_err(err)
}

/// The squared overlaps of the excited states of a structure (rows) with those of another
/// (columns), and the share of the occupied orbitals of the first that is still occupied
/// in the second: what `-es_follow` compares to stay with a state. Each structure is
/// `(orbitals, occupied, amplitudes)` as `Engine.state_characters` describes them.
///
/// With `signed` the overlaps themselves, with their signs (what a diabatization of two
/// states takes, `diabatic_states`), in the place of their squares.
#[pyfunction]
#[pyo3(signature = (first, second, signed=false))]
fn state_overlaps(
    first: StatesInput,
    second: StatesInput,
    signed: bool,
) -> PyResult<(Vec<Vec<f64>>, f64)> {
    let (first, second) = (state_characters(first)?, state_characters(second)?);
    let (overlaps, kept) = if signed {
        first.overlaps(&second)
    } else {
        first.squared_overlaps(&second)
    }
    .map_err(err)?;
    Ok((rows(&overlaps), kept))
}

fn crossing_dict<'py>(
    py: Python<'py>,
    crossing: &diabatic::DiabaticCrossing,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("states", crossing.states)?;
    d.set_item("position", crossing.position)?;
    d.set_item("energy", crossing.energy)?;
    d.set_item("coupling", crossing.coupling)?;
    d.set_item("slope_difference", crossing.slope_difference)?;
    d.set_item("slopes", crossing.slopes.to_vec())?;
    Ok(d)
}

/// Adiabatic states along a path as diabatic states (`mopt_rs::analysis::diabatic`, the
/// propagative block-diagonalisation diabatisation): `energies[k]` are the energies of the
/// states at point `k` (ascending, two or more), `arc_lengths` the positions of the
/// points, and `overlaps[k][i][j]` the overlap of state `i` at point `k` with state `j` at
/// point `k + 1` (signed; `state_overlaps(..., signed=True)` gives them). Returns the
/// diabatic potential matrix at every point (`potentials`: the energies of the diabatic
/// states on the diagonal, their couplings off it), the share of every adiabatic state in
/// every diabatic one (`weights`), what the next point keeps of the states (`kept`), the
/// `crossings` of two diabatic states (`states`: the two, counted from 0), and for every
/// two states that are neighbours in energy the crossing of the two-state model fitted to
/// their smallest gap (`from_gaps`, `None` where the gap has no minimum inside the path).
#[pyfunction]
fn diabatic_states<'py>(
    py: Python<'py>,
    arc_lengths: Vec<f64>,
    energies: Vec<Vec<f64>>,
    overlaps: Vec<Vec<Vec<f64>>>,
) -> PyResult<Bound<'py, PyDict>> {
    if arc_lengths.len() != energies.len() {
        return Err(PyValueError::new_err(
            "as many positions as points of energies are needed",
        ));
    }
    let overlaps = overlaps
        .iter()
        .map(|o| matrix(o))
        .collect::<PyResult<Vec<_>>>()?;
    let states = diabatic::from_overlaps(&energies, &overlaps).map_err(err)?;
    let d = PyDict::new(py);
    let crossings = diabatic::crossings(&arc_lengths, &states)
        .iter()
        .map(|crossing| crossing_dict(py, crossing))
        .collect::<PyResult<Vec<_>>>()?;
    d.set_item("crossings", crossings)?;
    let n = states.states();
    let level = |i: usize| energies.iter().map(|e| e[i]).collect::<Vec<f64>>();
    let mut from_gaps = Vec::with_capacity(n - 1);
    for i in 0..n - 1 {
        from_gaps.push(
            match diabatic::from_gap(&arc_lengths, &level(i), &level(i + 1)) {
                Some(mut model) => {
                    model.states = (i, i + 1);
                    crossing_dict(py, &model)?.into_any().unbind()
                }
                None => py.None(),
            },
        );
    }
    d.set_item("from_gaps", from_gaps)?;
    d.set_item(
        "potentials",
        states.potentials.iter().map(rows).collect::<Vec<_>>(),
    )?;
    d.set_item(
        "weights",
        states.weights.iter().map(rows).collect::<Vec<_>>(),
    )?;
    d.set_item("kept", states.kept)?;
    Ok(d)
}

/// The Landau-Zener probability that a passage through a crossing stays on its diabatic
/// state: the coupling (Hartree), the slope difference of the diabatic energies along a
/// mass-weighted path (Hartree per amu^(1/2) Bohr), and the kinetic energy of the motion
/// along the path (Hartree).
#[pyfunction]
fn landau_zener_stay(coupling: f64, slope_difference: f64, kinetic_energy: f64) -> f64 {
    diabatic::landau_zener_stay(coupling, slope_difference, kinetic_energy)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(state_overlaps, m)?)?;
    m.add_function(wrap_pyfunction!(diabatic_states, m)?)?;
    m.add_function(wrap_pyfunction!(landau_zener_stay, m)?)?;
    m.add_function(wrap_pyfunction!(images_between, m)?)?;
    m.add_function(wrap_pyfunction!(structure_distances, m)?)?;
    m.add_function(wrap_pyfunction!(classical_mds, m)?)?;
    m.add_function(wrap_pyfunction!(principal_components, m)?)?;
    m.add_function(wrap_pyfunction!(path_variables, m)?)?;
    m.add_function(wrap_pyfunction!(koopman_observables, m)?)?;
    m.add_function(wrap_pyfunction!(dynamic_modes, m)?)?;
    m.add_function(wrap_pyfunction!(follow_orbitals, m)?)?;
    m.add_function(wrap_pyfunction!(natural_reaction_orbitals, m)?)?;
    m.add_function(wrap_pyfunction!(mass_weighted_displacement, m)?)?;
    m.add_function(wrap_pyfunction!(reactive_orbitals, m)?)?;
    Ok(())
}
