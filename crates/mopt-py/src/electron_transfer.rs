//! The parts of the electron-transfer rate constants (`mopt_rs::kinetics::rate::inner_sphere`,
//! `couplings`, `solvent_activation`, `mopt_rs::workflow::electron_transfer`) one by one: the
//! inner sphere at a crossing, the tunnelling factor of Scher and Holstein, the Marcus and
//! Zhu-Nakamura rate constants at a crossing, the Mulliken-Hush couplings, the
//! charge-transfer index, the solvent activation of SAES, and the apparent free energy of
//! activation of a rate constant. Atomic units (Hartree, Bohr, e Bohr) unless stated.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-py (the Python interface of mopt-rs).
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use pyo3::prelude::*;
use pyo3::types::PyDict;

use mopt_rs::domain::Units;
use mopt_rs::kinetics::rate::couplings::{
    charge_transfer, fragment_charge_difference, generalized_mulliken_hush, mulliken_hush,
};
use mopt_rs::kinetics::rate::inner_sphere::{
    inner_sphere, marcus_levich_jortner, marcus_with_tunnelling, scher_holstein,
    zhu_nakamura_at_crossing, CrossingType,
};
use mopt_rs::kinetics::rate::pcet::{
    pcet_rate, proton_overlaps, proton_states, ProtonStates, PROTON_MASS_AMU,
};
use mopt_rs::kinetics::rate::solvent_activation::SolventActivation;
use mopt_rs::kinetics::rate::transfer_integral::{
    fock_from_orbitals, four_point_reorganization, fragment_transfer_integral, hopping_rate,
    matched_transfer_integral, Carrier, TransferIntegral,
};
use mopt_rs::workflow::electron_transfer::apparent_barrier;

use faer::{Col, Mat};
use mopt_rs::domain::{OrbitalSet, Orbitals};
use mopt_rs::error::MoptError;

use super::err;

/// A matrix from its rows, all of `columns` numbers (`None`: as many as the first).
fn matrix(rows: &[Vec<f64>], columns: Option<usize>, what: &str) -> PyResult<Mat<f64>> {
    let m = columns.unwrap_or_else(|| rows.first().map_or(0, Vec::len));
    if rows.iter().any(|r| r.len() != m) {
        return Err(err(MoptError::config(format!(
            "{what}: rows of different lengths"
        ))));
    }
    Ok(Mat::from_fn(rows.len(), m, |i, j| rows[i][j]))
}

fn transfer_dict<'py>(py: Python<'py>, t: &TransferIntegral) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("coupling", t.coupling)?;
    d.set_item("site_energies", t.site_energies.to_vec())?;
    d.set_item("overlap", t.overlap)?;
    d.set_item("effective", t.effective)?;
    d.set_item(
        "effective_site_energies",
        t.effective_site_energies.to_vec(),
    )?;
    d.set_item("degeneracy", t.degeneracy.to_vec())?;
    Ok(d)
}

/// The transfer integral (Hartree) of the orbitals `a` and `b` in the atomic orbitals of a
/// pair with its Fock matrix and overlap: `a` and `b` are lists of orbitals (one each, or the
/// degenerate sets, whose best-matched combinations are taken). A dict of `coupling`,
/// `site_energies`, `overlap`, `effective`, `effective_site_energies`, `degeneracy`.
#[pyfunction]
fn et_transfer_integral<'py>(
    py: Python<'py>,
    fock: Vec<Vec<f64>>,
    overlap: Vec<Vec<f64>>,
    a: Vec<Vec<f64>>,
    b: Vec<Vec<f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let n = fock.len();
    let f = matrix(&fock, Some(n), "the Fock matrix")?;
    let s = matrix(&overlap, Some(n), "the overlap")?;
    let column = |v: &Vec<f64>| Col::from_fn(v.len(), |i| v[i]);
    let a: Vec<Col<f64>> = a.iter().map(column).collect();
    let b: Vec<Col<f64>> = b.iter().map(column).collect();
    let t = matched_transfer_integral(&f, &s, &a, &b).map_err(err)?;
    transfer_dict(py, &t)
}

/// The Fock matrix `S C diag(e) C^T S` of the orbitals `coefficients` (rows of the atomic
/// orbitals, a column per orbital) with their energies and the overlap.
#[pyfunction]
fn et_fock_from_orbitals(
    overlap: Vec<Vec<f64>>,
    coefficients: Vec<Vec<f64>>,
    energies: Vec<f64>,
) -> PyResult<Vec<Vec<f64>>> {
    let n = overlap.len();
    let s = matrix(&overlap, Some(n), "the overlap")?;
    let c = matrix(&coefficients, Some(energies.len()), "the coefficients")?;
    let occupations = vec![0.0; energies.len()];
    let set = OrbitalSet::new(c, energies, occupations).map_err(err)?;
    let f = fock_from_orbitals(&s, &set);
    Ok((0..f.nrows())
        .map(|i| (0..f.ncols()).map(|j| f[(i, j)]).collect())
        .collect())
}

/// The orbitals of a closed shell: the overlap, the coefficients (rows of the atomic orbitals),
/// the energies, the occupations (up to 2), and the atom of every atomic orbital.
type ClosedShell = (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<f64>, Vec<f64>, Vec<usize>);

fn closed_shell(o: ClosedShell, what: &str) -> PyResult<Orbitals> {
    let (overlap, coefficients, energies, occupations, atoms) = o;
    let n = overlap.len();
    let s = matrix(&overlap, Some(n), what)?;
    let c = matrix(&coefficients, Some(energies.len()), what)?;
    let set = OrbitalSet::new(c, energies, occupations).map_err(err)?;
    Orbitals::new(s, set, None)
        .and_then(|o| o.with_atoms(atoms))
        .map_err(err)
}

/// The transfer integral of the projection method (Hartree) between the frontier orbitals of
/// `carrier` (`hole` or `electron`) of two molecules: the orbitals of the pair, of `a`, and of
/// `b` (each computed alone at its place in the pair), each a closed shell `(overlap,
/// coefficients, energies, occupations, atom_of_ao)`; `atoms_a`, `atoms_b` the atoms (0-based)
/// of the molecules in the pair. A dict as `et_transfer_integral`'s.
#[pyfunction]
#[pyo3(signature = (pair, a, b, atoms_a, atoms_b, carrier = "hole"))]
fn et_projection_coupling<'py>(
    py: Python<'py>,
    pair: ClosedShell,
    a: ClosedShell,
    b: ClosedShell,
    atoms_a: Vec<usize>,
    atoms_b: Vec<usize>,
    carrier: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let carrier = Carrier::parse(carrier).map_err(err)?;
    let pair = closed_shell(pair, "the orbitals of the pair")?;
    let a = closed_shell(a, "the orbitals of the first molecule")?;
    let b = closed_shell(b, "the orbitals of the second molecule")?;
    let t = fragment_transfer_integral(&pair, &a, &b, &atoms_a, &atoms_b, carrier).map_err(err)?;
    transfer_dict(py, &t)
}

fn states_dict<'py>(py: Python<'py>, st: &ProtonStates) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("energies", st.energies.clone())?;
    let rows: Vec<Vec<f64>> = (0..st.vectors.nrows())
        .map(|i| {
            (0..st.vectors.ncols())
                .map(|k| st.vectors[(i, k)])
                .collect()
        })
        .collect();
    d.set_item("vectors", rows)?;
    d.set_item(
        "mean_positions",
        (0..st.len())
            .map(|k| st.mean_position(k))
            .collect::<Vec<f64>>(),
    )?;
    d.set_item("grid", (st.grid.0, st.grid.1))?;
    Ok(d)
}

/// The lowest `count` vibrational states of a proton (or of a particle of `mass_amu`) in the
/// potential `potential` (Hartree) at the points `first + i spacing` (Bohr), by the discrete
/// variable representation of Colbert and Miller: a dict of `energies`, `vectors` (the rows of
/// the points, a column per state, of unit length), `mean_positions`, and `grid`.
#[pyfunction]
#[pyo3(signature = (first, spacing, potential, mass_amu = PROTON_MASS_AMU, count = 10))]
fn et_proton_states<'py>(
    py: Python<'py>,
    first: f64,
    spacing: f64,
    potential: Vec<f64>,
    mass_amu: f64,
    count: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let st = proton_states(first, spacing, &potential, mass_amu, count).map_err(err)?;
    states_dict(py, &st)
}

/// The nonadiabatic PCET rate (1/s) between the proton potentials of the reactant and of the
/// product on one grid (Hartree, with the electronic energies of their diabatic states), the
/// electronic coupling, the solvent reorganization energy, and the free energy `delta_g0` the
/// potentials do not hold (Hartree) at `temperature` (K), with `count` proton states on each
/// side: a dict of `rate`, `channels` (`(mu, nu, share)`, largest first), `overlaps` (rows of
/// the reactant's states), and the states of both sides.
#[pyfunction]
#[pyo3(signature = (first, spacing, reactant, product, coupling, reorganization,
    delta_g0 = 0.0, temperature = 298.15, mass_amu = PROTON_MASS_AMU, count = 10))]
#[allow(clippy::too_many_arguments)]
fn et_pcet_rate<'py>(
    py: Python<'py>,
    first: f64,
    spacing: f64,
    reactant: Vec<f64>,
    product: Vec<f64>,
    coupling: f64,
    reorganization: f64,
    delta_g0: f64,
    temperature: f64,
    mass_amu: f64,
    count: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let r = proton_states(first, spacing, &reactant, mass_amu, count).map_err(err)?;
    let p = proton_states(first, spacing, &product, mass_amu, count).map_err(err)?;
    let k = pcet_rate(&r, &p, coupling, reorganization, delta_g0, kt(temperature)).map_err(err)?;
    let s = proton_overlaps(&r, &p).map_err(err)?;
    let d = PyDict::new(py);
    d.set_item("rate", k.rate)?;
    d.set_item("channels", k.channels)?;
    let rows: Vec<Vec<f64>> = (0..s.nrows())
        .map(|i| (0..s.ncols()).map(|j| s[(i, j)]).collect())
        .collect();
    d.set_item("overlaps", rows)?;
    d.set_item("reactant", states_dict(py, &r)?)?;
    d.set_item("product", states_dict(py, &p)?)?;
    Ok(d)
}

/// The four-point reorganization energy (Hartree) of a hop from `a` to `b`: for each molecule
/// its energy in the charge it ends with, at the equilibrium of the charge it starts with and
/// at its own, `(at_start, at_own)`.
#[pyfunction]
fn et_four_point_reorganization(a: [f64; 2], b: [f64; 2]) -> f64 {
    four_point_reorganization(a, b)
}

/// The Marcus rate (1/s) of a hop with the transfer integral, the reorganization energy, and
/// the free energy of the hop (Hartree) at `temperature` (K).
#[pyfunction]
#[pyo3(signature = (coupling, reorganization, delta_g, temperature = 298.15))]
fn et_hopping_rate(
    coupling: f64,
    reorganization: f64,
    delta_g: f64,
    temperature: f64,
) -> PyResult<f64> {
    hopping_rate(coupling, reorganization, delta_g, kt(temperature)).map_err(err)
}

fn kt(temperature: f64) -> f64 {
    Units::BOLTZMANN_HARTREE_PER_K * temperature
}

/// The inner sphere at a crossing (thesis of A. Mutsuji, eqn (4.S4)-(4.S9)) from the
/// mass-weighted gradients of the two states (Hartree / (Bohr amu^(1/2))) and the inner
/// barrier (Hartree): a dict of `quantum`, `wavenumber_cm1`, `reaction_energy`,
/// `reorganization`, `q_crossing`, `q_final`, `slopes`.
#[pyfunction]
fn et_inner_sphere<'py>(
    py: Python<'py>,
    g_initial: Vec<f64>,
    g_final: Vec<f64>,
    barrier: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let p = inner_sphere(&g_initial, &g_final, barrier).map_err(err)?;
    let d = PyDict::new(py);
    d.set_item("quantum", p.quantum)?;
    d.set_item("wavenumber_cm1", p.wavenumber())?;
    d.set_item("reaction_energy", p.reaction_energy)?;
    d.set_item("reorganization", p.reorganization)?;
    d.set_item("q_crossing", p.q_crossing)?;
    d.set_item("q_final", p.q_final)?;
    d.set_item("slopes", (p.slopes[0], p.slopes[1]))?;
    Ok(d)
}

/// The tunnelling factor of Scher and Holstein (eqn (4.12)-(4.16)) of a mode of the quantum
/// `quantum` with the reaction and reorganization energies of the inner sphere, at
/// `temperature` (K): `(ln Gamma, dG(T), dG(inf))`.
#[pyfunction]
#[pyo3(signature = (quantum, reaction_energy, reorganization, temperature = 298.15))]
fn et_scher_holstein(
    quantum: f64,
    reaction_energy: f64,
    reorganization: f64,
    temperature: f64,
) -> PyResult<(f64, f64, f64)> {
    let f =
        scher_holstein(quantum, reaction_energy, reorganization, kt(temperature)).map_err(err)?;
    Ok((f.ln_gamma, f.barrier_quantum, f.barrier_classical))
}

/// The Marcus rate constant (1/s) with the tunnelling factor `ln_gamma`, eqn (4.1).
#[pyfunction]
#[pyo3(signature = (ln_gamma, coupling, reorganization, barrier, temperature = 298.15))]
fn et_marcus_rate(
    ln_gamma: f64,
    coupling: f64,
    reorganization: f64,
    barrier: f64,
    temperature: f64,
) -> PyResult<f64> {
    marcus_with_tunnelling(ln_gamma, coupling, reorganization, barrier, kt(temperature))
        .map_err(err)
}

/// The Zhu-Nakamura rate constant (1/s) at the crossing of the mass-weighted gradients
/// `g_initial`, `g_final` with the inner barrier `inner_barrier`, the coupling, and the whole
/// barrier `barrier` (with the solvent activation): `(kind, ln kappa, share below the
/// crossing, rate)`, `kind` `peaked` or `sloped`.
#[pyfunction]
#[pyo3(signature = (g_initial, g_final, inner_barrier, coupling, barrier, temperature = 298.15))]
fn et_zhu_nakamura_rate(
    g_initial: Vec<f64>,
    g_final: Vec<f64>,
    inner_barrier: f64,
    coupling: f64,
    barrier: f64,
    temperature: f64,
) -> PyResult<(String, f64, f64, f64)> {
    let p = inner_sphere(&g_initial, &g_final, inner_barrier).map_err(err)?;
    let z = zhu_nakamura_at_crossing(&p, coupling, barrier, inner_barrier, kt(temperature), 1.0)
        .map_err(err)?;
    let kind = match z.kind {
        CrossingType::Peaked => "peaked",
        CrossingType::Sloped => "sloped",
    };
    Ok((kind.to_string(), z.ln_kappa, z.below_crossing, z.rate))
}

/// The Marcus-Levich-Jortner rate constant (1/s): one quantum mode `quantum` with the inner
/// reorganization, a classical solvent of the reorganization `solvent_reorganization`, and the
/// reaction free energy.
#[pyfunction]
#[pyo3(signature = (coupling, quantum, inner_reorganization, solvent_reorganization,
    reaction_energy, temperature = 298.15))]
fn et_marcus_levich_jortner(
    coupling: f64,
    quantum: f64,
    inner_reorganization: f64,
    solvent_reorganization: f64,
    reaction_energy: f64,
    temperature: f64,
) -> PyResult<f64> {
    marcus_levich_jortner(
        coupling,
        quantum,
        inner_reorganization,
        solvent_reorganization,
        reaction_energy,
        kt(temperature),
    )
    .map_err(err)
}

/// The Mulliken-Hush coupling (Hartree), eqn (4.18).
#[pyfunction]
fn et_mulliken_hush(excitation: f64, transition_dipole: [f64; 3], distance: f64) -> PyResult<f64> {
    mulliken_hush(excitation, transition_dipole, distance).map_err(err)
}

/// The generalized Mulliken-Hush coupling (Hartree) of Cave and Newton.
#[pyfunction]
fn et_generalized_mulliken_hush(
    excitation: f64,
    transition_dipole: [f64; 3],
    dipole_difference: [f64; 3],
) -> PyResult<f64> {
    generalized_mulliken_hush(excitation, transition_dipole, dipole_difference).map_err(err)
}

/// The fragment-charge-difference coupling (Hartree) of Voityuk and Rosch from the excitation
/// energy, the populations of the transition density on each atom, the change of the
/// electrons on each atom from the ground to the excited state, and the atoms (0-based) of the
/// donor and of the acceptor.
#[pyfunction]
fn et_fragment_charge_difference(
    excitation: f64,
    transition_population: Vec<f64>,
    population_change: Vec<f64>,
    donor: Vec<usize>,
    acceptor: Vec<usize>,
) -> PyResult<f64> {
    fragment_charge_difference(
        excitation,
        &transition_population,
        &population_change,
        &donor,
        &acceptor,
    )
    .map_err(err)
}

/// The charge-transfer index of the changes of the atomic populations at the positions
/// (Bohr): `(q_CT, d_CT, gained_at, lost_at)`.
#[pyfunction]
fn et_charge_transfer(
    positions: Vec<[f64; 3]>,
    population_change: Vec<f64>,
) -> PyResult<(f64, f64, [f64; 3], [f64; 3])> {
    let c = charge_transfer(&positions, &population_change).map_err(err)?;
    Ok((c.charge, c.distance, c.gained_at, c.lost_at))
}

/// The shifts `(G_SA^i, G_SA^f)` (Hartree) of the solvent activation (SAES) at `alpha` for
/// the solvent reorganization energies of the two states and the reaction energy, with the
/// solvent coordinate of the crossing: `(G_SA^i, G_SA^f, x_s)`.
#[pyfunction]
fn et_solvent_shifts(
    lambda_initial: f64,
    lambda_final: f64,
    reaction_energy: f64,
    alpha: f64,
) -> PyResult<(f64, f64, f64)> {
    let s = SolventActivation::new(lambda_initial, lambda_final, reaction_energy).map_err(err)?;
    let [gi, gf] = s.shifts(alpha).map_err(err)?;
    Ok((gi, gf, s.coordinate(alpha).map_err(err)?))
}

/// The free energy of activation (Hartree) that the Eyring equation gives a unimolecular
/// rate constant (1/s) at `temperature` (K).
#[pyfunction]
#[pyo3(signature = (rate, temperature = 298.15))]
fn et_apparent_barrier(rate: f64, temperature: f64) -> f64 {
    apparent_barrier(rate, temperature)
}

/// Run `etmain` with `args` (as on the command line, without the program name) and return,
/// for every input, its run directory and its crossings (Hartree, 1/s): `alpha_SA`, the
/// barriers `barrier` (the electronic energies with the solvent shift) and
/// `free_energy_barrier`, the coupling, the reorganization energy, the rate constants with
/// each barrier, and the largest rate constants along the scan as `(alpha_SA, log10 k)`
/// (`method` `edeel`). With `-et_method hopping` (`method` `hopping`): the transfer integral
/// `coupling` (`J_eff`), `coupling_raw`, `overlap`, `site_energies`, `degeneracy`, the
/// four-point `reorganization` both ways and their mean, the `reaction_energy` of the hop,
/// the `rates` both ways, and their `apparent_barriers`.
#[pyfunction]
#[pyo3(signature = (args, console = true))]
fn run_etmain<'py>(
    py: Python<'py>,
    args: Vec<String>,
    console: bool,
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    use mopt_rs::cli::etmain;
    use mopt_rs::core_infra::log::Level;
    let registry = super::pycalc::python_registry();
    let refs: Vec<&str> = args.iter().map(String::as_str).collect();
    let config = etmain::config_from_cli(&refs).map_err(err)?;
    let level = console.then_some(Level::Info);
    let runs = py
        .allow_threads(|| etmain::run_with(&config, std::path::Path::new("."), level, &registry))
        .map_err(err)?;
    let mut all = Vec::with_capacity(runs.len());
    for run in runs {
        let d = PyDict::new(py);
        d.set_item("directory", run.directory.to_string_lossy().into_owned())?;
        if let Some(h) = &run.hopping {
            d.set_item("method", "hopping")?;
            let t = &h.coupling;
            d.set_item("coupling", t.effective)?;
            d.set_item("coupling_raw", t.coupling)?;
            d.set_item("overlap", t.overlap)?;
            d.set_item("site_energies", t.site_energies.to_vec())?;
            d.set_item("degeneracy", t.degeneracy.to_vec())?;
            d.set_item("reorganization", h.reorganization.to_vec())?;
            d.set_item("mean_reorganization", h.mean_reorganization())?;
            d.set_item("reaction_energy", h.reaction_energy)?;
            d.set_item("rates", h.rates.to_vec())?;
            d.set_item("apparent_barriers", h.apparent_barriers.to_vec())?;
            all.push(d);
            continue;
        }
        let Some(r) = run.transfer.as_ref() else {
            all.push(d);
            continue;
        };
        d.set_item("method", "edeel")?;
        d.set_item("initial_energy", r.initial.1)?;
        d.set_item("final_energy", r.final_state.1)?;
        d.set_item("initial_free_energy", r.initial_free_energy)?;
        let mut crossings = Vec::with_capacity(r.crossings.len());
        for c in &r.crossings {
            let x = PyDict::new(py);
            x.set_item("alpha_SA", c.alpha)?;
            x.set_item("converged", c.converged)?;
            x.set_item("barrier", c.barrier)?;
            x.set_item("inner_barrier", c.inner_barrier)?;
            x.set_item("free_energy_barrier", c.free_energy_barrier)?;
            x.set_item("seam_imaginary", c.seam_imaginary)?;
            x.set_item("coupling", c.coupling)?;
            x.set_item("coupling_from_others", c.coupling_from_others)?;
            x.set_item("reorganization", c.reorganization)?;
            x.set_item("log10_gamma", c.tunnelling.map(|t| t.log10()))?;
            x.set_item("marcus", c.marcus)?;
            x.set_item("marcus_free", c.marcus_free)?;
            x.set_item("zhu_nakamura", c.zhu_nakamura.map(|z| z.rate))?;
            x.set_item("zhu_nakamura_free", c.zhu_nakamura_free.map(|z| z.rate))?;
            x.set_item("marcus_levich_jortner", c.marcus_levich_jortner)?;
            crossings.push(x);
        }
        d.set_item("crossings", crossings)?;
        d.set_item("best_marcus", r.best_marcus)?;
        d.set_item("best_zhu_nakamura", r.best_zhu_nakamura)?;
        d.set_item("best_marcus_free", r.best_marcus_free)?;
        d.set_item("best_zhu_nakamura_free", r.best_zhu_nakamura_free)?;
        d.set_item("best_marcus_levich_jortner", r.best_marcus_levich_jortner)?;
        all.push(d);
    }
    Ok(all)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_etmain, m)?)?;
    m.add_function(wrap_pyfunction!(et_inner_sphere, m)?)?;
    m.add_function(wrap_pyfunction!(et_scher_holstein, m)?)?;
    m.add_function(wrap_pyfunction!(et_marcus_rate, m)?)?;
    m.add_function(wrap_pyfunction!(et_marcus_levich_jortner, m)?)?;
    m.add_function(wrap_pyfunction!(et_zhu_nakamura_rate, m)?)?;
    m.add_function(wrap_pyfunction!(et_mulliken_hush, m)?)?;
    m.add_function(wrap_pyfunction!(et_generalized_mulliken_hush, m)?)?;
    m.add_function(wrap_pyfunction!(et_fragment_charge_difference, m)?)?;
    m.add_function(wrap_pyfunction!(et_charge_transfer, m)?)?;
    m.add_function(wrap_pyfunction!(et_solvent_shifts, m)?)?;
    m.add_function(wrap_pyfunction!(et_apparent_barrier, m)?)?;
    m.add_function(wrap_pyfunction!(et_transfer_integral, m)?)?;
    m.add_function(wrap_pyfunction!(et_fock_from_orbitals, m)?)?;
    m.add_function(wrap_pyfunction!(et_projection_coupling, m)?)?;
    m.add_function(wrap_pyfunction!(et_four_point_reorganization, m)?)?;
    m.add_function(wrap_pyfunction!(et_hopping_rate, m)?)?;
    m.add_function(wrap_pyfunction!(et_proton_states, m)?)?;
    m.add_function(wrap_pyfunction!(et_pcet_rate, m)?)?;
    Ok(())
}
