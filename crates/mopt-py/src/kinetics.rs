//! The kinetics of `mopt_rs.kinetics`: a rate matrix `K[i][j]` (from `j` to `i`, the diagonal
//! ignored) solved by a solver of the registry, contracted with given eliminated states, its
//! greedy order with the reference times of RCMC, and the scores of the states for an
//! exploration. Detailed balance is detected from the matrix (for the time rule `eigen` and
//! the solver `spectral`). Beyond the canonical model: the models `egme`, `two_grain`,
//! `semi_markov` of a network file, the refinement tasks, the single corrections (Kramers,
//! Grote-Hynes, weak coupling, curvature tunnelling), the direct count of states, the
//! statistics of trajectory pairs, Markov state models, and the contraction of memory kernels.
//!
//! MultiOptPy: `Utils/rcmc.py` (no Python access of its parts besides the queue).
//!
//! This file is part of mopt-py (the Python interface of mopt-rs).
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use mopt_rs::core_infra::json::JsonValue;
use mopt_rs::core_infra::rng::Rng;
use mopt_rs::kinetics::branching::{branching, TrajectoryPair, Weighting};
use mopt_rs::kinetics::contract::{contract, greedy_order, TimeRule};
use mopt_rs::kinetics::derivative::derivatives;
use mopt_rs::kinetics::egme::{Bath, CollisionModel, EgmeSettings};
use mopt_rs::kinetics::hot::HotSettings;
use mopt_rs::kinetics::mass_action::DEFAULT_ODE_TOLERANCE;
use mopt_rs::kinetics::msm::{
    count_matrix, dwell_statistics, implied_timescales, rate_matrix, reversible_transition_matrix,
};
use mopt_rs::kinetics::network::CvtPath;
use mopt_rs::kinetics::network::Friction;
use mopt_rs::kinetics::observe::{scores, Score};
use mopt_rs::kinetics::rate::electron_transfer::{transfer_barrier, TransferForm, TransferStep};
use mopt_rs::kinetics::rate::{
    cvt_ln_factor, BellBarrier, BellSettings, BellTruncation, QrrhoSettings,
    RateCorrectionRegistry, RateSettings, StepContext,
};
use mopt_rs::kinetics::rate::{
    grote_hynes_frequency, kramers_ln_kappa, weak_coupling, RateTheoryRegistry,
};
use mopt_rs::kinetics::refine::{refinement_tasks, Observable, Uncertainties};
use mopt_rs::kinetics::rrkm::{beyer_swinehart, RrkmSettings};
use mopt_rs::kinetics::semi_markov::kernel_contraction;
use mopt_rs::kinetics::solve::{Closure, Evolution, SolverRegistry, SolverSettings};
use mopt_rs::kinetics::species::standard_state_kcal_mol;
use mopt_rs::kinetics::ModelRegistry;
use mopt_rs::kinetics::{
    initial_populations, prepare, Association, Bottleneck, Conditions, EnergyKind, Generator,
    KineticNetwork, KineticsParts, KineticsSettings, ModelSettings, SpeciesSettings,
};
use mopt_rs::workflow::kinetics::{apply_annotations, read_network, ReadNetwork};
use mopt_rs::workflow::mapper::network::Level as NetworkLevel;

use super::err;

/// The tolerance of the detection of detailed balance (`ln` of the ratio of the fluxes).
const BALANCE: f64 = 1e-9;

fn generator(rates: &[Vec<f64>]) -> PyResult<Generator> {
    Ok(Generator::from_matrix(rates)
        .map_err(err)?
        .with_detected_balance(BALANCE))
}

fn settings(tolerance: f64, time_rule: &str, closure: &str) -> PyResult<SolverSettings> {
    Ok(SolverSettings {
        time_rule: TimeRule::parse(time_rule).map_err(err)?,
        tolerance,
        closure: Closure::parse(closure).map_err(err)?,
        legacy_start_known: true,
        ode_tolerance: DEFAULT_ODE_TOLERANCE,
        ..SolverSettings::default()
    })
}

/// The network file `network` with the annotations file `annotations`.
fn annotated(
    network: &str,
    annotations: Option<&str>,
    level: u64,
    energy: EnergyKind,
) -> PyResult<ReadNetwork> {
    let mut read = read_network(
        std::path::Path::new(network),
        NetworkLevel::from_number(level).map_err(err)?,
        energy,
    )
    .map_err(err)?;
    if let Some(path) = annotations {
        let text = std::fs::read_to_string(path)
            .map_err(|e| PyValueError::new_err(format!("{path}: {e}")))?;
        let doc = JsonValue::parse(&text).map_err(err)?;
        apply_annotations(&mut read.network, &doc).map_err(err)?;
    }
    Ok(read)
}

/// The species settings of `concentrations` (mol/L) and `association`.
fn species(concentrations: Vec<(String, f64)>, association: &str) -> PyResult<SpeciesSettings> {
    Ok(SpeciesSettings {
        concentrations,
        association: Association::parse(association).map_err(err)?,
        ..SpeciesSettings::default()
    })
}

/// The populations at `times` from `p0` by `solver`, with the diagnostics of every time.
#[pyfunction]
#[pyo3(signature = (rates, p0, times, *, solver = "propagate", tolerance = 1e-8, time_rule = "gershgorin", closure = "a", trajectories = 10_000, seed = 1))]
#[allow(clippy::too_many_arguments)]
fn kinetics_solve<'py>(
    py: Python<'py>,
    rates: Vec<Vec<f64>>,
    p0: Vec<f64>,
    times: Vec<f64>,
    solver: &str,
    tolerance: f64,
    time_rule: &str,
    closure: &str,
    trajectories: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyDict>> {
    let ev = Evolution::Linear(generator(&rates)?);
    let solver_settings = SolverSettings {
        kmc_trajectories: trajectories,
        ..settings(tolerance, time_rule, closure)?
    };
    let mut s = SolverRegistry::default()
        .create(solver, &solver_settings)
        .map_err(err)?;
    let mut rng = Rng::from_seed(seed);
    let sol = s.solve(&ev, &p0, &times, &mut rng).map_err(err)?;
    let d = PyDict::new(py);
    d.set_item("solver", sol.solver)?;
    d.set_item("times", sol.times)?;
    d.set_item("populations", sol.populations)?;
    let diag = &sol.diagnostics;
    d.set_item(
        "eliminated",
        diag.iter().map(|x| x.eliminated).collect::<Vec<_>>(),
    )?;
    d.set_item(
        "eps_hat",
        diag.iter().map(|x| x.eps_hat).collect::<Vec<_>>(),
    )?;
    d.set_item(
        "sigma_t",
        diag.iter().map(|x| x.sigma_t).collect::<Vec<_>>(),
    )?;
    d.set_item(
        "conservation",
        diag.iter().map(|x| x.conservation).collect::<Vec<_>>(),
    )?;
    d.set_item(
        "superstates",
        sol.superstates
            .iter()
            .map(|s| s.as_ref().map(|s| s.members.clone()))
            .collect::<Vec<_>>(),
    )?;
    d.set_item(
        "detailed_balance",
        ev.generator().map_err(err)?.ln_pi.is_some(),
    )?;
    Ok(d)
}

/// The contraction with the states `eliminate` (in that order): kept, eliminated, `D`, `B`,
/// `C` (row-major), and `1^T M`.
#[pyfunction]
fn kinetics_contract<'py>(
    py: Python<'py>,
    rates: Vec<Vec<f64>>,
    eliminate: Vec<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    let g = generator(&rates)?;
    let c = contract(&g.rates(), &eliminate).map_err(err)?;
    let d = PyDict::new(py);
    d.set_item("kept", c.kept.clone())?;
    d.set_item("eliminated", c.eliminated.clone())?;
    d.set_item("d", c.d.clone())?;
    d.set_item("b", c.b.clone())?;
    d.set_item("c", c.c.clone())?;
    d.set_item("weights", c.weights.clone())?;
    Ok(d)
}

/// The greedy order of RCMC with the reference times of `time_rule`: per step the state,
/// `|D_jj|`, `sigma_hat`, `rho_hat`, the reference time, and `eps_hat`.
#[pyfunction]
#[pyo3(signature = (rates, *, time_rule = "gershgorin"))]
fn kinetics_greedy_order(
    rates: Vec<Vec<f64>>,
    time_rule: &str,
) -> PyResult<Vec<(usize, f64, f64, f64, f64, f64)>> {
    let g = generator(&rates)?;
    let rule = TimeRule::parse(time_rule).map_err(err)?;
    let order = greedy_order(&g.rates(), rule, g.ln_pi.is_some()).map_err(err)?;
    Ok(order
        .steps
        .iter()
        .map(|s| (s.state, s.out, s.sigma_hat, s.rho_hat, s.t_ref, s.eps_hat))
        .collect())
}

/// The scores of the states (`population`, `max_population`, `residence`,
/// `discovery_flux`) at the horizon `t` from `p0`.
#[pyfunction]
#[pyo3(signature = (rates, p0, t, *, score = "population", solver = "propagate", visits = None, tolerance = 1e-8, time_rule = "gershgorin"))]
#[allow(clippy::too_many_arguments)]
fn kinetics_scores(
    rates: Vec<Vec<f64>>,
    p0: Vec<f64>,
    t: f64,
    score: &str,
    solver: &str,
    visits: Option<Vec<usize>>,
    tolerance: f64,
    time_rule: &str,
) -> PyResult<Vec<f64>> {
    let ev = Evolution::Linear(generator(&rates)?);
    if visits.as_ref().is_some_and(|v| v.len() != p0.len()) {
        return Err(PyValueError::new_err("one visit count per state"));
    }
    let mut s = SolverRegistry::default()
        .create(solver, &settings(tolerance, time_rule, "a")?)
        .map_err(err)?;
    let mut rng = Rng::from_seed(1);
    scores(
        s.as_mut(),
        &ev,
        &p0,
        t,
        Score::parse(score).map_err(err)?,
        visits.as_deref(),
        &mut rng,
    )
    .map_err(err)
}

/// The derivatives of the yields of the network file `network` (`reaction_network.json` of
/// `run_mapper`, its run directory, or MultiOptPy's network) at `times` and at `t = inf`, with
/// respect to the energy of every node (1/Hartree), `ln k` of every step, `T` (1/K), and
/// `ln [S]` and `G°` of every species held at a concentration (`concentrations`, mol/L;
/// bimolecular edges and species from the annotations file `annotations`).
#[pyfunction]
#[pyo3(signature = (network, times, *, temperature = 300.0, solver = "propagate", energy_type = "electronic", start = None, level = 1, tolerance = 1e-8, time_rule = "gershgorin", rate_theory = "eyring", corrections = Vec::new(), annotations = None, concentrations = Vec::new(), association = "none", pressure = 101_325.0, standard_concentration = None))]
#[allow(clippy::too_many_arguments)]
fn kinetics_network_derivatives<'py>(
    py: Python<'py>,
    network: &str,
    times: Vec<f64>,
    temperature: f64,
    solver: &str,
    energy_type: &str,
    start: Option<usize>,
    level: u64,
    tolerance: f64,
    time_rule: &str,
    rate_theory: &str,
    corrections: Vec<String>,
    annotations: Option<&str>,
    concentrations: Vec<(String, f64)>,
    association: &str,
    pressure: f64,
    standard_concentration: Option<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let energy = EnergyKind::parse(energy_type).map_err(err)?;
    let read = annotated(network, annotations, level, energy)?;
    let kinetic = KineticsSettings {
        conditions: Conditions {
            temperature,
            energy,
            pressure,
            concentration: standard_concentration,
            ..Conditions::default()
        },
        rate_theory: rate_theory.to_string(),
        corrections,
        solver: solver.to_string(),
        solver_settings: settings(tolerance, time_rule, "a")?,
        species: species(concentrations, association)?,
        ..KineticsSettings::default()
    };
    let start = start
        .or(read.default_start)
        .ok_or_else(|| PyValueError::new_err("the network has no state to start from"))?;
    let d = derivatives(
        &read.network,
        &kinetic,
        &KineticsParts::default(),
        &[(start, 1.0)],
        &times,
    )
    .map_err(err)?;
    let out = PyDict::new(py);
    out.set_item("solver", d.solver.clone())?;
    out.set_item("times", d.times.clone())?;
    out.set_item("labels", d.labels.clone())?;
    out.set_item("populations", d.populations.clone())?;
    out.set_item("final_populations", d.final_populations.clone())?;
    out.set_item(
        "parameters",
        d.parameters
            .iter()
            .map(|p| {
                (
                    p.parameter.kind().to_string(),
                    p.label.clone(),
                    p.value,
                    p.parameter.unit().to_string(),
                )
            })
            .collect::<Vec<_>>(),
    )?;
    out.set_item("d_populations", d.d_populations.clone())?;
    out.set_item("d_final", d.d_final.clone())?;
    out.set_item("kt_hartree", d.kt)?;
    out.set_item("passes", d.passes)?;
    Ok(out)
}

/// The concentrations (mol/L) of the states and species of the network file `network` under
/// mass action at `times` from the start node with the states together at
/// `total_concentration` and the species at `concentrations` (mol/L; 0 for one not given),
/// integrated by ROS2; with the conserved totals.
#[pyfunction]
#[pyo3(signature = (network, times, *, total_concentration, concentrations = Vec::new(), annotations = None, temperature = 300.0, energy_type = "electronic", rate_theory = "eyring", corrections = Vec::new(), start = None, level = 1, ode_tolerance = DEFAULT_ODE_TOLERANCE, association = "none", pressure = 101_325.0, standard_concentration = None, solver = "rosenbrock", contract_theta = 100.0))]
#[allow(clippy::too_many_arguments)]
fn kinetics_network_mass_action<'py>(
    py: Python<'py>,
    network: &str,
    times: Vec<f64>,
    total_concentration: f64,
    concentrations: Vec<(String, f64)>,
    annotations: Option<&str>,
    temperature: f64,
    energy_type: &str,
    rate_theory: &str,
    corrections: Vec<String>,
    start: Option<usize>,
    level: u64,
    ode_tolerance: f64,
    association: &str,
    pressure: f64,
    standard_concentration: Option<f64>,
    solver: &str,
    contract_theta: f64,
) -> PyResult<Bound<'py, PyDict>> {
    if !matches!(solver, "rosenbrock" | "contract" | "windows") {
        return Err(PyValueError::new_err(
            "mass action is solved by rosenbrock, contract, or windows",
        ));
    }
    let energy = EnergyKind::parse(energy_type).map_err(err)?;
    let read = annotated(network, annotations, level, energy)?;
    let kinetic = KineticsSettings {
        conditions: Conditions {
            temperature,
            energy,
            pressure,
            concentration: standard_concentration,
            ..Conditions::default()
        },
        rate_theory: rate_theory.to_string(),
        corrections,
        model: "mass_action".to_string(),
        solver: solver.to_string(),
        solver_settings: SolverSettings {
            ode_tolerance,
            contract_theta,
            ..SolverSettings::default()
        },
        species: species(concentrations, association)?,
        model_settings: ModelSettings {
            total_concentration: Some(total_concentration),
            ..ModelSettings::default()
        },
        ..KineticsSettings::default()
    };
    let parts = KineticsParts::default();
    let prepared = prepare(&read.network, &kinetic, &parts).map_err(err)?;
    let start = start
        .or(read.default_start)
        .ok_or_else(|| PyValueError::new_err("the network has no state to start from"))?;
    let p0 = initial_populations(&read.network, &prepared.rates, &[(start, 1.0)]).map_err(err)?;
    let mut solver = parts
        .solvers
        .create(&kinetic.solver, &kinetic.solver_settings)
        .map_err(err)?;
    let mut rng = Rng::from_seed(1);
    let sol = solver
        .solve(&prepared.evolution, &p0, &times, &mut rng)
        .map_err(err)?;
    let Evolution::MassAction(sys) = &prepared.evolution else {
        return Err(PyValueError::new_err(
            "mass action: not a model of concentrations",
        ));
    };
    let out = PyDict::new(py);
    out.set_item("labels", sys.labels.clone())?;
    out.set_item("states", sys.n_states)?;
    out.set_item("times", sol.times.clone())?;
    out.set_item("concentrations", sol.populations.clone())?;
    out.set_item("initial", sys.initial(&p0).map_err(err)?)?;
    out.set_item(
        "conservation",
        sol.diagnostics
            .iter()
            .map(|d| d.conservation)
            .collect::<Vec<_>>(),
    )?;
    out.set_item(
        "conserved",
        sys.conserved
            .iter()
            .map(|(n, _)| n.clone())
            .collect::<Vec<_>>(),
    )?;
    out.set_item("notes", sol.notes.clone())?;
    Ok(out)
}

/// `RT ln(c° R T / p)` (kcal/mol): the conversion of a free energy from the ideal gas at `p`
/// (Pa) to the standard concentration `c°` (mol/L) at `t` (K).
#[pyfunction]
#[pyo3(signature = (temperature = 298.15, pressure = 101_325.0, standard_concentration = 1.0))]
fn kinetics_standard_state_shift(
    temperature: f64,
    pressure: f64,
    standard_concentration: f64,
) -> f64 {
    standard_state_kcal_mol(temperature, pressure, standard_concentration)
}

/// The factor `kappa` of the correction `name` (`wigner`, `bell`, `eckart`, `skodje_truhlar`)
/// of a saddle point with the imaginary wavenumber `imaginary_wavenumber_cm1` and the barriers
/// `barriers_hartree` from its two minima at `temperature` (K), with its diagnostics; `None`
/// where it does not apply.
#[pyfunction]
#[pyo3(signature = (name, imaginary_wavenumber_cm1, barriers_hartree, temperature, *, bell_truncation = "auto", bell_barrier = "higher"))]
fn kinetics_tunnelling(
    name: &str,
    imaginary_wavenumber_cm1: f64,
    barriers_hartree: (f64, f64),
    temperature: f64,
    bell_truncation: &str,
    bell_barrier: &str,
) -> PyResult<Option<(f64, Vec<(String, f64)>)>> {
    let settings = RateSettings {
        bell: BellSettings {
            truncation: BellTruncation::parse(bell_truncation).map_err(err)?,
            barrier: BellBarrier::parse(bell_barrier).map_err(err)?,
        },
        ..RateSettings::default()
    };
    let correction = RateCorrectionRegistry::default()
        .create(name, &settings)
        .map_err(err)?;
    let mut b = Bottleneck::saddle(0, "TS", [0, 1], 0.0);
    b.imaginary_wavenumber = Some(-imaginary_wavenumber_cm1.abs());
    let step = StepContext {
        barriers: [barriers_hartree.0, barriers_hartree.1],
        zero_point: false,
    };
    let c = Conditions {
        temperature,
        ..Conditions::default()
    };
    let out = correction
        .ln_factor(&b, &step, &KineticNetwork::default(), &c)
        .map_err(err)?;
    Ok(out.map(|(ln_k, d)| {
        (
            ln_k.exp(),
            d.into_iter().map(|x| (x.name, x.value)).collect(),
        )
    }))
}

/// The rate constants (1/s, forward and backward) of an electron-transfer step between two
/// states `dg0` apart (Hartree) with the reorganization energy and the coupling (Hartree), in
/// the form `form` (`nonadiabatic`, `adiabatic`, or `zhu_nakamura` with the wavenumber of the
/// mode along which the states cross, 1/cm), with the free energy of its bottleneck above the
/// first state.
#[pyfunction]
#[pyo3(signature = (form, reorganization, coupling, dg0, temperature = 298.15, wavenumber = None))]
fn kinetics_transfer_rate(
    form: &str,
    reorganization: f64,
    coupling: f64,
    dg0: f64,
    temperature: f64,
    wavenumber: Option<f64>,
) -> PyResult<(f64, f64, f64)> {
    let step = TransferStep {
        reorganization,
        form: TransferForm::parse(form).map_err(err)?,
        delta_g0: None,
        wavenumber,
    };
    let b = transfer_barrier(&step, Some(coupling), [0.0, dg0], temperature, "ET").map_err(err)?;
    let kt = mopt_rs::domain::Units::BOLTZMANN_HARTREE_PER_K * temperature;
    Ok((
        (b.ln_prefactor - b.top / kt).exp(),
        (b.ln_prefactor - (b.top - dg0) / kt).exp(),
        b.top,
    ))
}

/// The factor of canonical variational TST of a path with the arc lengths, the energies
/// (Hartree), and the transverse wavenumbers at every point (cm^-1) at `temperature` (K), with
/// the arc length of the variational dividing surface and its free energy above the saddle
/// point (Hartree).
#[pyfunction]
#[pyo3(signature = (arc_length, energies, frequencies, temperature = 298.15, qrrho = "grimme", qrrho_cutoff = 100.0))]
fn kinetics_cvt_factor(
    arc_length: Vec<f64>,
    energies: Vec<f64>,
    frequencies: Vec<Vec<f64>>,
    temperature: f64,
    qrrho: &str,
    qrrho_cutoff: f64,
) -> PyResult<(f64, f64, f64)> {
    let path = CvtPath {
        arc_length,
        energies,
        frequencies: Some(frequencies),
        source: "Python".to_string(),
        ..CvtPath::default()
    };
    let settings = QrrhoSettings {
        method: mopt_rs::vib::QuasiHarmonic::parse(qrrho).map_err(err)?,
        cutoff: qrrho_cutoff,
        ..QrrhoSettings::default()
    };
    let c = Conditions {
        temperature,
        ..Conditions::default()
    };
    let (ln_f, at, shift) = cvt_ln_factor(&path, &settings, &c, "path").map_err(err)?;
    Ok((ln_f.exp(), at, shift))
}

/// The names of the solvers.
#[pyfunction]
fn kinetics_solver_names() -> Vec<String> {
    SolverRegistry::default()
        .names()
        .into_iter()
        .map(str::to_string)
        .collect()
}

/// The names of the registered rate theories, corrections, and models.
#[pyfunction]
fn kinetics_part_names() -> (Vec<String>, Vec<String>, Vec<String>) {
    let names = |v: Vec<&str>| v.into_iter().map(str::to_string).collect::<Vec<_>>();
    (
        names(RateTheoryRegistry::default().names()),
        names(RateCorrectionRegistry::default().names()),
        names(ModelRegistry::default().names()),
    )
}

/// Kramers' transmission coefficient for the friction `gamma` and the barrier frequency
/// `omega_b` (both 1/s).
#[pyfunction]
fn kinetics_kramers(gamma: f64, omega_b: f64) -> f64 {
    kramers_ln_kappa(gamma, omega_b).exp()
}

/// Grote and Hynes' transmission coefficient for the barrier frequency `omega_b` (1/s) and a
/// friction kernel: `gamma` (1/s) with the memory time `memory` (s), or a table of `times` (s)
/// and `values` (1/s^2).
#[pyfunction]
#[pyo3(signature = (omega_b, *, gamma = None, memory = None, times = None, values = None))]
fn kinetics_grote_hynes(
    omega_b: f64,
    gamma: Option<f64>,
    memory: Option<f64>,
    times: Option<Vec<f64>>,
    values: Option<Vec<f64>>,
) -> PyResult<f64> {
    let kernel = match (times, values) {
        (Some(t), Some(v)) => Some((t, v)),
        (None, None) => None,
        _ => {
            return Err(PyValueError::new_err(
                "a kernel table needs times and values",
            ))
        }
    };
    let f = Friction {
        gamma,
        memory,
        kernel,
        source: "Python".to_string(),
    };
    Ok(grote_hynes_frequency(&f, omega_b).map_err(err)? / omega_b)
}

/// The weak-coupling (Airy) factor of a crossing: the coupling (Hartree), the difference and the
/// product of the slopes (Hartree/Bohr, Hartree^2/Bohr^2), the reduced mass (amu), the
/// temperature (K), the barrier from the higher minimum (Hartree).
#[pyfunction]
fn kinetics_weak_coupling<'py>(
    py: Python<'py>,
    coupling: f64,
    gradient_difference: f64,
    slope_product: f64,
    reduced_mass: f64,
    temperature: f64,
    barrier: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let kt = mopt_rs::domain::Units::BOLTZMANN_HARTREE_PER_K * temperature;
    let w = weak_coupling(
        coupling,
        gradient_difference,
        slope_product,
        reduced_mass,
        kt,
        barrier,
    )
    .map_err(err)?;
    let d = PyDict::new(py);
    d.set_item("kappa", w.ln_kappa.exp())?;
    d.set_item("p", w.p)?;
    d.set_item("classical", w.classical)?;
    d.set_item("lz_exponent", w.lz_exponent)?;
    Ok(d)
}

/// The small- and zero-curvature transmission coefficients of an adiabatic barrier (arc length
/// in sqrt(amu) Bohr, energies in Hartree, `mu_eff / mu`) at `temperature` (K).
#[pyfunction]
fn kinetics_curvature_tunnelling(
    arc_length: Vec<f64>,
    adiabatic: Vec<f64>,
    mass_ratio: Vec<f64>,
    temperature: f64,
) -> PyResult<(f64, f64)> {
    let ones = vec![1.0; arc_length.len()];
    let sct = mopt_rs::vib::transmission(&arc_length, &adiabatic, &mass_ratio, temperature)
        .map_err(err)?;
    let zct =
        mopt_rs::vib::transmission(&arc_length, &adiabatic, &ones, temperature).map_err(err)?;
    Ok((sct, zct))
}

/// The harmonic states of `wavenumbers` (cm^-1) in `cells` cells of `cell` cm^-1 (Beyer and
/// Swinehart), from one state at zero.
#[pyfunction]
fn kinetics_beyer_swinehart(wavenumbers: Vec<f64>, cell: f64, cells: usize) -> PyResult<Vec<f64>> {
    if cell.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) || cells == 0 {
        return Err(PyValueError::new_err(
            "the cells need a positive width and number",
        ));
    }
    let mut counts = vec![0.0; cells];
    counts[0] = 1.0;
    beyer_swinehart(&mut counts, &wavenumbers, cell);
    Ok(counts)
}

/// The statistics of trajectory pairs `(speed, forward, backward)` (the ends are state numbers
/// or `None`), weighted by `flux` or `count`.
#[pyfunction]
#[pyo3(signature = (pairs, weighting = "flux"))]
fn kinetics_branching<'py>(
    py: Python<'py>,
    pairs: Vec<(f64, Option<usize>, Option<usize>)>,
    weighting: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let pairs: Vec<TrajectoryPair> = pairs
        .into_iter()
        .map(|(speed, forward, backward)| TrajectoryPair {
            speed,
            forward,
            backward,
        })
        .collect();
    let b = branching(&pairs, Weighting::parse(weighting).map_err(err)?).map_err(err)?;
    let d = PyDict::new(py);
    d.set_item("pairs", b.pairs)?;
    d.set_item("undecided", b.undecided)?;
    d.set_item("recrossing", b.recrossing)?;
    d.set_item("kappa", b.kappa)?;
    d.set_item("kappa_error", b.kappa_error)?;
    d.set_item(
        "outcomes",
        b.outcomes
            .iter()
            .map(|o| (o.from, o.to, o.fraction, o.error, o.kappa, o.pairs))
            .collect::<Vec<_>>(),
    )?;
    Ok(d)
}

/// A Markov state model of discrete trajectories (states numbered from 0) at the lag `lag`
/// (frames) of `lag_time` (s): counts, the reversible transition matrix and its stationary
/// distribution, the implied time scales, the rate matrix `ln(T) / lag_time` (`[to][from]`) and
/// its most negative off-diagonal entry over the largest rate.
#[pyfunction]
fn kinetics_msm<'py>(
    py: Python<'py>,
    dtrajs: Vec<Vec<usize>>,
    states: usize,
    lag: usize,
    lag_time: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let c = count_matrix(&dtrajs, states, lag).map_err(err)?;
    let (t, pi) = reversible_transition_matrix(&c).map_err(err)?;
    let ts = implied_timescales(&t, &pi, lag_time).map_err(err)?;
    let (k, worst) = rate_matrix(&t, &pi, lag_time).map_err(err)?;
    let d = PyDict::new(py);
    d.set_item("counts", c)?;
    d.set_item("transition", t)?;
    d.set_item("stationary", pi)?;
    d.set_item("timescales", ts)?;
    d.set_item("rates", k)?;
    d.set_item("most_negative", worst)?;
    Ok(d)
}

/// The residence times of core-set trajectories (states numbered from 0, `dt` s between
/// frames): visits, mean residence times, their second moments, jumps `[to][from]`, and the
/// Markov limit `P / tau`.
#[pyfunction]
fn kinetics_dwell<'py>(
    py: Python<'py>,
    dtrajs: Vec<Vec<usize>>,
    states: usize,
    dt: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let s = dwell_statistics(&dtrajs, states, dt).map_err(err)?;
    let d = PyDict::new(py);
    d.set_item("visits", s.visits.clone())?;
    d.set_item("mean", s.mean.clone())?;
    d.set_item("second_moment", s.second_moment.clone())?;
    d.set_item("jumps", s.jumps.clone())?;
    d.set_item("markov_limit", s.markov_limit())?;
    Ok(d)
}

fn mat(rows: &[Vec<f64>]) -> PyResult<faer::Mat<f64>> {
    let n = rows.len();
    if rows.iter().any(|r| r.len() != n) {
        return Err(PyValueError::new_err("the matrix must be square"));
    }
    Ok(faer::Mat::from_fn(n, n, |i, j| rows[i][j]))
}

fn rows(m: &faer::Mat<f64>) -> Vec<Vec<f64>> {
    (0..m.nrows())
        .map(|i| (0..m.ncols()).map(|j| m[(i, j)]).collect())
        .collect()
}

/// The contraction of the kernel `K0 + s K1` (with their diagonals) with the states
/// `eliminated` taken out: the kept states, `D(0)`, `D'(0)`, and `G = D(0) (I - D'(0))^-1`.
#[pyfunction]
fn kinetics_kernel_contraction<'py>(
    py: Python<'py>,
    k0: Vec<Vec<f64>>,
    k1: Vec<Vec<f64>>,
    eliminated: Vec<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    let kc = kernel_contraction(&mat(&k0)?, &mat(&k1)?, &eliminated).map_err(err)?;
    let d = PyDict::new(py);
    d.set_item("kept", kc.kept.clone())?;
    d.set_item("d0", rows(&kc.d0))?;
    d.set_item("d1", rows(&kc.d1))?;
    d.set_item("g", rows(&kc.g))?;
    Ok(d)
}

/// The settings of a network model of these functions.
#[allow(clippy::too_many_arguments)]
fn model_settings(
    temperature: f64,
    pressure: f64,
    rate_theory: &str,
    corrections: Vec<String>,
    model: &str,
    solver: &str,
    egme: (
        f64,
        f64,
        &str,
        Option<f64>,
        Option<f64>,
        Option<&str>,
        Option<(f64, f64)>,
    ),
    cooling: Option<f64>,
    seed_trajectories: usize,
) -> PyResult<KineticsSettings> {
    let (grain, above, collision, transfer, frequency, bath, lennard_jones) = egme;
    Ok(KineticsSettings {
        conditions: Conditions {
            temperature,
            pressure,
            ..Conditions::default()
        },
        rate_theory: rate_theory.to_string(),
        corrections,
        model: model.to_string(),
        solver: solver.to_string(),
        rate: RateSettings {
            rrkm: RrkmSettings::default(),
            ..RateSettings::default()
        },
        solver_settings: SolverSettings {
            kmc_trajectories: seed_trajectories,
            ..SolverSettings::default()
        },
        model_settings: ModelSettings {
            egme: EgmeSettings {
                grain,
                above_kt: above,
                collision: if model == "egme" {
                    Some(CollisionModel::parse(collision, transfer).map_err(err)?)
                } else {
                    None
                },
                frequency,
                bath: bath.map(Bath::parse).transpose().map_err(err)?,
                lennard_jones,
            },
            hot: HotSettings {
                cooling,
                ..HotSettings::default()
            },
            ..ModelSettings::default()
        },
        ..KineticsSettings::default()
    })
}

/// The populations at `times` of the network file `network` under the model `model`
/// (`canonical`, `egme`, `two_grain`, `semi_markov`), folded onto its states, and the
/// phenomenological rate constants of `egme` (by contraction, closures A and B, and by the
/// eigenpairs, with the high-pressure ones).
#[pyfunction]
#[pyo3(signature = (network, times, *, model = "egme", temperature = 300.0, pressure = 101_325.0, rate_theory = "rrkm", corrections = Vec::new(), solver = "propagate", annotations = None, start = None, level = 1, grain = 50.0, above = 40.0, collision = "exponential_down", energy_transfer = None, collision_frequency = None, bath = None, lennard_jones = None, cooling_rate = None, trajectories = 10_000, seed = 42))]
#[allow(clippy::too_many_arguments)]
fn kinetics_network_model<'py>(
    py: Python<'py>,
    network: &str,
    times: Vec<f64>,
    model: &str,
    temperature: f64,
    pressure: f64,
    rate_theory: &str,
    corrections: Vec<String>,
    solver: &str,
    annotations: Option<&str>,
    start: Option<usize>,
    level: u64,
    grain: f64,
    above: f64,
    collision: &str,
    energy_transfer: Option<f64>,
    collision_frequency: Option<f64>,
    bath: Option<&str>,
    lennard_jones: Option<(f64, f64)>,
    cooling_rate: Option<f64>,
    trajectories: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyDict>> {
    let read = annotated(network, annotations, level, EnergyKind::Electronic)?;
    let s = model_settings(
        temperature,
        pressure,
        rate_theory,
        corrections,
        model,
        solver,
        (
            grain,
            above,
            collision,
            energy_transfer,
            collision_frequency,
            bath,
            lennard_jones,
        ),
        cooling_rate,
        trajectories,
    )?;
    let parts = KineticsParts::default();
    let prepared = prepare(&read.network, &s, &parts).map_err(err)?;
    let start = start
        .or(read.default_start)
        .ok_or_else(|| PyValueError::new_err("the network has no state to start from"))?;
    let p0 = initial_populations(&read.network, &prepared.rates, &[(start, 1.0)]).map_err(err)?;
    let ev = &prepared.evolution;
    let mut solver = parts
        .solvers
        .create(&s.solver, &s.solver_settings)
        .map_err(err)?;
    let mut rng = Rng::from_seed(seed);
    let sol = solver
        .solve(ev, &ev.expand(&p0), &times, &mut rng)
        .map_err(err)?;
    let out = PyDict::new(py);
    out.set_item("labels", prepared.rates.labels.clone())?;
    out.set_item("times", sol.times.clone())?;
    out.set_item(
        "populations",
        sol.populations
            .iter()
            .map(|p| ev.fold(p))
            .collect::<Vec<_>>(),
    )?;
    out.set_item("notes", sol.notes.clone())?;
    if let Evolution::Resolved(r) = ev {
        out.set_item("fine_states", r.generator.n)?;
        out.set_item("model_notes", r.notes.clone())?;
        if let Some(ph) = &r.phenomenological {
            out.set_item("contraction_a", rows(&ph.contraction_a))?;
            out.set_item("contraction_b", rows(&ph.contraction_b))?;
            out.set_item("eigen", ph.eigen.as_ref().map(rows))?;
            out.set_item("eigen_note", ph.eigen_note.clone())?;
            out.set_item("eigenvalues", ph.eigenvalues.clone())?;
            out.set_item("gap", ph.gap)?;
            out.set_item("high_pressure", rows(&ph.high_pressure))?;
        }
    }
    if let Evolution::SemiMarkov(sm) = ev {
        out.set_item("shapes", sm.shapes.clone())?;
        out.set_item("memory", sm.memory.as_ref().map(rows))?;
    }
    Ok(out)
}

/// The refinement tasks of the network file `network` for the populations of the nodes
/// `targets` at the last of `times`, with the standard deviation `sigma` (kcal/mol) of the
/// energies without their own: `(observable, task, target, parameter, sensitivity, sigma,
/// score)`, highest score first for every observable.
#[pyfunction]
#[pyo3(signature = (network, times, targets, *, sigma = 1.0, temperature = 300.0, solver = "propagate", rate_theory = "eyring", corrections = Vec::new(), energy_type = "electronic", annotations = None, start = None, level = 1))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn kinetics_network_refinement(
    network: &str,
    times: Vec<f64>,
    targets: Vec<usize>,
    sigma: f64,
    temperature: f64,
    solver: &str,
    rate_theory: &str,
    corrections: Vec<String>,
    energy_type: &str,
    annotations: Option<&str>,
    start: Option<usize>,
    level: u64,
) -> PyResult<Vec<(String, String, String, String, f64, f64, f64)>> {
    let energy = EnergyKind::parse(energy_type).map_err(err)?;
    let read = annotated(network, annotations, level, energy)?;
    let s = KineticsSettings {
        conditions: Conditions {
            temperature,
            energy,
            ..Conditions::default()
        },
        rate_theory: rate_theory.to_string(),
        corrections,
        solver: solver.to_string(),
        ..KineticsSettings::default()
    };
    let parts = KineticsParts::default();
    let start = start
        .or(read.default_start)
        .ok_or_else(|| PyValueError::new_err("the network has no state to start from"))?;
    let d = derivatives(&read.network, &s, &parts, &[(start, 1.0)], &times).map_err(err)?;
    let prepared = prepare(&read.network, &s, &parts).map_err(err)?;
    let last = times.len().saturating_sub(1);
    let mut out = Vec::new();
    for id in targets {
        let i = read
            .network
            .index_of(id)
            .and_then(|k| prepared.rates.index_of(k))
            .ok_or_else(|| {
                PyValueError::new_err(format!("node {id} is not a state of the model"))
            })?;
        let tasks = refinement_tasks(
            &d,
            &read.network,
            Observable {
                state: i,
                time: Some(last),
            },
            &Uncertainties {
                energy: sigma * mopt_rs::domain::Units::KCAL_MOL_TO_HARTREE,
            },
        )
        .map_err(err)?;
        for t in tasks {
            out.push((
                prepared.rates.labels[i].clone(),
                t.kind.name().to_string(),
                t.target,
                t.parameter,
                t.sensitivity,
                t.sigma,
                t.score,
            ));
        }
    }
    Ok(out)
}

/// Register the functions of this module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(kinetics_part_names, m)?)?;
    m.add_function(wrap_pyfunction!(kinetics_kramers, m)?)?;
    m.add_function(wrap_pyfunction!(kinetics_grote_hynes, m)?)?;
    m.add_function(wrap_pyfunction!(kinetics_weak_coupling, m)?)?;
    m.add_function(wrap_pyfunction!(kinetics_curvature_tunnelling, m)?)?;
    m.add_function(wrap_pyfunction!(kinetics_beyer_swinehart, m)?)?;
    m.add_function(wrap_pyfunction!(kinetics_branching, m)?)?;
    m.add_function(wrap_pyfunction!(kinetics_msm, m)?)?;
    m.add_function(wrap_pyfunction!(kinetics_dwell, m)?)?;
    m.add_function(wrap_pyfunction!(kinetics_kernel_contraction, m)?)?;
    m.add_function(wrap_pyfunction!(kinetics_network_model, m)?)?;
    m.add_function(wrap_pyfunction!(kinetics_network_refinement, m)?)?;
    m.add_function(wrap_pyfunction!(kinetics_solve, m)?)?;
    m.add_function(wrap_pyfunction!(kinetics_contract, m)?)?;
    m.add_function(wrap_pyfunction!(kinetics_greedy_order, m)?)?;
    m.add_function(wrap_pyfunction!(kinetics_scores, m)?)?;
    m.add_function(wrap_pyfunction!(kinetics_solver_names, m)?)?;
    m.add_function(wrap_pyfunction!(kinetics_network_derivatives, m)?)?;
    m.add_function(wrap_pyfunction!(kinetics_network_mass_action, m)?)?;
    m.add_function(wrap_pyfunction!(kinetics_standard_state_shift, m)?)?;
    m.add_function(wrap_pyfunction!(kinetics_tunnelling, m)?)?;
    m.add_function(wrap_pyfunction!(kinetics_transfer_rate, m)?)?;
    m.add_function(wrap_pyfunction!(kinetics_cvt_factor, m)?)?;
    Ok(())
}
