"""The kinetics of reaction networks from Python.

The functions take a rate matrix ``K[i][j]`` (1/s, the rate constant from state ``j`` to
state ``i``; the diagonal is ignored and rebuilt from the columns), as a NumPy array or nested
lists, and use the solvers of the Rust side: :func:`solve` (``propagate``, ``rcmc``,
``legacy_rcmc``, ``expm``, ``spectral``, ``steady``), :func:`contract` (the blocks of the
subtraction-free contraction with given eliminated states), :func:`greedy_order` (the order of
RCMC with its reference times), and :func:`scores` (the kinetic priorities of an exploration).
Detailed balance is detected from the matrix. :func:`derivatives` differentiates the yields of
a network file with respect to the energies of its nodes, ``ln k`` of its steps, the
temperature, and ``ln [S]`` of the species of its bimolecular steps (the apparent reaction
orders; dual numbers through the solver). :func:`mass_action` integrates the concentrations of
a network with bimolecular steps under mass action (ROS2), and :func:`standard_state_shift`
gives the conversion ``RT ln(c° R T / p)`` of a free energy from the gas to a concentration.
:func:`run_kinetics` runs the command ``run_kinetics`` on a network file
(``reaction_network.json`` of ``run_mapper`` or MultiOptPy's network) with its options and
writes its files.

Beyond the canonical model: :func:`network_model` (the models ``egme`` with its
phenomenological rate constants, ``two_grain``, ``semi_markov`` of a network file),
:func:`refinement_tasks` (ranked by sensitivity times uncertainty), the single corrections
:func:`kramers`, :func:`grote_hynes`, :func:`weak_coupling`, :func:`curvature_tunnelling`, the
direct count :func:`beyer_swinehart`, trajectories from a saddle point (:func:`saddle_trajectories`
with :func:`branching`), Markov state models (:func:`markov_state_model`, :func:`dwell_times`),
and the contraction of memory kernels (:func:`kernel_contraction`).

MultiOptPy: `Utils/rcmc.py` (``RCMCQueue``, ``main``).

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from mopt_rs import _core

FloatArray = npt.NDArray[np.float64]

__all__ = [
    "Branching",
    "Contraction",
    "KineticSolution",
    "MassActionSolution",
    "YieldDerivatives",
    "beyer_swinehart",
    "branching",
    "contract",
    "curvature_tunnelling",
    "cvt_factor",
    "derivatives",
    "dwell_times",
    "greedy_order",
    "grote_hynes",
    "kernel_contraction",
    "kramers",
    "markov_state_model",
    "mass_action",
    "network_model",
    "part_names",
    "refinement_tasks",
    "run_kinetics",
    "saddle_trajectories",
    "scores",
    "solve",
    "solver_names",
    "standard_state_shift",
    "transfer_rate",
    "tunnelling",
    "weak_coupling",
]


def _matrix(rates: npt.ArrayLike) -> list[list[float]]:
    k = np.asarray(rates, dtype=float)
    if k.ndim != 2 or k.shape[0] != k.shape[1]:
        raise ValueError("the rate matrix must be square")
    rows: list[list[float]] = k.tolist()
    return rows


@dataclass(frozen=True)
class KineticSolution:
    """Populations at the output times, with the diagnostics of every time."""

    #: The solver.
    solver: str
    #: The output times (s).
    times: FloatArray
    #: ``populations[k, i]``: state ``i`` at ``times[k]``.
    populations: FloatArray
    #: States eliminated at every time.
    eliminated: list[int]
    #: The separation ratio of the cut (``None`` without a contraction).
    eps_hat: list[float | None]
    #: ``|1^T p(t) - 1^T p(0)|``.
    conservation: list[float]
    #: The members of every superstate (``None`` without a contraction).
    superstates: list[list[list[int]] | None]
    #: Whether detailed balance was detected.
    detailed_balance: bool


def solve(
    rates: npt.ArrayLike,
    p0: Sequence[float],
    times: Sequence[float],
    *,
    solver: str = "propagate",
    tolerance: float = 1e-8,
    time_rule: str = "gershgorin",
    closure: str = "a",
    trajectories: int = 10000,
    seed: int = 1,
) -> KineticSolution:
    """The populations at ``times`` from ``p0`` (``trajectories`` and ``seed`` are those of the
    solver ``kmc``)."""
    d = _core.kinetics_solve(
        _matrix(rates),
        [float(x) for x in p0],
        [float(t) for t in times],
        solver=solver,
        tolerance=tolerance,
        time_rule=time_rule,
        closure=closure,
        trajectories=int(trajectories),
        seed=int(seed),
    )
    return KineticSolution(
        solver=d["solver"],
        times=np.asarray(d["times"], dtype=float),
        populations=np.asarray(d["populations"], dtype=float),
        eliminated=list(d["eliminated"]),
        eps_hat=list(d["eps_hat"]),
        conservation=list(d["conservation"]),
        superstates=list(d["superstates"]),
        detailed_balance=bool(d["detailed_balance"]),
    )


@dataclass(frozen=True)
class Contraction:
    """The blocks of a contraction (``kinetics_beyond_tst.md``, section 3.4)."""

    #: The kept states, ascending.
    kept: list[int]
    #: The eliminated states, in the order of elimination.
    eliminated: list[int]
    #: The Schur complement ``D`` (``|T| x |T|``).
    d: FloatArray
    #: ``B = -K_TS K_SS^-1`` (``|T| x |S|``).
    b: FloatArray
    #: ``C = -K_SS^-1 K_ST`` (``|S| x |T|``).
    c: FloatArray
    #: ``1^T M``.
    weights: FloatArray

    def lump(self, p: Sequence[float]) -> FloatArray:
        """``q = p_T + B p_S``."""
        x = np.asarray(p, dtype=float)
        out: FloatArray = x[self.kept] + self.b @ x[self.eliminated]
        return out

    def generator(self) -> FloatArray:
        """The superstate generator of closure A, ``D diag(1^T M)^-1``."""
        out: FloatArray = self.d / self.weights[np.newaxis, :]
        return out


def contract(rates: npt.ArrayLike, eliminate: Sequence[int]) -> Contraction:
    """The contraction of ``rates`` with the states ``eliminate`` (in that order)."""
    d = _core.kinetics_contract(_matrix(rates), [int(i) for i in eliminate])
    nt, ns = len(d["kept"]), len(d["eliminated"])
    return Contraction(
        kept=list(d["kept"]),
        eliminated=list(d["eliminated"]),
        d=np.asarray(d["d"], dtype=float).reshape(nt, nt),
        b=np.asarray(d["b"], dtype=float).reshape(nt, ns),
        c=np.asarray(d["c"], dtype=float).reshape(ns, nt),
        weights=np.asarray(d["weights"], dtype=float),
    )


def greedy_order(
    rates: npt.ArrayLike, *, time_rule: str = "gershgorin"
) -> list[tuple[int, float, float, float, float, float]]:
    """The steps of RCMC: (state, ``|D_jj|``, ``sigma_hat``, ``rho_hat``, reference time,
    ``eps_hat``)."""
    return list(_core.kinetics_greedy_order(_matrix(rates), time_rule=time_rule))


def scores(
    rates: npt.ArrayLike,
    p0: Sequence[float],
    t: float,
    *,
    score: str = "population",
    solver: str = "propagate",
    visits: Sequence[int] | None = None,
    tolerance: float = 1e-8,
    time_rule: str = "gershgorin",
) -> FloatArray:
    """The kinetic scores of the states at the horizon ``t``: ``population``,
    ``max_population``, ``residence``, ``discovery_flux`` (with ``visits``)."""
    out = _core.kinetics_scores(
        _matrix(rates),
        [float(x) for x in p0],
        float(t),
        score=score,
        solver=solver,
        visits=None if visits is None else [int(v) for v in visits],
        tolerance=tolerance,
        time_rule=time_rule,
    )
    return np.asarray(out, dtype=float)


@dataclass(frozen=True)
class YieldDerivatives:
    """Derivatives of the yields (populations) of a network.

    ``d_populations[m, k, i] = dp_i(times[k]) / dtheta_m`` and ``d_final[m, i] = dp_i(inf) /
    dtheta_m`` for the parameters ``parameters[m] = (kind, label, value, unit)``: the energy of
    a state or of a bottleneck (``state_energy``, ``bottleneck_energy``; Hartree, derivative in
    1/Hartree), ``ln k`` of a step with both directions scaled together (``ln_k``; the degree
    of rate control is ``d ln p / d ln k``), the temperature (``temperature``; 1/K), and for
    every species held at a concentration ``ln [S]`` (``ln_concentration``; the apparent
    reaction order is ``d ln p / d ln [S]``) and its standard free energy (``species_energy``;
    1/Hartree).
    """

    #: The solver.
    solver: str
    #: The output times (s).
    times: FloatArray
    #: Labels of the states.
    labels: list[str]
    #: ``populations[k, i]``.
    populations: FloatArray
    #: The final populations (``t -> inf``).
    final_populations: FloatArray
    #: ``(kind, label, value, unit)`` of every parameter.
    parameters: list[tuple[str, str, float, str]]
    #: ``d_populations[m, k, i]``.
    d_populations: FloatArray
    #: ``d_final[m, i]``.
    d_final: FloatArray
    #: ``k_B T`` (Hartree).
    kt_hartree: float

    def rate_control(self, state: str, k: int = -1) -> dict[str, float]:
        """``d ln p / d ln k`` of every step for the yield of ``state`` at ``times[k]``."""
        i = self.labels.index(state)
        p = self.populations[k, i]
        return {
            label: float(self.d_populations[m, k, i] / p)
            for m, (kind, label, _, _) in enumerate(self.parameters)
            if kind == "ln_k"
        }

    def apparent_orders(self, state: str, k: int = -1) -> dict[str, float]:
        """``d ln p / d ln [S]`` of every species for the yield of ``state`` at ``times[k]``."""
        i = self.labels.index(state)
        p = self.populations[k, i]
        return {
            label: float(self.d_populations[m, k, i] / p)
            for m, (kind, label, _, _) in enumerate(self.parameters)
            if kind == "ln_concentration"
        }


def _concentrations(c: Mapping[str, float] | None) -> list[tuple[str, float]]:
    return [] if c is None else [(str(k), float(v)) for k, v in c.items()]


def derivatives(
    network: str,
    times: Sequence[float],
    *,
    temperature: float = 300.0,
    solver: str = "propagate",
    energy_type: str = "electronic",
    start: int | None = None,
    level: int = 1,
    tolerance: float = 1e-8,
    time_rule: str = "gershgorin",
    rate_theory: str = "eyring",
    corrections: Sequence[str] = (),
    annotations: str | None = None,
    concentrations: Mapping[str, float] | None = None,
    association: str = "none",
    pressure: float = 101325.0,
    standard_concentration: float | None = None,
) -> YieldDerivatives:
    """The derivatives of the yields of the network file ``network`` at ``times`` and at
    ``t -> inf`` by forward-mode dual numbers through ``solver`` (``propagate``, ``rcmc``,
    ``expm``, ``steady``). Bimolecular steps and their species come from the annotations
    file ``annotations``; ``concentrations`` (mol/L) holds the species fixed (pseudo-first
    order)."""
    d = _core.kinetics_network_derivatives(
        network,
        [float(t) for t in times],
        temperature=temperature,
        solver=solver,
        energy_type=energy_type,
        start=start,
        level=level,
        tolerance=tolerance,
        time_rule=time_rule,
        rate_theory=rate_theory,
        corrections=list(corrections),
        annotations=annotations,
        concentrations=_concentrations(concentrations),
        association=association,
        pressure=pressure,
        standard_concentration=standard_concentration,
    )
    return YieldDerivatives(
        solver=d["solver"],
        times=np.asarray(d["times"], dtype=float),
        labels=list(d["labels"]),
        populations=np.asarray(d["populations"], dtype=float),
        final_populations=np.asarray(d["final_populations"], dtype=float),
        parameters=[tuple(p) for p in d["parameters"]],
        d_populations=np.asarray(d["d_populations"], dtype=float),
        d_final=np.asarray(d["d_final"], dtype=float),
        kt_hartree=float(d["kt_hartree"]),
    )


@dataclass(frozen=True)
class MassActionSolution:
    """Concentrations (mol/L) of the states and species of a network under mass action."""

    #: The states of the model, then the species.
    labels: list[str]
    #: How many of the labels are states.
    states: int
    #: The output times (s).
    times: FloatArray
    #: ``concentrations[k, i]``: entity ``i`` at ``times[k]``.
    concentrations: FloatArray
    #: The initial concentrations.
    initial: FloatArray
    #: The largest change of a conserved total at every time (relative).
    conservation: list[float]
    #: The conserved totals.
    conserved: list[str]
    #: What the integrator says.
    notes: list[str]

    def final_yields(self) -> dict[str, float]:
        """The concentrations at the last time over the initial concentration of the states
        together (states) or over their own initial concentration (species given one)."""
        total = float(self.initial[: self.states].sum())
        last = self.concentrations[-1]
        out: dict[str, float] = {}
        for i, label in enumerate(self.labels):
            start = float(self.initial[i])
            base = total if i < self.states or start <= 0.0 else start
            out[label] = float(last[i] / base)
        return out


def mass_action(
    network: str,
    times: Sequence[float],
    *,
    total_concentration: float,
    concentrations: Mapping[str, float] | None = None,
    annotations: str | None = None,
    temperature: float = 300.0,
    energy_type: str = "electronic",
    rate_theory: str = "eyring",
    corrections: Sequence[str] = (),
    start: int | None = None,
    level: int = 1,
    ode_tolerance: float = 1e-7,
    association: str = "none",
    pressure: float = 101325.0,
    standard_concentration: float | None = None,
    solver: str = "rosenbrock",
    contract_theta: float = 100.0,
) -> MassActionSolution:
    """The concentrations of the network file ``network`` under mass action at ``times``:
    the states together at ``total_concentration`` (mol/L) in the start node, the species at
    ``concentrations`` (mol/L, 0 for one not given), the bimolecular steps and species from
    ``annotations``; integrated by ROS2 (``solver="rosenbrock"``), contracted with the fast
    species on their quasi-steady manifold (``solver="contract"``, RCMC's extension to mass
    action; ``contract_theta`` the threshold of ``rate * t``), or by pseudo-first-order windows
    (``solver="windows"``)."""
    d = _core.kinetics_network_mass_action(
        network,
        [float(t) for t in times],
        total_concentration=float(total_concentration),
        concentrations=_concentrations(concentrations),
        annotations=annotations,
        temperature=temperature,
        energy_type=energy_type,
        rate_theory=rate_theory,
        corrections=list(corrections),
        start=start,
        level=level,
        ode_tolerance=ode_tolerance,
        association=association,
        pressure=pressure,
        standard_concentration=standard_concentration,
        solver=solver,
        contract_theta=float(contract_theta),
    )
    return MassActionSolution(
        labels=list(d["labels"]),
        states=int(d["states"]),
        times=np.asarray(d["times"], dtype=float),
        concentrations=np.asarray(d["concentrations"], dtype=float),
        initial=np.asarray(d["initial"], dtype=float),
        conservation=list(d["conservation"]),
        conserved=list(d["conserved"]),
        notes=list(d["notes"]),
    )


def standard_state_shift(
    temperature: float = 298.15,
    pressure: float = 101325.0,
    standard_concentration: float = 1.0,
) -> float:
    """``RT ln(c° R T / p)`` (kcal/mol per molecule): the conversion of a free energy from the
    ideal gas at ``pressure`` (Pa) to ``standard_concentration`` (mol/L), 1.894 kcal/mol at
    298.15 K and 1 atm."""
    return float(
        _core.kinetics_standard_state_shift(
            temperature=temperature,
            pressure=pressure,
            standard_concentration=standard_concentration,
        )
    )


def tunnelling(
    name: str,
    imaginary_wavenumber_cm1: float,
    barriers_hartree: tuple[float, float],
    temperature: float,
    *,
    bell_truncation: str = "auto",
    bell_barrier: str = "higher",
) -> tuple[float, dict[str, float]] | None:
    """The factor ``kappa`` of the tunnelling correction ``name`` (``wigner``, ``bell``,
    ``eckart``, ``skodje_truhlar``) of a saddle point with the imaginary wavenumber and the
    barriers from its two minima (Hartree) at ``temperature`` (K), with its diagnostics
    (``u``, ``T/Tc``, ...); ``None`` where it does not apply."""
    out = _core.kinetics_tunnelling(
        name,
        float(imaginary_wavenumber_cm1),
        (float(barriers_hartree[0]), float(barriers_hartree[1])),
        float(temperature),
        bell_truncation=bell_truncation,
        bell_barrier=bell_barrier,
    )
    if out is None:
        return None
    kappa, diagnostics = out
    return float(kappa), {str(k): float(v) for k, v in diagnostics}


def transfer_rate(
    form: str,
    reorganization: float,
    coupling: float,
    dg0: float,
    temperature: float = 298.15,
    wavenumber: float | None = None,
) -> tuple[float, float, float]:
    """The rate constants (1/s, forward and backward) of an electron transfer between two
    states ``dg0`` apart (Hartree) with the reorganization energy and the coupling (Hartree):
    ``nonadiabatic`` (Marcus-Levich), ``adiabatic`` (TST on the two-state surface), or
    ``zhu_nakamura`` (the thermal Zhu-Nakamura transmission of the crossing along a mode of the
    wavenumber ``wavenumber``, 1/cm), with the free energy of its bottleneck above the first
    state (Hartree)."""
    k_f, k_b, top = _core.kinetics_transfer_rate(
        form,
        float(reorganization),
        float(coupling),
        float(dg0),
        float(temperature),
        None if wavenumber is None else float(wavenumber),
    )
    return float(k_f), float(k_b), float(top)


def cvt_factor(
    arc_length: Sequence[float],
    energies: Sequence[float],
    frequencies: Sequence[Sequence[float]],
    temperature: float = 298.15,
    *,
    qrrho: str = "grimme",
    qrrho_cutoff: float = 100.0,
) -> tuple[float, float, float]:
    """The factor of canonical variational TST of a path (arc lengths, energies in Hartree,
    transverse wavenumbers in cm^-1 at every point) at ``temperature``, with the arc length of
    the variational dividing surface and its free energy above the saddle point (Hartree)."""
    f, at, shift = _core.kinetics_cvt_factor(
        [float(x) for x in arc_length],
        [float(x) for x in energies],
        [[float(w) for w in row] for row in frequencies],
        float(temperature),
        qrrho,
        float(qrrho_cutoff),
    )
    return float(f), float(at), float(shift)


def solver_names() -> list[str]:
    """The registered solvers."""
    return list(_core.kinetics_solver_names())


def part_names() -> dict[str, list[str]]:
    """The registered rate theories, corrections, models, and solvers."""
    theories, corrections, models = _core.kinetics_part_names()
    return {
        "theories": list(theories),
        "corrections": list(corrections),
        "models": list(models),
        "solvers": solver_names(),
    }


def kramers(gamma: float, omega_b: float) -> float:
    """Kramers' transmission coefficient for the friction ``gamma`` and the angular barrier
    frequency ``omega_b`` (both 1/s)."""
    return float(_core.kinetics_kramers(float(gamma), float(omega_b)))


def grote_hynes(
    omega_b: float,
    *,
    gamma: float | None = None,
    memory: float | None = None,
    times: Sequence[float] | None = None,
    values: Sequence[float] | None = None,
) -> float:
    """Grote and Hynes' transmission coefficient for ``omega_b`` (1/s) and a friction kernel:
    ``gamma`` (1/s) with the exponential memory time ``memory`` (s), or a table ``times`` (s),
    ``values`` (1/s^2)."""
    return float(
        _core.kinetics_grote_hynes(
            float(omega_b),
            gamma=gamma,
            memory=memory,
            times=None if times is None else [float(x) for x in times],
            values=None if values is None else [float(x) for x in values],
        )
    )


def weak_coupling(
    coupling: float,
    gradient_difference: float,
    slope_product: float,
    reduced_mass: float,
    temperature: float,
    barrier: float,
) -> dict[str, float]:
    """The weak-coupling (Airy) factor of a crossing with tunnelling below it: ``kappa``, the
    Airy parameter ``p``, the classical weak-coupling limit, and the Landau-Zener exponent at
    ``k_B T`` (atomic units: Hartree, Hartree/Bohr, amu; the barrier from the higher minimum)."""
    d = _core.kinetics_weak_coupling(
        float(coupling),
        float(gradient_difference),
        float(slope_product),
        float(reduced_mass),
        float(temperature),
        float(barrier),
    )
    return {str(k): float(v) for k, v in d.items()}


def curvature_tunnelling(
    arc_length: Sequence[float],
    adiabatic: Sequence[float],
    mass_ratio: Sequence[float],
    temperature: float,
) -> tuple[float, float]:
    """The small- and zero-curvature transmission coefficients ``(sct, zct)`` of the adiabatic
    barrier of a path (``path_profile.csv`` of ``freqmain -path``)."""
    sct, zct = _core.kinetics_curvature_tunnelling(
        [float(x) for x in arc_length],
        [float(x) for x in adiabatic],
        [float(x) for x in mass_ratio],
        float(temperature),
    )
    return float(sct), float(zct)


def beyer_swinehart(wavenumbers: Sequence[float], cell: float, cells: int) -> FloatArray:
    """The harmonic states of ``wavenumbers`` (cm^-1) in ``cells`` cells of ``cell`` cm^-1."""
    return np.asarray(
        _core.kinetics_beyer_swinehart([float(w) for w in wavenumbers], float(cell), int(cells)),
        dtype=np.float64,
    )


@dataclass(frozen=True)
class Branching:
    """The statistics of trajectory pairs from a saddle point."""

    #: Pairs, undecided pairs, pairs that end where they started (recrossing).
    pairs: int
    undecided: int
    recrossing: int
    #: The transmission coefficient and its standard error.
    kappa: float
    kappa_error: float
    #: ``(from, to, fraction, error, kappa share, pairs)``, the largest fraction first.
    outcomes: list[tuple[int, int, float, float, float, int]]

    def annotations(self, edge: str, labels: Sequence[str]) -> dict[str, object]:
        """The annotations of ``run_kinetics`` for the saddle point ``edge``: the transmission
        coefficient (``reactive_flux``) and the fractions of the state the most flux comes
        from (``outcomes``), the states named by ``labels``."""
        entry: dict[str, object] = {
            "reactive_flux_kappa": self.kappa,
            "reactive_flux_error": self.kappa_error if np.isfinite(self.kappa_error) else 0.0,
        }
        totals: dict[int, float] = {}
        for o in self.outcomes:
            totals[o[0]] = totals.get(o[0], 0.0) + o[2]
        if totals:
            source = max(totals, key=lambda k: totals[k])
            chosen = [o for o in self.outcomes if o[0] == source]
            entry["outcomes"] = {
                "from": labels[source],
                "to": {labels[o[1]]: o[2] for o in chosen},
            }
            entry["outcome_errors"] = {
                labels[o[1]]: (o[3] if np.isfinite(o[3]) else 0.0) for o in chosen
            }
        return {"edges": {edge: entry}}


def branching(
    pairs: Sequence[tuple[float, int | None, int | None]], weighting: str = "flux"
) -> Branching:
    """The transmission coefficient and the branching fractions of trajectory pairs
    ``(speed along the imaginary mode, forward end, backward end)`` (ends as state numbers or
    ``None``), weighted by the flux (``flux``) or counted (``count``)."""
    d = _core.kinetics_branching(
        [(float(s), f, b) for s, f, b in pairs], weighting
    )
    return Branching(
        pairs=int(d["pairs"]),
        undecided=int(d["undecided"]),
        recrossing=int(d["recrossing"]),
        kappa=float(d["kappa"]),
        kappa_error=float(d["kappa_error"]),
        outcomes=[
            (int(a), int(b), float(f), float(e), float(k), int(n))
            for a, b, f, e, k, n in d["outcomes"]
        ],
    )


def saddle_trajectories(
    surface: object,
    x: npt.ArrayLike,
    hessian: npt.ArrayLike,
    classify: Callable[[FloatArray], int | None],
    *,
    pairs: int = 100,
    temperature: float = 298.15,
    time_step: float = 20.0,
    max_steps: int = 2000,
    check_every: int = 10,
    commit: int = 5,
    seed: int = 42,
    masses: Sequence[float] | None = None,
    symbols: Sequence[str] | None = None,
    charge: int = 0,
    multiplicity: int = 1,
) -> list[tuple[float, int | None, int | None]]:
    """Pairs of trajectories from the dividing surface of the saddle point ``x`` (with its
    ``hessian``) on ``surface`` (as :func:`mopt_rs.md.run_md` takes it): classical thermal
    samples of the transverse modes (mass-weighted with ``masses`` in atomic units; 1 for plain
    variables) and a thermal momentum along the imaginary mode, each run forwards and
    reversed without a thermostat until ``classify(x)`` names the same state ``commit`` times in
    a row (every ``check_every`` steps of ``time_step`` atomic units of time). Returns the
    pairs for :func:`branching` (the Rust driver for structures and objectives is
    ``workflow::kinetics_dynamics``)."""
    from mopt_rs.md import run_md

    x0 = np.asarray(x, dtype=float).ravel()
    n = x0.size
    m = np.ones(n) if masses is None else np.repeat(np.asarray(masses, dtype=float), n // len(masses))
    h = np.asarray(hessian, dtype=float).reshape(n, n)
    weighted = h / np.sqrt(np.outer(m, m))
    lam, u = np.linalg.eigh(0.5 * (weighted + weighted.T))
    negative = np.flatnonzero(lam < 0.0)
    if negative.size != 1:
        raise ValueError(f"a first-order saddle point has one negative curvature, not {negative.size}")
    kt = _core.BOLTZMANN_HARTREE_PER_K * float(temperature)
    rng = np.random.default_rng(seed)
    out: list[tuple[float, int | None, int | None]] = []
    for _ in range(int(pairs)):
        q = np.where(lam > 0.0, np.sqrt(kt / np.where(lam > 0.0, lam, 1.0)), 0.0) * rng.standard_normal(n)
        p = np.sqrt(kt) * rng.standard_normal(n)
        xs = x0 + (u @ q) / np.sqrt(m)
        vs = (u @ p) / np.sqrt(m)
        speed = float(p[negative[0]])
        ends: list[int | None] = []
        for sign in (1.0, -1.0):
            state: dict[str, int | None] = {"last": None, "streak": 0}

            def watch(report: dict[str, object], state: dict[str, int | None] = state) -> bool:
                step = int(report["step"])  # type: ignore[call-overload]
                if step == 0 or step % check_every:
                    return True
                here = classify(np.asarray(report["x"], dtype=float))
                if here is not None and here == state["last"]:
                    state["streak"] = (state["streak"] or 0) + 1
                else:
                    state["streak"] = 0 if here is None else 1
                    state["last"] = here
                return (state["streak"] or 0) < commit

            run_md(
                surface,  # type: ignore[arg-type]
                xs,
                symbols=symbols,
                masses=None if masses is None else list(masses),
                charge=charge,
                multiplicity=multiplicity,
                ensemble="nve",
                time_step=float(time_step),
                steps=int(max_steps),
                temperature=float(temperature),
                velocities=sign * vs,
                callback=watch,
                keep_every=max(int(max_steps), 1),
            )
            committed = (state["streak"] or 0) >= commit
            ends.append(state["last"] if committed else None)
        forward, backward = (ends[0], ends[1]) if speed >= 0.0 else (ends[1], ends[0])
        out.append((abs(speed), forward, backward))
    return out


def markov_state_model(
    dtrajs: Sequence[Sequence[int]], states: int, lag: int, lag_time: float
) -> dict[str, object]:
    """A Markov state model of discrete trajectories at the lag ``lag`` (frames) of
    ``lag_time`` (s): ``counts``, ``transition`` (reversible, ``[to][from]``), ``stationary``,
    ``timescales`` (implied, s), ``rates`` (``ln(T) / lag_time``), ``most_negative``."""
    d = _core.kinetics_msm([list(map(int, d)) for d in dtrajs], int(states), int(lag), float(lag_time))
    return {str(k): v for k, v in d.items()}


def dwell_times(dtrajs: Sequence[Sequence[int]], states: int, dt: float) -> dict[str, object]:
    """The residence times of core-set trajectories (``dt`` s between frames): ``visits``,
    ``mean``, ``second_moment``, ``jumps`` (``[to][from]``), ``markov_limit``."""
    d = _core.kinetics_dwell([list(map(int, d)) for d in dtrajs], int(states), float(dt))
    return {str(k): v for k, v in d.items()}


def kernel_contraction(
    k0: npt.ArrayLike, k1: npt.ArrayLike, eliminated: Sequence[int]
) -> dict[str, object]:
    """The contraction of the memory kernel ``K0 + s K1`` (with their diagonals) with the states
    ``eliminated``: ``kept``, ``d0``, ``d1``, and ``g = D(0) (I - D'(0))^-1``."""
    d = _core.kinetics_kernel_contraction(
        np.asarray(k0, dtype=float).tolist(),
        np.asarray(k1, dtype=float).tolist(),
        [int(i) for i in eliminated],
    )
    return {str(k): (np.asarray(v, dtype=float) if k != "kept" else list(v)) for k, v in d.items()}


def network_model(network: str, times: Sequence[float], **options: object) -> dict[str, object]:
    """The populations of the network file ``network`` at ``times`` under ``model`` (``egme``
    by default; ``canonical``, ``two_grain``, ``semi_markov``), with the phenomenological rate
    constants of ``egme`` (``contraction_a``, ``contraction_b``, ``eigen``, ``high_pressure``,
    ``gap``). Options: ``temperature``, ``pressure``, ``rate_theory`` (``rrkm``),
    ``corrections``, ``solver``, ``annotations``, ``start``, ``grain``, ``above``,
    ``collision``, ``energy_transfer``, ``collision_frequency``, ``bath``, ``lennard_jones``,
    ``cooling_rate``, ``trajectories``, ``seed``."""
    d = _core.kinetics_network_model(network, [float(t) for t in times], **options)
    return {str(k): v for k, v in d.items()}


def refinement_tasks(
    network: str, times: Sequence[float], targets: Sequence[int], **options: object
) -> list[tuple[str, str, str, str, float, float, float]]:
    """The refinement tasks ``(observable, task, target, parameter, sensitivity, sigma,
    score)`` of the network file ``network`` for the populations of ``targets`` at the last of
    ``times``, ranked by ``|d ln y / d theta| sigma`` (``sigma`` in kcal/mol for the energies
    without their own ``uncertainty_kcal_mol``)."""
    return list(
        _core.kinetics_network_refinement(
            network, [float(t) for t in times], [int(i) for i in targets], **options
        )
    )


def run_kinetics(network: str, *options: str) -> int:
    """The command ``run_kinetics NETWORK OPTIONS...``; returns its exit code."""
    return int(_core.run_cli("run_kinetics", [network, *options]))
