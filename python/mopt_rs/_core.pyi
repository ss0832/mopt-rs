# This file is part of mopt-py (the Python interface of mopt-rs).
"""Type stubs of the native module ``mopt_rs._core``."""

import os
from collections.abc import Callable
from typing import Any

__version__: str

# CODATA 2018 conversion factors (mopt_rs::domain::Units).
BOHR_TO_ANGSTROM: float
ANGSTROM_TO_BOHR: float
HARTREE_TO_EV: float
HARTREE_TO_KCAL_MOL: float
HARTREE_TO_KJ_MOL: float
HARTREE_TO_WAVENUMBER: float
BOLTZMANN_HARTREE_PER_K: float
AMU_TO_AU_MASS: float
AU_TIME_TO_FS: float
GPA_PER_AU_PRESSURE: float
# Eigenvalues of the mass-weighted Hessian below minus this are imaginary modes
# (mopt_rs::opt::mode_kill::IMAGINARY_CURVATURE).
IMAGINARY_CURVATURE: float

def version() -> str: ...
def run_cli(command: str, args: list[str]) -> int: ...
def run_cli_counted(command: str, args: list[str]) -> tuple[int, int, int, int]: ...
def main_from_argv(argv: list[str]) -> int: ...

# callback(positions_bohr, cell_bohr, iteration, need_hessian) -> (energy, gradient, hessian | None)
TermCallback = Callable[[list[float], list[float] | None, int, bool], tuple[float, Any, Any]]

def units() -> dict[str, float]: ...
def atomic_numbers(symbols: list[str]) -> list[int]: ...
def native_calculators() -> list[str]: ...
def model_hessian_names() -> list[str]: ...
def mcp_component_tools() -> str: ...
def mcp_component(name: str, arguments: str, structures: str) -> str: ...
def model_hessian(
    name: str,
    symbols: list[str],
    positions_bohr: list[float],
    gradient: list[float] | None = None,
    charge: int = 0,
    multiplicity: int = 1,
    cell_bohr: list[float] | None = None,
) -> list[float]: ...
def dispersion(
    model: str,
    symbols: list[str],
    positions_bohr: list[float],
    charge: int = 0,
    need_hessian: bool = False,
    cell_bohr: list[float] | None = None,
) -> tuple[float, list[float], list[float] | None]: ...
def dispersion_properties(
    symbols: list[str],
    positions_bohr: list[float],
    charge: int = 0,
    cell_bohr: list[float] | None = None,
) -> dict[str, list[float]]: ...
def constraint_values(
    expressions: list[str],
    symbols: list[str],
    positions_bohr: list[float],
    cell_bohr: list[float] | None = None,
) -> dict[str, list[float]]: ...
def run_optmain(args: list[str], console: bool = True) -> list[dict[str, Any]]:
    """``optmain`` with the options of the command line; what every job ended with."""

def evaluate_native(
    name: str,
    symbols: list[str],
    positions_bohr: list[float],
    charge: int = 0,
    multiplicity: int | None = None,
    need_hessian: bool = False,
    cell_bohr: list[float] | None = None,
    unpaired_electrons: int | None = None,
    settings: list[tuple[str, str]] | None = None,
    *,
    pbc: tuple[bool, bool, bool] | None = None,
    excited_state: int = 0,
) -> tuple[float, list[float], list[float] | None]: ...
def native_stress(
    name: str,
    symbols: list[str],
    positions_bohr: list[float],
    cell_bohr: list[float],
    settings: list[tuple[str, str]] | None = None,
) -> list[list[float]]: ...
def stress_to_ase_voigt(stress: list[list[float]]) -> list[float]: ...
def stress_from_ase_voigt(voigt: list[float]) -> list[list[float]]: ...

class FollowedEvaluator:
    def __init__(self, calculator: Any) -> None: ...
    def evaluate(
        self,
        symbols: list[str],
        positions_bohr: list[float],
        charge: int,
        multiplicity: int,
        excited_state: int,
        need_hessian: bool = False,
        cell_bohr: list[float] | None = None,
        pbc: tuple[bool, bool, bool] | None = None,
    ) -> tuple[float, list[float], list[float] | None]: ...
    def take_warnings(self) -> list[str]: ...

class MapperNetwork:
    def __init__(self, json: str) -> None: ...
    def tasks(
        self,
        node: int,
        method: str = "afir_ts",
        options: dict[str, str] | None = None,
        top_n: int = 1,
    ) -> list[dict[str, Any]]: ...
    def select(
        self,
        priority: str,
        candidates: list[tuple[int, int, int]],
        *,
        temperature: float = 298.15,
        job: int = 0,
        seed: int = 0,
        options: dict[str, str] | None = None,
    ) -> int | None: ...
    def save(
        self,
        directory: str,
        *,
        stores: list[str] | None = None,
        reason: str | None = None,
        interrupted: bool = False,
    ) -> None: ...

def relax_band(
    evaluate: Callable[..., Any] | str | OniomCalculator,
    images: list[list[float]],
    *,
    symbols: list[str] | None = None,
    charge: int = 0,
    multiplicity: int | None = None,
    unpaired_electrons: int | None = None,
    cell_bohr: list[float] | None = None,
    pbc: tuple[bool, bool, bool] | None = None,
    # The options of nebmain as on the command line (["-qsm"], ["-lup"], ...).
    options: list[str] | None = None,
    coordinates: str | None = None,
    inserted: int | None = None,
    equal_spacing: bool | None = None,
    spacing: float | None = None,
    max_iterations: int | None = None,
    tolerance: float | None = None,
    surrogate: bool | None = None,
    fix_atoms: list[int] | None = None,
    constraints: list[str] | None = None,
    user_constraints: list[str] | None = None,
    constraint_functions: list[TermCallback | dict[str, Any]] | None = None,
    rigid_body_invariant: bool = True,
    output_dir: str | os.PathLike[str] | None = None,
    observer: Callable[..., bool | None] | None = None,
) -> dict[str, Any]: ...
def relax_environment(
    evaluate: Callable[..., Any],
    initial: list[float],
    *,
    method: str = "lbfgs",
    max_iterations: int = 500,
    tolerance: float = 1e-5,
    has_hessian: bool = False,
    mode_kill: bool = True,
) -> dict[str, Any]: ...

class NativeEvaluator:
    """A built-in calculator kept between evaluations (GFN-FF keeps its first topology)."""

    name: str
    def __init__(
        self,
        name: str,
        settings: list[tuple[str, str]] | None = None,
        *,
        follow_states: bool = False,
    ) -> None: ...
    def evaluate(
        self,
        symbols: list[str],
        positions_bohr: list[float],
        charge: int = 0,
        multiplicity: int | None = None,
        need_hessian: bool = False,
        cell_bohr: list[float] | None = None,
        unpaired_electrons: int | None = None,
        *,
        pbc: tuple[bool, bool, bool] | None = None,
        excited_state: int = 0,
    ) -> tuple[float, list[float], list[float] | None]: ...
    def take_warnings(self) -> list[str]: ...

def evaluate_bias(
    option: str,
    args: list[str],
    symbols: list[str],
    positions_bohr: list[float],
    need_hessian: bool = False,
    cell_bohr: list[float] | None = None,
    iteration: int = 0,
    time_au: float | None = None,
) -> tuple[float, list[float], list[float] | None]: ...
def check_term(
    callback: TermCallback,
    symbols: list[str],
    positions_bohr: list[float],
    cell_bohr: list[float] | None = None,
    step: float = 1e-5,
) -> dict[str, float]: ...
def check_bias(
    option: str,
    args: list[str],
    symbols: list[str],
    positions_bohr: list[float],
    cell_bohr: list[float] | None = None,
    step: float = 1e-5,
) -> dict[str, float]: ...
def bias_constant_derivatives(
    option: str,
    args: list[str],
    symbols: list[str],
    positions_bohr: list[float],
    hessian: list[float] | None = None,
    unbiased_gradient: list[float] | None = None,
    removed: list[float] | None = None,
    cell_bohr: list[float] | None = None,
    iteration: int = 0,
) -> list[dict[str, Any]]: ...

class Optimizer:
    def __init__(
        self,
        symbols: list[str],
        positions_bohr: list[float],
        *,
        charge: int = 0,
        multiplicity: int | None = None,
        unpaired_electrons: int | None = None,
        methods: list[str] = ...,
        saddle_order: int = 0,
        max_iterations: int = 1000,
        criteria: str = "normal",
        coordinates: str = "cart",
        exact_hessian_interval: int | None = None,
        max_trust_radius: float | None = None,
        min_trust_radius: float = 0.01,
        rigid_body_invariant: bool = True,
        dissociation_distance_angstrom: float | None = 10.0,
        cell_bohr: list[float] | None = None,
        ccqn_bonds: list[tuple[int, int]] | None = None,
        ccqn_product_bohr: list[float] | None = None,
        ccqn_cos_phi: float | None = None,
        fix_atoms: list[int] | None = None,
        constraints: list[str] | None = None,
        user_constraints: list[str] | None = None,
        constraint_functions: list[TermCallback | dict[str, Any]] | None = None,
        model_hessian: str | None = None,
        model_hessian_interval: int = 50,
        mode_kill: bool = False,
        crossing_check: bool = True,
        projection_trust: bool = False,
        step_factor: float | None = None,
        fragments: list[list[int]] | None = None,
        pbc: tuple[bool, bool, bool] | None = None,
        model_function: list[str] | None = None,
        excited_state: int = 0,
        # A 3N x 3N array (Hartree/Bohr^2), a file as optmain -ih reads it, or a model
        # Hessian name as optmain -modelhess takes it.
        initial_hessian: str | os.PathLike[str] | Any | None = None,
        calculator: OniomCalculator | None = None,
    ) -> None: ...
    @property
    def has_own_calculator(self) -> bool: ...
    @property
    def calculator_note(self) -> str | None: ...
    def evaluate_own(
        self, positions_bohr: list[float], need_hessian: bool = False, excited_state: int = 0
    ) -> tuple[float, list[float], list[float] | None]: ...
    @property
    def whole_positions_bohr(self) -> list[float]: ...
    def add_afir(
        self,
        gamma: float,
        fragment_a: list[int],
        fragment_b: list[int],
        gamma_final: float | None = None,
    ) -> None: ...
    def add_bias(self, option: str, args: list[str]) -> int: ...
    def add_term(self, label: str, callback: TermCallback) -> None: ...
    def ask(self) -> list[dict[str, Any]] | None: ...
    def tell(
        self,
        results: list[tuple[float, list[float], list[float] | None]],
        spin_squares: list[float | None] | None = None,
    ) -> dict[str, Any]: ...
    @property
    def wants_o1numhess(self) -> bool: ...
    @property
    def termination(self) -> str | None: ...
    @property
    def iterations(self) -> int: ...
    @property
    def positions_bohr(self) -> list[float]: ...

class VectorOptimizer:
    def __init__(
        self,
        x0: list[float],
        *,
        methods: list[str] = ...,
        saddle_order: int = 0,
        max_iterations: int = 1000,
        criteria: str = "normal",
        exact_hessian_interval: int | None = None,
        max_trust_radius: float | None = None,
        min_trust_radius: float = 0.01,
        ccqn_direction: list[float] | None = None,
        ccqn_cos_phi: float | None = None,
        mode_kill: bool = False,
    ) -> None: ...
    def hold(
        self,
        variables: list[tuple[int, float]] = ...,
        resper: list[tuple[ResperMap, int, float]] = ...,
    ) -> None: ...
    def ask(self) -> tuple[list[float], bool] | None: ...
    def tell(
        self, value: float, gradient: list[float], hessian: list[float] | None = None
    ) -> dict[str, Any]: ...
    @property
    def termination(self) -> str | None: ...
    @property
    def iterations(self) -> int: ...
    @property
    def x(self) -> list[float]: ...
    @property
    def dimension(self) -> int: ...

class ResperMap:
    """ReSPer coordinates: the map of reference structures and Gower's projection."""

    def __init__(
        self,
        symbols: list[str],
        frames_bohr: list[list[float]],
        atoms: list[int] | None = None,
        mass_weighted: bool = True,
        cell_bohr: list[float] | None = None,
    ) -> None: ...
    @staticmethod
    def from_points(points: list[list[float]]) -> ResperMap: ...
    @property
    def components(self) -> int: ...
    @property
    def coordinates(self) -> list[list[float]]: ...
    @property
    def eigenvalues(self) -> list[float]: ...
    @property
    def contributions(self) -> list[float]: ...
    @property
    def reproducibility(self) -> float: ...
    @property
    def symbols(self) -> list[str]: ...
    def project(self, x: list[float]) -> tuple[list[float], list[list[float]]]: ...
    def preimage(
        self,
        components: list[int],
        target: list[float],
        correct: bool = True,
        outside_factor: float = 2.0,
    ) -> dict[str, Any]: ...

def scan_points(
    ranges: list[tuple[float, float]],
    counts: list[int],
    grid: bool = False,
    order: str = "snake",
) -> dict[str, Any]: ...
def scan_stationary_points(
    shape: list[int], energies: list[float], periodic: list[bool] | None = None
) -> list[dict[str, Any]]: ...
def rigid_scan_structure(
    symbols: list[str],
    positions_bohr: list[float],
    targets: list[tuple[str, str, float]],
    cell_bohr: list[float] | None = None,
) -> list[float]: ...
def run_relaxedscan(args: list[str], console: bool = True) -> dict[str, Any]:
    """``relaxedscan`` with the options of the command line: its points and stationary points."""

class Bitss:
    def __init__(
        self,
        first: list[float],
        second: list[float],
        *,
        methods: list[str] = ...,
        max_iterations: int = 1000,
        criteria: str = "normal",
        exact_hessian_interval: int | None = None,
        max_trust_radius: float | None = None,
        min_trust_radius: float = 0.01,
        reduction: float = 0.5,
        alpha: float = 10.0,
        beta: float = 0.1,
        coefficient_interval: int = 100,
        interpolation_points: int = 3,
        min_separation: float = 0.01,
        max_stages: int = 30,
        step_share: float | None = 0.5,
        midpoint_shift: float | None = None,
        stop_on_midpoint_gradient: bool = True,
        frozen: list[int] | None = None,
    ) -> None: ...
    def ask(self) -> list[dict[str, Any]] | None: ...
    def tell(
        self, values: list[tuple[float, list[float] | None, list[float] | None]]
    ) -> dict[str, Any]: ...
    @property
    def end(self) -> str | None: ...
    @property
    def converged(self) -> bool: ...
    @property
    def images(self) -> tuple[list[float], list[float]]: ...
    @property
    def transition_state_estimate(self) -> list[float]: ...
    @property
    def midpoint_energy(self) -> float | None: ...
    @property
    def stage(self) -> int: ...
    @property
    def target_separation(self) -> float: ...
    @property
    def iterations(self) -> int: ...
    @property
    def dimension(self) -> int: ...

_PointValues = list[tuple[float, list[float] | None, list[float] | None]]
_ImageValue = tuple[float, list[float], float | None, list[float] | None]

class ElasticImagePair:
    def __init__(
        self,
        first: list[float],
        second: list[float],
        *,
        max_iterations: int = 999,
        beta: float = 1.0,
        elements: list[str] | None = None,
        microiterations: int = 0,
        micro_tolerance: float = 0.01,
        frozen: list[int] | None = None,
    ) -> None: ...
    def ask(self) -> tuple[list[float], list[float]] | None: ...
    def tell(self, images: tuple[_ImageValue, _ImageValue]) -> dict[str, Any]: ...
    @property
    def end(self) -> str | None: ...
    @property
    def images(self) -> tuple[list[float], list[float]]: ...
    @property
    def transition_state_estimate(self) -> list[float]: ...
    @property
    def iterations(self) -> int: ...
    @property
    def separation(self) -> float: ...
    @property
    def beta(self) -> float: ...

class Dimer:
    def __init__(
        self,
        start: list[float],
        axis: list[float],
        *,
        separation: float = 1e-4,
        trial_angle: float = ...,
        max_iterations: int = 1000,
        max_rotations: int = 8,
        rotation_tolerance: float = ...,
        max_step: float = 0.1,
        criteria: str = "normal",
        frozen: list[int] | None = None,
    ) -> None: ...
    def ask(self) -> list[dict[str, Any]] | None: ...
    def tell(self, values: _PointValues) -> dict[str, Any]: ...
    @property
    def center(self) -> list[float]: ...
    @property
    def axis(self) -> list[float]: ...
    @property
    def curvature(self) -> float: ...
    @property
    def converged(self) -> bool | None: ...

class HighIndexSaddleSearch:
    def __init__(
        self,
        start: list[float],
        directions: list[list[float]],
        *,
        order: int,
        method: str = "hiosd",
        update: str | None = None,
        curvature: str = "forward",
        guard: bool = False,
        flow: str = "reflected",
        alpha0: float = 1e-7,
        alpha_rate: float = 0.5,
        dimer_length: float = 1e-3,
        shrink: float = 0.0,
        min_dimer_length: float = 1e-4,
        first_step: float = 0.05,
        max_step: float = 0.2,
        max_iterations: int = 1000,
        criteria: str = "normal",
        hand_off: tuple[float, float] | None = None,
        symbols: list[str] | None = None,
        rigid: bool = True,
        mass_weighted: bool = False,
        frozen: list[int] | None = None,
        seed: int = 1,
    ) -> None: ...
    def ask(self) -> list[dict[str, Any]] | None: ...
    def tell(self, values: _PointValues) -> dict[str, Any]: ...
    @property
    def point(self) -> list[float]: ...
    @property
    def directions(self) -> list[list[float]]: ...
    @property
    def curvatures(self) -> list[float]: ...
    @property
    def end(self) -> str | None: ...
    @property
    def gradients(self) -> int: ...
    @property
    def hessians(self) -> int: ...

class LowestModes:
    def __init__(
        self,
        point: list[float],
        *,
        count: int = 2,
        start: list[list[float]] | None = None,
        curvature: str = "forward",
        dimer_length: float = 1e-3,
        max_iterations: int = 100,
        tolerance: float = 1e-3,
        symbols: list[str] | None = None,
        rigid: bool = True,
        mass_weighted: bool = False,
        seed: int = 1,
    ) -> None: ...
    def ask(self) -> list[dict[str, Any]] | None: ...
    def tell(self, values: _PointValues) -> None: ...
    @property
    def values(self) -> list[float]: ...
    @property
    def directions(self) -> list[list[float]]: ...
    @property
    def converged(self) -> bool: ...

class RidgeBand:
    def __init__(
        self,
        chain: list[list[float]],
        modes: tuple[list[float], list[float]],
        *,
        spring_force: str = "full",
        transverse: str = "lowest",
        step_rule: str = "fire",
        spring: float = 0.1,
        climb_rms: float = 3e-3,
        climb_after: int = 50,
        tolerance: float = 3e-4,
        max_iterations: int = 500,
        curvature: str = "forward",
        dimer_length: float = 1e-3,
        symbols: list[str] | None = None,
        rigid: bool = True,
        mass_weighted: bool = False,
    ) -> None: ...
    def ask(self) -> list[dict[str, Any]] | None: ...
    def tell(self, values: _PointValues) -> dict[str, Any]: ...
    @property
    def chain(self) -> list[list[float]]: ...
    @property
    def highest(self) -> int: ...
    @property
    def saddle_directions(self) -> list[list[float]]: ...

class FixedRidgeBand:
    def __init__(
        self,
        first: list[float],
        last: list[float],
        first_mode: list[float],
        last_mode: list[float],
        *,
        images: int = 7,
        tolerance: float = 3e-4,
        max_iterations: int = 300,
        line_tolerance: float = 1e-5,
        min_overlap: float = 0.7,
        symbols: list[str] | None = None,
        rigid: bool = True,
        mass_weighted: bool = False,
    ) -> None: ...
    def ask(self) -> list[dict[str, Any]] | None: ...
    def tell(self, values: _PointValues) -> dict[str, Any] | None: ...
    @property
    def saddle_point(self) -> list[float] | None: ...
    @property
    def saddle_directions(self) -> list[list[float]]: ...
    @property
    def end(self) -> str | None: ...

def hisd_upward_starts(
    point: list[float],
    unstable: list[list[float]],
    stable: list[list[float]],
    displacement: float = 0.05,
) -> list[dict[str, Any]]: ...
def hisd_downward_starts(
    point: list[float],
    unstable: list[list[float]],
    displacement: float = 0.05,
    fan: int = 0,
) -> list[dict[str, Any]]: ...

class SpringPair:
    def __init__(
        self,
        start: list[float],
        direction: list[float],
        *,
        spring_length: float = 0.1,
        offset: float = 0.1,
        drift_rate: float = 0.5,
        spring_rate: float = 1.0,
        climb_rate: float = 5.0,
        drift_tolerance: float = 2e-3,
        drift_steps: int = 200,
        max_cycles: int = 999,
        max_step: float = 0.1,
        criteria: str = "normal",
    ) -> None: ...
    def ask(self) -> list[dict[str, Any]] | None: ...
    def tell(self, values: _PointValues) -> dict[str, Any]: ...
    @property
    def saddle_estimate(self) -> list[float] | None: ...
    @property
    def converged(self) -> bool | None: ...

class TwoPointShs:
    def __init__(
        self,
        center: list[float],
        other: list[float],
        *,
        step: float = 0.05,
        max_spheres: int = 300,
        sphere_iterations: int = 30,
        sphere_gradient: float = 1e-4,
    ) -> None: ...
    def ask(self) -> list[dict[str, Any]] | None: ...
    def tell(self, values: _PointValues) -> tuple[int, float, list[float], float] | None: ...
    @property
    def path(self) -> list[tuple[int, float, list[float], float]]: ...
    @property
    def transition_state_estimate(self) -> tuple[list[float], float] | None: ...

class NewtonTrajectory:
    def __init__(
        self,
        start: list[float],
        *,
        final_point: list[float] | None = None,
        direction: list[float] | None = None,
        step_length: float = 0.5,
        rms_tolerance: float = 1.7e-3,
        corrector_iterations: int = 25,
        max_iterations: int = 999,
    ) -> None: ...
    def ask(self) -> list[dict[str, Any]] | None: ...
    def tell(self, values: _PointValues) -> dict[str, Any]: ...
    @property
    def points(self) -> list[tuple[list[float], float]]: ...
    @property
    def transition_state_estimate(self) -> tuple[list[float], float] | None: ...

class Addf:
    def __init__(
        self,
        start: list[float],
        *,
        initial_length: float = 0.0567,
        step_length: float = 0.189,
        max_spheres: int = 300,
        paths: int = 5,
        sphere_iterations: int = 30,
        sphere_gradient: float = 1e-4,
        duplicate_angle: float = 0.01,
        gradient_tolerance: float = 1e-3,
    ) -> None: ...
    def ask(self) -> list[dict[str, Any]] | None: ...
    def tell(self, values: _PointValues) -> list[str]: ...
    @property
    def transition_states(self) -> list[tuple[int, list[float], float]]: ...
    @property
    def paths(self) -> list[list[tuple[int, float, list[float], float, float]]]: ...
    @property
    def finished(self) -> bool: ...

class GrowingString:
    def __init__(
        self,
        first: list[float],
        second: list[float] | None = None,
        *,
        driving: list[str] | None = None,
        settings: dict[str, str] | None = None,
    ) -> None: ...
    def ask(self) -> list[dict[str, Any]] | None: ...
    def tell(self, values: _PointValues) -> int: ...
    @property
    def transition_state_estimate(self) -> tuple[list[float], float] | None: ...
    @property
    def converged(self) -> bool | None: ...

class FreezingString:
    def __init__(
        self, first: list[float], second: list[float], *, settings: dict[str, str] | None = None
    ) -> None: ...
    def ask(self) -> list[dict[str, Any]] | None: ...
    def tell(self, values: _PointValues) -> bool: ...
    @property
    def transition_state_estimate(self) -> tuple[list[float], float] | None: ...
    @property
    def path(self) -> list[tuple[list[float], float]]: ...

class DsAfir:
    def __init__(
        self, first: list[float], second: list[float], *, settings: dict[str, str] | None = None
    ) -> None: ...
    def ask(self) -> list[dict[str, Any]] | None: ...
    def tell(self, values: _PointValues) -> float: ...
    @property
    def transition_state_estimate(self) -> tuple[list[float], float] | None: ...
    @property
    def converged(self) -> bool | None: ...

def pair_methods() -> list[tuple[str, str, str, list[tuple[str, str]]]]: ...
def run_pair_method(
    name: str,
    objective: str,
    points: list[list[float]],
    *,
    settings: dict[str, str] | None = None,
    driving: list[str] | None = None,
    directory: str = ".",
) -> dict[str, Any]: ...
def bitss_function(
    kappa_e: float,
    kappa_d: float,
    target: float,
    first: list[float],
    second: list[float],
    energies: tuple[float, float],
    gradient_first: list[float],
    gradient_second: list[float],
) -> tuple[float, list[float]]: ...
def bitss_coefficients(
    alpha: float, beta: float, target: float, barrier: float, projected: tuple[float, float]
) -> tuple[float, float]: ...

class DirectMaxFlux:
    def __init__(
        self,
        chain: list[list[float]],
        *,
        points: int = 3,
        guards: int = 1,
        segments: int = 4,
        degree: int = 3,
        beta: float | None = None,
        moves: bool = True,
        move_settings: dict[str, float] | None = None,
        tolerance: float | None = None,
        max_iterations: int = 200,
        max_step: float = 0.3,
        first_curvature: float | None = None,
        memory: int = 6,
        rise: float = 0.01,
        settle: float = 1e-3,
        patience: int = 8,
        refit_unevenness: float = 3.0,
        fixed: list[int] | None = None,
    ) -> None: ...
    def ask(self) -> list[dict[str, Any]] | None: ...
    def tell(self, values: list[tuple[float, list[float]]]) -> dict[str, Any] | None: ...
    @property
    def report(self) -> dict[str, Any] | None: ...
    @property
    def result(self) -> dict[str, Any] | None: ...
    @property
    def end(self) -> str | None: ...
    @property
    def converged(self) -> bool: ...
    @property
    def transition_state_estimate(self) -> list[float] | None: ...
    def path_points(self, count: int = 41) -> list[list[float]]: ...
    @property
    def control_points(self) -> list[list[float]]: ...
    @property
    def iterations(self) -> int: ...
    @property
    def dimension(self) -> int: ...

class DmfPath:
    def __init__(self, chain: list[list[float]], *, degree: int = 3, segments: int = 4) -> None: ...
    @staticmethod
    def from_control(control: list[list[float]], *, degree: int = 3) -> DmfPath: ...
    @property
    def control(self) -> list[list[float]]: ...
    def at(self, t: float) -> list[float]: ...
    def velocity(self, t: float) -> list[float]: ...
    @property
    def length(self) -> float: ...
    def parameter(self, share: float) -> float: ...
    def points(self, shares: list[float]) -> list[list[float]]: ...
    @property
    def unevenness(self) -> float: ...
    def action(
        self,
        shares: list[float],
        energies: list[float],
        gradients: list[list[float]],
        beta: float,
    ) -> dict[str, Any]: ...
    @property
    def dimension(self) -> int: ...

def dmf_profile(
    shares: list[float], energies: list[float], slopes: list[float], samples: int = 41
) -> dict[str, Any]: ...

class Irc:
    def __init__(
        self,
        x: list[float],
        *,
        symbols: list[str] | None = None,
        masses: list[float] | None = None,
        project_rigid: bool = True,
        step_size: float = 0.1,
        max_steps: int = 300,
        method: str = "lqa",
        exact_hessian_interval: int | None = None,
        options: list[str] | None = None,
        cell_bohr: list[float] | None = None,
        pbc: tuple[bool, bool, bool] | None = None,
        fix_atoms: list[int] | None = None,
        constraints: list[str] | None = None,
        user_constraints: list[str] | None = None,
        constraint_functions: list[TermCallback | dict[str, Any]] | None = None,
    ) -> None: ...
    def ask(self) -> dict[str, Any] | None: ...
    def tell(
        self,
        energy: float,
        gradient: list[float],
        hessian: list[float] | None = None,
        *,
        raw_energy: float | None = None,
        raw_gradient: list[float] | None = None,
        bias_hessian: list[float] | None = None,
    ) -> None: ...
    def events(self) -> list[dict[str, Any]]: ...
    def mass_weighted(self, x: list[float]) -> list[float]: ...
    @property
    def start(self) -> list[float]: ...
    @property
    def kind(self) -> str | None: ...
    @property
    def evaluations(self) -> int: ...
    @property
    def method(self) -> str: ...

def irc_lqa_step(
    hessian: list[float], gradient: list[float], step_size: float, gradient_floor: float = 1e-8
) -> list[float]: ...
def irc_first_step_test(
    displacement: list[float],
    eigenvalue: float,
    energy_ts: float,
    gradient_ts: list[float],
    energy_trial: float,
    gradient_trial: list[float],
) -> tuple[float, float, float, float]: ...
def irc_methods() -> list[str]: ...

# Molecular dynamics.
class MolecularDynamics:
    def __init__(
        self,
        x: list[float],
        *,
        symbols: list[str] | None = None,
        masses: list[float] | None = None,
        cell_bohr: list[float] | None = None,
        charge: int = 0,
        multiplicity: int = 1,
        ensemble: str = "nve",
        time_step: float | None = None,
        steps: int = 1000,
        temperature: float = 298.15,
        pressure: float = 101.3,
        options: list[str] | None = None,
        velocities: list[float] | None = None,
        fix_atoms: list[int] | None = None,
        constraints: list[str] | None = None,
        user_constraints: list[str] | None = None,
        constraint_functions: list[Any] | None = None,
        rigid: str | None = None,
        coordinates: str | None = None,
        invariant: bool = True,
        electronic_state: int | None = None,
    ) -> None: ...
    def ask(self) -> dict[str, Any] | None: ...
    def tell(
        self,
        energy: float,
        gradient: list[float],
        hessian: list[float] | None = None,
        *,
        raw_energy: float | None = None,
        stress: list[float] | None = None,
        state_energies: list[float] | None = None,
        state_gradients: list[list[float] | None] | None = None,
    ) -> dict[str, Any]: ...
    @property
    def dof(self) -> float: ...
    @property
    def finished(self) -> bool: ...
    @property
    def evaluations(self) -> int: ...
    @property
    def frozen_modes(self) -> int: ...
    @property
    def parts(self) -> tuple[str, str, str | None]: ...
    @property
    def references(self) -> list[str]: ...

def md_parts() -> tuple[list[str], list[str], list[str], list[str]]: ...
def wham(
    centers: list[float],
    k: list[float],
    samples: list[list[float]],
    kt: float,
    bins: int = 50,
    range: tuple[float, float] | None = None,
    period: float | None = None,
) -> dict[str, Any]: ...
def mbar(
    centers: list[float],
    k: list[float],
    samples: list[list[float]],
    kt: float,
    bins: int = 50,
    range: tuple[float, float] | None = None,
    period: float | None = None,
) -> dict[str, Any]: ...

# Trajectory analysis.
class TrajectoryAnalysis:
    def __init__(
        self,
        symbols: list[str],
        positions_angstrom: list[list[float]],
        times_fs: list[float],
        *,
        cells_angstrom: list[list[float]] | None = None,
        velocities_angstrom_per_fs: list[list[float]] | None = None,
    ) -> None: ...
    @staticmethod
    def from_file(
        path: str | os.PathLike[str], *, interval_fs: float = 1.0
    ) -> TrajectoryAnalysis: ...
    def __len__(self) -> int: ...
    @property
    def times_fs(self) -> list[float]: ...
    @property
    def symbols(self) -> list[str]: ...
    @property
    def has_velocities(self) -> bool: ...
    @property
    def periodic(self) -> bool: ...
    def skip(self, time_fs: float) -> TrajectoryAnalysis: ...
    def rdf(self, bin_width: float = 0.02, r_max: float | None = None) -> dict[str, Any]: ...
    def msd(
        self, fit: tuple[float, float] = (0.1, 0.9), remove_drift: bool = True
    ) -> dict[str, Any]: ...
    def vacf(self, share: float = 0.5) -> dict[str, Any]: ...
    def vdos(
        self,
        share: float = 0.5,
        max_wavenumber: float | None = None,
        window: str = "hann",
        padding: int = 4,
    ) -> dict[str, Any]: ...
    def diffusion_green_kubo(self, lags: int | None = None, blocks: int = 10) -> dict[str, Any]: ...
    def orientational(
        self, bonds: list[tuple[int, int]], order: int = 2, share: float = 0.5
    ) -> list[float]: ...
    def fluctuations(self) -> dict[str, list[float]]: ...
    def geometry(self, atoms: list[list[int]]) -> dict[str, list[float]]: ...
    def pca(self, components: int = 3) -> dict[str, Any]: ...

def block_average(values: list[float]) -> dict[str, Any]: ...
def autocorrelation(x: list[float], lags: int, direct: bool = False) -> list[float]: ...
def power_spectrum(
    x: list[float], dt_fs: float, share: float = 0.5, window: str = "hann", padding: int = 4
) -> dict[str, Any]: ...
def infrared_spectrum(
    c: list[float],
    dt_fs: float,
    temperature: float = 300.0,
    correction: str = "harmonic",
    window: str = "hann",
    padding: int = 4,
    max_wavenumber: float | None = None,
) -> dict[str, Any]: ...
def green_kubo(
    components: list[list[float]],
    dt_fs: float,
    prefactor: float = 1.0,
    lags: int | None = None,
    blocks: int = 10,
) -> dict[str, Any]: ...
def analyze_md_run(
    path: str | os.PathLike[str],
    options: list[str] = ...,
    out_dir: str | os.PathLike[str] | None = None,
) -> list[str]: ...

class StepMethod:
    def __init__(
        self,
        method: str,
        n: int,
        *,
        saddle_order: int = 0,
        max_step: float | None = None,
    ) -> None: ...
    def propose(self, x: list[float], value: float, gradient: list[float]) -> list[float]: ...
    def set_hessian(self, hessian: list[float]) -> None: ...
    def reset(self) -> None: ...
    @property
    def hessian(self) -> list[float]: ...
    @property
    def name(self) -> str: ...
    @property
    def enhancers(self) -> list[str]: ...
    @property
    def iterations(self) -> int: ...

def objective_names() -> list[str]: ...
def evaluate_objective(
    name: str,
    x: list[float],
    need_hessian: bool = False,
    options: list[tuple[str, str]] | None = None,
) -> tuple[float, list[float], list[float] | None]: ...

class Coordinates:
    def __init__(
        self,
        name: str,
        symbols: list[str],
        positions_bohr: list[float],
        cell_bohr: list[float] | None = None,
    ) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def dimension(self) -> int: ...
    def values(self, positions_bohr: list[float]) -> list[float]: ...
    def wilson_b(self, positions_bohr: list[float]) -> list[float]: ...
    def displacement(self, to_bohr: list[float], from_bohr: list[float]) -> list[float]: ...
    def cartesian_step(self, positions_bohr: list[float], dq: list[float]) -> list[float]: ...
    def needs_reinitialization(self, positions_bohr: list[float]) -> bool: ...
    def reinitialize(self, positions_bohr: list[float]) -> None: ...

def coordinate_names() -> list[str]: ...
def to_hyperspherical(x: list[float]) -> list[float]: ...
def from_hyperspherical(q: list[float]) -> list[float]: ...
def hyperspherical_jacobian(q: list[float]) -> list[float]: ...
def hyperspherical_wilson(x: list[float]) -> list[float]: ...
def gradient_to_hyperspherical(x: list[float], gradient: list[float]) -> list[float]: ...
def align(
    symbols: list[str],
    mobile_bohr: list[float],
    reference_bohr: list[float],
    mass_weighted: bool = False,
) -> tuple[list[float], float]: ...
def connectivity(
    symbols: list[str],
    positions_bohr: list[float],
    scale: float = 1.3,
    cell_bohr: list[float] | None = None,
) -> dict[str, list[list[int]]]: ...
def sampling_priority_names() -> list[str]: ...

# The kinetics of networks (mopt_rs.kinetics).
def kinetics_solve(
    rates: list[list[float]],
    p0: list[float],
    times: list[float],
    *,
    solver: str = "propagate",
    tolerance: float = 1e-8,
    time_rule: str = "gershgorin",
    closure: str = "a",
    trajectories: int = 10000,
    seed: int = 1,
) -> dict[str, Any]: ...
def kinetics_contract(rates: list[list[float]], eliminate: list[int]) -> dict[str, Any]: ...
def kinetics_greedy_order(
    rates: list[list[float]], *, time_rule: str = "gershgorin"
) -> list[tuple[int, float, float, float, float, float]]: ...
def kinetics_scores(
    rates: list[list[float]],
    p0: list[float],
    t: float,
    *,
    score: str = "population",
    solver: str = "propagate",
    visits: list[int] | None = None,
    tolerance: float = 1e-8,
    time_rule: str = "gershgorin",
) -> list[float]: ...
def kinetics_part_names() -> tuple[list[str], list[str], list[str]]: ...
def kinetics_kramers(gamma: float, omega_b: float) -> float: ...
def kinetics_grote_hynes(
    omega_b: float,
    *,
    gamma: float | None = None,
    memory: float | None = None,
    times: list[float] | None = None,
    values: list[float] | None = None,
) -> float: ...
def kinetics_weak_coupling(
    coupling: float,
    gradient_difference: float,
    slope_product: float,
    reduced_mass: float,
    temperature: float,
    barrier: float,
) -> dict[str, Any]: ...
def kinetics_curvature_tunnelling(
    arc_length: list[float], adiabatic: list[float], mass_ratio: list[float], temperature: float
) -> tuple[float, float]: ...
def kinetics_beyer_swinehart(wavenumbers: list[float], cell: float, cells: int) -> list[float]: ...
def kinetics_branching(
    pairs: list[tuple[float, int | None, int | None]], weighting: str = "flux"
) -> dict[str, Any]: ...
def kinetics_msm(
    dtrajs: list[list[int]], states: int, lag: int, lag_time: float
) -> dict[str, Any]: ...
def kinetics_dwell(dtrajs: list[list[int]], states: int, dt: float) -> dict[str, Any]: ...
def kinetics_kernel_contraction(
    k0: list[list[float]], k1: list[list[float]], eliminated: list[int]
) -> dict[str, Any]: ...
def kinetics_network_model(
    network: str, times: list[float], **options: object
) -> dict[str, Any]: ...
def kinetics_network_refinement(
    network: str, times: list[float], targets: list[int], **options: object
) -> list[tuple[str, str, str, str, float, float, float]]: ...
def kinetics_solver_names() -> list[str]: ...
def kinetics_network_derivatives(
    network: str,
    times: list[float],
    *,
    temperature: float = 300.0,
    solver: str = "propagate",
    energy_type: str = "electronic",
    start: int | None = None,
    level: int = 1,
    tolerance: float = 1e-8,
    time_rule: str = "gershgorin",
    rate_theory: str = "eyring",
    corrections: list[str] = ...,
    annotations: str | None = None,
    concentrations: list[tuple[str, float]] = ...,
    association: str = "none",
    pressure: float = 101325.0,
    standard_concentration: float | None = None,
) -> dict[str, Any]: ...
def kinetics_network_mass_action(
    network: str,
    times: list[float],
    *,
    total_concentration: float,
    concentrations: list[tuple[str, float]] = ...,
    annotations: str | None = None,
    temperature: float = 300.0,
    energy_type: str = "electronic",
    rate_theory: str = "eyring",
    corrections: list[str] = ...,
    start: int | None = None,
    level: int = 1,
    ode_tolerance: float = 1e-7,
    association: str = "none",
    pressure: float = 101325.0,
    standard_concentration: float | None = None,
    solver: str = "rosenbrock",
    contract_theta: float = 100.0,
) -> dict[str, Any]: ...
def kinetics_tunnelling(
    name: str,
    imaginary_wavenumber_cm1: float,
    barriers_hartree: tuple[float, float],
    temperature: float,
    *,
    bell_truncation: str = "auto",
    bell_barrier: str = "higher",
) -> tuple[float, list[tuple[str, float]]] | None: ...
def kinetics_transfer_rate(
    form: str,
    reorganization: float,
    coupling: float,
    dg0: float,
    temperature: float = 298.15,
    wavenumber: float | None = None,
) -> tuple[float, float, float]: ...
def kinetics_cvt_factor(
    arc_length: list[float],
    energies: list[float],
    frequencies: list[list[float]],
    temperature: float = 298.15,
    qrrho: str = "grimme",
    qrrho_cutoff: float = 100.0,
) -> tuple[float, float, float]: ...
def kinetics_standard_state_shift(
    temperature: float = 298.15,
    pressure: float = 101325.0,
    standard_concentration: float = 1.0,
) -> float: ...

class SamplingPriority:
    def __init__(self, name: str, seed: int = 0) -> None: ...
    def select(self, energies: list[float], visits: list[int]) -> int | None: ...

# The orientation search of orientsearch.
def complex_parts(
    symbols: list[str],
    positions_bohr: list[float],
    parts: list[list[int]] | None = None,
    *,
    fixed: list[int] | None = None,
    cell_bohr: list[float] | None = None,
    bond_scale: float = 1.2,
) -> dict[str, Any]: ...
def compare_structures(
    symbols: list[str],
    a_bohr: list[float],
    b_bohr: list[float],
    *,
    cell_bohr: list[float] | None = None,
    bond_scale: float = 1.2,
    mirror: bool = True,
) -> dict[str, Any]: ...

class OrientationSearch:
    def __init__(
        self,
        symbols: list[str],
        positions_bohr: list[float],
        parts: list[list[int]] | None = None,
        *,
        samples: int = 5,
        scheme: str = "ring",
        seed: int = 1,
        distance_angstrom: float = 5.0,
        contact: tuple[float, float] = (1.0, 1.2),
        clash: float = 0.5,
        refine: float = 0.0,
        priority: str = "tabu",
        rmsd_angstrom: float = 0.25,
        energy_tolerance: float = 1e-4,
        mirror: bool = True,
        bond_scale: float = 1.2,
        fixed: list[int] | None = None,
        cell_bohr: list[float] | None = None,
        charge: int = 0,
        multiplicity: int = 1,
    ) -> None: ...
    def ask(self) -> dict[str, Any] | None: ...
    def tell(
        self, positions_bohr: list[float] | None, energy: float, converged: bool = True
    ) -> dict[str, Any]: ...
    def minima(self, temperature: float = 298.15) -> list[dict[str, Any]]: ...
    @property
    def parts(self) -> list[list[int]]: ...

# The parts of the conformation search.
def bond_graph(
    symbols: list[str],
    positions_bohr: list[float],
    scale: float = 1.1,
    cell_bohr: list[float] | None = None,
) -> dict[str, list[Any]]: ...
def graph_isomorphism(
    symbols: list[str],
    first_bohr: list[float],
    second_bohr: list[float],
    scale: float = 1.1,
) -> list[int] | None: ...
def invariant_rmsd(
    symbols: list[str],
    first_bohr: list[float],
    second_bohr: list[float],
    permutations: bool = True,
    mirror: bool = False,
) -> tuple[float, list[int], bool]: ...
def zmatrix(
    symbols: list[str], positions_bohr: list[float]
) -> tuple[list[list[int]], list[list[float]]]: ...
def run_confsearch(
    args: list[str], moves: dict[str, Callable[..., Any]] | None = None
) -> tuple[str, int, list[float]]: ...
def conformer_move_names() -> list[str]: ...
def crossover_conformers(
    symbols: list[str],
    reference_bohr: list[float],
    parents_bohr: list[list[float]],
    max: int = 300,
    rmsd_bohr: float | None = None,
    seed: int = 1,
) -> list[list[float]]: ...

class ConformerEnsemble:
    def __init__(
        self,
        energy: float | None = None,
        rmsd: float | None = None,
        rotational: float | None = None,
        permutations: bool = True,
        mirror: bool = True,
    ) -> None: ...
    def add(
        self, symbols: list[str], positions_bohr: list[float], energy: float, origin: str = ""
    ) -> tuple[str, int]: ...
    def energies(self) -> list[float]: ...
    def thermo(self, temperature: float = 298.15) -> dict[str, Any]: ...
    def __len__(self) -> int: ...

class RmsdMetadynamics:
    def __init__(self, height: float, alpha: float, atoms: list[int] | None = None) -> None: ...
    def add_reference(self, symbols: list[str], positions_bohr: list[float]) -> None: ...
    @property
    def references(self) -> int: ...
    def evaluate(
        self, symbols: list[str], positions_bohr: list[float]
    ) -> tuple[float, list[float]]: ...

def electron_transfer_rate(
    g_crossing: float,
    g_initial: float,
    g_final: float,
    v12: float,
    temperature: float = 298.15,
    intermolecular: bool = True,
    nuclear_frequency: float | None = None,
    standard_pressure: float = 101_325.0,
) -> dict[str, float]: ...
def run_etmain(args: list[str], console: bool = True) -> list[dict[str, Any]]: ...
def et_inner_sphere(
    g_initial: list[float], g_final: list[float], barrier: float
) -> dict[str, Any]: ...
def et_scher_holstein(
    quantum: float, reaction_energy: float, reorganization: float, temperature: float = 298.15
) -> tuple[float, float, float]: ...
def et_marcus_rate(
    ln_gamma: float,
    coupling: float,
    reorganization: float,
    barrier: float,
    temperature: float = 298.15,
) -> float: ...
def et_zhu_nakamura_rate(
    g_initial: list[float],
    g_final: list[float],
    inner_barrier: float,
    coupling: float,
    barrier: float,
    temperature: float = 298.15,
) -> tuple[str, float, float, float]: ...
def et_marcus_levich_jortner(
    coupling: float,
    quantum: float,
    inner_reorganization: float,
    solvent_reorganization: float,
    reaction_energy: float,
    temperature: float = 298.15,
) -> float: ...
def et_mulliken_hush(
    excitation: float, transition_dipole: list[float], distance: float
) -> float: ...
def et_fragment_charge_difference(
    excitation: float,
    transition_population: list[float],
    population_change: list[float],
    donor: list[int],
    acceptor: list[int],
) -> float: ...
def et_generalized_mulliken_hush(
    excitation: float, transition_dipole: list[float], dipole_difference: list[float]
) -> float: ...
def et_charge_transfer(
    positions: list[list[float]], population_change: list[float]
) -> tuple[float, float, list[float], list[float]]: ...
def et_solvent_shifts(
    lambda_initial: float, lambda_final: float, reaction_energy: float, alpha: float
) -> tuple[float, float, float]: ...
def et_apparent_barrier(rate: float, temperature: float = 298.15) -> float: ...

_ClosedShell = tuple[list[list[float]], list[list[float]], list[float], list[float], list[int]]

def et_transfer_integral(
    fock: list[list[float]],
    overlap: list[list[float]],
    a: list[list[float]],
    b: list[list[float]],
) -> dict[str, Any]: ...
def et_fock_from_orbitals(
    overlap: list[list[float]], coefficients: list[list[float]], energies: list[float]
) -> list[list[float]]: ...
def et_projection_coupling(
    pair: _ClosedShell,
    a: _ClosedShell,
    b: _ClosedShell,
    atoms_a: list[int],
    atoms_b: list[int],
    carrier: str = "hole",
) -> dict[str, Any]: ...
def et_four_point_reorganization(a: tuple[float, float], b: tuple[float, float]) -> float: ...
def et_hopping_rate(
    coupling: float, reorganization: float, delta_g: float, temperature: float = 298.15
) -> float: ...
def et_proton_states(
    first: float,
    spacing: float,
    potential: list[float],
    mass_amu: float = 1.007276466621,
    count: int = 10,
) -> dict[str, Any]: ...
def et_pcet_rate(
    first: float,
    spacing: float,
    reactant: list[float],
    product: list[float],
    coupling: float,
    reorganization: float,
    delta_g0: float = 0.0,
    temperature: float = 298.15,
    mass_amu: float = 1.007276466621,
    count: int = 10,
) -> dict[str, Any]: ...
def images_between(
    method: str,
    first_bohr: list[float],
    last_bohr: list[float],
    count: int,
    symbols: list[str] | None = None,
    cell_bohr: list[float] | None = None,
    fixed_atoms: list[int] | None = None,
    max_iterations: int = 500,
) -> tuple[list[list[float]], int, bool]: ...
def structure_distances(
    symbols: list[str], frames_bohr: list[list[float]], cell_bohr: list[float] | None = None
) -> list[list[float]]: ...
def classical_mds(distances: list[list[float]], components: int = 2) -> dict[str, Any]: ...
def principal_components(
    data: list[list[float]], standardize: bool = False, components: int | None = None
) -> dict[str, Any]: ...
def path_variables(
    kind: str,
    symbols: list[str],
    frames_bohr: list[list[float]],
    cell_bohr: list[float] | None = None,
) -> tuple[list[list[float]], list[str]]: ...
def koopman_observables(
    positions: list[float], gradient: list[float] | None = None, degree: int = 2
) -> list[float]: ...
def dynamic_modes(
    snapshots: list[list[float]], rank_threshold: float = 1e-5
) -> dict[str, Any] | None: ...
def normal_modes(
    symbols: list[str],
    positions_bohr: list[float],
    hessian: list[float],
    project: bool = True,
    cell_bohr: list[float] | None = None,
    gradient: list[float] | None = None,
) -> dict[str, Any]: ...
def point_group(
    symbols: list[str], positions_bohr: list[float], tolerance: float = 0.01
) -> tuple[str, int, int, float]: ...
def rotational_constants(symbols: list[str], positions_bohr: list[float]) -> list[float]: ...
def vibrational_analysis(
    symbols: list[str],
    positions_bohr: list[float],
    hessian: list[float],
    energy: float,
    charge: int = 0,
    multiplicity: int = 1,
    options: dict[str, Any] | None = None,
    gradient: list[float] | None = None,
) -> dict[str, Any]: ...
def transverse_profile(
    symbols: list[str],
    positions_bohr: list[list[float]],
    hessians: list[list[float]],
    gradients: list[list[float]] | None = None,
    tangent: str = "gradient",
    tangents: list[list[float]] | None = None,
    project: bool = True,
    stationary_rms: float = 3e-4,
) -> dict[str, Any]: ...

_OrbitalSet = tuple[list[list[float]], list[float], list[float]]

def native_orbitals(
    name: str,
    symbols: list[str],
    positions_bohr: list[float],
    charge: int = 0,
    multiplicity: int | None = None,
    unpaired_electrons: int | None = None,
    settings: list[tuple[str, str]] | None = None,
) -> (
    tuple[
        list[float],
        int,
        list[tuple[list[float], int, list[float], list[float]]],
        list[int] | None,
    ]
    | None
): ...
def follow_orbitals(
    overlap: list[list[float]],
    reference: _OrbitalSet,
    other: _OrbitalSet,
    degeneracy: float = 1e-5,
) -> dict[str, Any]: ...
def natural_reaction_orbitals(
    overlap: list[list[float]],
    reference: _OrbitalSet,
    plus: _OrbitalSet,
    minus: _OrbitalSet,
    step: float,
    degeneracy: float = 1e-5,
) -> dict[str, Any]: ...
def mass_weighted_displacement(
    masses_amu: list[float], direction: list[float], step: float
) -> list[float]: ...
def reactive_orbitals(
    first: list[float], last: list[float], occupations: list[float]
) -> tuple[list[float], int | None, int | None]: ...
def state_overlaps(
    first: tuple[list[list[float]], int, list[list[float]]],
    second: tuple[list[list[float]], int, list[list[float]]],
    signed: bool = False,
) -> tuple[list[list[float]], float]: ...
def diabatic_states(
    arc_lengths: list[float],
    energies: list[list[float]],
    overlaps: list[list[list[float]]],
) -> dict[str, Any]: ...
def landau_zener_stay(coupling: float, slope_difference: float, kinetic_energy: float) -> float: ...

class O1NumHess:
    def __init__(
        self,
        symbols: list[str],
        positions_bohr: list[float],
        gradient: list[float],
        *,
        cell_bohr: list[float] | None = None,
        translation: bool = True,
        rotation: bool | None = None,
        delta_bohr: float = 0.005,
        central: bool = False,
        follow_negative_modes: bool = True,
    ) -> None: ...
    def ask(self) -> list[list[float]] | None: ...
    def tell(self, gradients: list[list[float]]) -> None: ...
    def hessian(self) -> list[float]: ...
    def directions(self) -> list[float]: ...
    @property
    def gradient_count(self) -> int: ...
    @property
    def residual(self) -> float: ...
    @property
    def notes(self) -> list[str]: ...

# Structure files (mopt_rs::io::xyz).
def xyz_read(path: str | os.PathLike[str], first_only: bool = False) -> list[dict[str, Any]]: ...
def xyz_write(
    path: str | os.PathLike[str],
    # (symbols, positions in Angstrom, comment, lattice in Angstrom or None, pbc or None)
    frames: list[tuple[list[str], list[float], str, list[float] | None, list[bool] | None]],
    append: bool = False,
) -> None: ...
# The ONIOM calculator (the options -os, -oniom, -oniom_middle, -oniom_opt, -calc_opt).
class OniomCalculator:
    def __init__(
        self,
        high_atoms: str | list[int],
        low: str,
        *,
        high_calculator: str,
        links: str | list[int] | None = None,
        middle_atoms: str | list[int] | None = None,
        middle_links: str | list[int] | None = None,
        middle_calculator: str | None = None,
        link_placement: str | None = None,
        embedding: str | None = None,
        high_charge: int | None = None,
        high_multiplicity: int | None = None,
        middle_charge: int | None = None,
        middle_multiplicity: int | None = None,
        settings: dict[str, Any] | list[tuple[str, Any]] | None = None,
        calculator_settings: dict[str, Any] | list[tuple[str, Any]] | None = None,
    ) -> None: ...
    @property
    def options(self) -> list[str]: ...
    @property
    def high_calculator(self) -> str: ...
    @property
    def microiterations(self) -> bool: ...
    def evaluate(
        self,
        symbols: list[str],
        positions_bohr: list[float],
        charge: int = 0,
        multiplicity: int | None = None,
        need_hessian: bool = False,
        cell_bohr: list[float] | None = None,
        unpaired_electrons: int | None = None,
        *,
        pbc: tuple[bool, bool, bool] | None = None,
        excited_state: int = 0,
    ) -> tuple[float, list[float], list[float] | None]: ...