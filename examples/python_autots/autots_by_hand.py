"""The AutoTS workflow assembled from the Python building blocks of mopt-rs.

The four steps of ``run_autots`` (AFIR push, chain relaxation, saddle optimization with a
vibrational analysis, IRC with the optimization of its ends), each made with the Python
counterpart of the command that ``run_autots`` runs for it (``optmain -ma``, ``nebmain``,
``optmain -order 1`` with ``-freq``, ``optmain -irc``). The page
``docs/source/python_workflows.md`` explains every step; its code blocks are parts of this
file (``tests/test_python_workflows.py`` checks that they still are).

    python autots_by_hand.py OUTPUT_DIRECTORY [--input SN2.xyz] [--engine gfn1xtb]

Writes into OUTPUT_DIRECTORY the path of the push, the relaxed chain, the transition
state, and the optimized ends of its IRC (xyz, Angstrom), ``results.json`` with the
numbers, and prints a summary. Exits with 1 when an optimization does not converge or a
result is not what the next step needs (no first-order saddle point, an end of the IRC
that is not a minimum, no IRC that joins the reactant with the product), with a message
that says which.

MultiOptPy: `Wrapper/autots.py:AutoTSWorkflow` (the steps); the building blocks are new.

This file is part of mopt-rs.
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mopt_rs import _core
from mopt_rs.irc import irc_path
from mopt_rs.optimize import Evaluation, NativeCalculator, Optimizer, Task, run
from mopt_rs.orient import compare_structures
from mopt_rs.workflows import relax_band
from mopt_rs.xyz import read_xyz, write_xyz

HERE = Path(__file__).resolve().parent
Frames = list[np.ndarray]


class WorkflowError(Exception):
    """A step that did not give what the next step needs."""


# --- structures ---
def read_structure(path: Path) -> tuple[list[str], np.ndarray, int, int]:
    """Symbols, positions (Bohr, shape (N, 3)), charge and multiplicity of the first
    structure of an xyz file, read as the commands read it (``charge multiplicity`` on the
    second line; 0 and 1 when the line says something else)."""
    frame = read_xyz(path)
    charge, multiplicity = frame.state or (0, 1)
    return frame.symbols, frame.positions_bohr, charge, multiplicity


def write_frames(path: Path, symbols: list[str], frames: Frames, comments: list[str]) -> None:
    """Frames (Bohr) as an xyz file in Angstrom, one comment line each, written as the
    commands write their trajectories."""
    rows = [
        (symbols, np.reshape(x, (-1, 3)) * _core.BOHR_TO_ANGSTROM, comment)
        for x, comment in zip(frames, comments, strict=True)
    ]
    write_xyz(path, rows)


# --- the surface ---
class CountedEngine:
    """A built-in calculator that counts what it is asked for."""

    def __init__(self, name: str, symbols: list[str], charge: int, multiplicity: int) -> None:
        self.calculator = NativeCalculator(name)
        self.symbols, self.charge, self.multiplicity = symbols, charge, multiplicity
        self.gradients = 0
        self.hessians = 0

    def evaluate(self, task: Task) -> Evaluation:
        """What :func:`mopt_rs.optimize.run` asks for."""
        self.gradients += 1
        self.hessians += int(task.need_hessian)
        return self.calculator.evaluate(task)

    def at(self, x: np.ndarray, need_hessian: bool = False) -> Evaluation:
        """Energy, gradient (and Hessian) at the positions ``x`` (Bohr)."""
        task = Task(
            label="main",
            calculator=self.calculator.name,
            symbols=self.symbols,
            positions_bohr=np.reshape(x, (-1, 3)),
            charge=self.charge,
            multiplicity=self.multiplicity,
            need_hessian=need_hessian,
        )
        return self.evaluate(task)


def converged_point(
    name: str, optimizer: Optimizer, engine: CountedEngine
) -> tuple[np.ndarray, float, Frames]:
    """Run ``optimizer`` to its end: the structure and energy where it converged, and the
    structures it evaluated. A run that did not converge ends the workflow."""
    result = run(optimizer, engine)
    if not result.converged:
        raise WorkflowError(
            f"{name}: the optimization ended with '{result.termination}' after "
            f"{result.iterations} iterations, not on its criteria"
        )
    frames = [np.asarray(r["positions_bohr"], dtype=np.float64) for r in result.reports]
    return frames[-1], float(result.reports[-1]["energy"]), frames


# --- step 1 ---
def push_reactants(
    engine: CountedEngine, x0: np.ndarray, gamma: float, a: list[int], b: list[int]
) -> Frames:
    """``optmain -ma GAMMA A B -opt rsirfo_block_fsb -modelhess fischerd3``: a minimization
    with the artificial force between the atoms ``a`` and ``b`` (numbers from 1); the
    structures it evaluated are the path."""
    push = Optimizer(
        engine.symbols,
        x0.ravel().tolist(),
        charge=engine.charge,
        multiplicity=engine.multiplicity,
        methods=["rsirfo_block_fsb"],
        model_hessian="fischerd3",
    )
    push.add_afir(gamma, a, b)
    _, _, frames = converged_point("step 1 (AFIR push)", push, engine)
    return frames


# --- step 2 ---
def relax_path(
    engine: CountedEngine, frames: Frames, spacing_angstrom: float, iterations: int
) -> tuple[Frames, list[float]]:
    """``nebmain``: the chain from the structures of the push, relaxed on the surface
    alone (the artificial force of step 1 is not part of it)."""

    def surface(x: list[float], image: int, need_hessian: bool) -> tuple[float, list[float], None]:
        found = engine.at(np.asarray(x), need_hessian)
        return found.energy, np.ravel(found.gradient).tolist(), None

    relaxed = relax_band(
        surface,
        [f.tolist() for f in frames],
        symbols=engine.symbols,
        spacing=spacing_angstrom * _core.ANGSTROM_TO_BOHR,
        max_iterations=iterations,
        tolerance=3e-4,
    )
    if relaxed["termination"] != "Converged":
        print(f"step 2: the chain ended with {relaxed['termination']}; its maxima go on")
    return [np.asarray(x) for x in relaxed["images"]], [float(e) for e in relaxed["energies"]]


def candidates(energies: list[float], top_n: int) -> list[int]:
    """The images that ``run_autots`` optimizes: the inner images above both neighbours
    (the highest inner image when there is none), highest first, at most ``top_n``."""
    inner = range(1, len(energies) - 1)
    maxima = [i for i in inner if energies[i] > energies[i - 1] and energies[i] > energies[i + 1]]
    if not maxima:
        maxima = [max(inner, key=lambda i: energies[i])]
    return sorted(maxima, key=lambda i: energies[i], reverse=True)[:top_n]


# --- step 3 ---
def saddle_point(engine: CountedEngine, guess: np.ndarray, name: str) -> tuple[np.ndarray, float]:
    """``optmain -order 1 -opt rsirfo_block_bofill -fc 5 -tcc`` with a largest trust radius
    of 0.2: the exact Hessian at the first iteration, which sets the mode to climb, and at
    every fifth."""
    ts = Optimizer(
        engine.symbols,
        guess.ravel().tolist(),
        charge=engine.charge,
        multiplicity=engine.multiplicity,
        methods=["rsirfo_block_bofill"],
        saddle_order=1,
        exact_hessian_interval=5,
        criteria="tight",
        max_trust_radius=0.2,
    )
    x, energy, _ = converged_point(name, ts, engine)
    return x, energy


# --- step 4 ---
def imaginary_wavenumbers(engine: CountedEngine, x: np.ndarray) -> list[float]:
    """``-freq``: the wavenumbers (cm^-1) of the imaginary modes, after translations and
    rotations are projected out. A mode is imaginary when its curvature (mass-weighted) is
    below ``-IMAGINARY_CURVATURE`` (about 0.5 cm^-1, the numerical flatness of the
    projected motions), whatever the size of its wavenumber."""
    found = engine.at(x, need_hessian=True)
    analysis = _core.vibrational_analysis(
        engine.symbols,
        x.ravel().tolist(),
        np.ravel(found.hessian).tolist(),
        found.energy,
        engine.charge,
        engine.multiplicity,
    )
    modes = analysis["modes"]
    return [
        w
        for w, k in zip(modes["wavenumbers"], modes["curvatures"], strict=True)
        if k < -_core.IMAGINARY_CURVATURE
    ]


def first_order_saddle(engine: CountedEngine, x: np.ndarray, name: str) -> float:
    """The imaginary wavenumber of ``x``; a structure without exactly one imaginary mode
    ends the use of this candidate."""
    imaginary = imaginary_wavenumbers(engine, x)
    if len(imaginary) != 1:
        listed = ", ".join(f"{w:.1f}" for w in imaginary)
        raise WorkflowError(
            f"{name}: {len(imaginary)} imaginary modes ({listed} cm^-1); "
            "a transition state has exactly one"
        )
    return imaginary[0]


# --- step 5 ---
def follow_irc(engine: CountedEngine, x_ts: np.ndarray) -> Frames:
    """``optmain -irc 0.5 200 lqa -fc 10``: both sides of the IRC from the transition
    state; the last point of each side."""

    def energy_and_gradient(x: np.ndarray) -> tuple[float, np.ndarray]:
        found = engine.at(x)
        return found.energy, np.ravel(found.gradient)

    def hessian(x: np.ndarray) -> np.ndarray:
        return np.asarray(engine.at(x, need_hessian=True).hessian)

    path = irc_path(
        energy_and_gradient,
        x_ts.ravel(),
        symbols=engine.symbols,
        step_size=0.5,
        max_steps=200,
        method="lqa",
        exact_hessian_interval=10,
        hessian=hessian,
    )
    if path.kind != "transition_state":
        raise WorkflowError(f"step 5: the IRC started as '{path.kind}', not at a saddle point")
    ends = []
    for branch in ("forward", "backward"):
        side = path.side(branch)
        if side is None or len(side.points) == 0:
            raise WorkflowError(f"step 5: the {branch} side of the IRC has no points")
        ends.append(side.points[-1])
    return ends


def minimum(engine: CountedEngine, x: np.ndarray, name: str) -> tuple[np.ndarray, float]:
    """The end of a side optimized to a minimum (``step4b_opt_method``, ``-fc 10 -tcc``).
    The mode kill (``-modekill``) steps off a stationary point that has an imaginary mode,
    and the vibrational analysis confirms that the point is a minimum."""
    end = Optimizer(
        engine.symbols,
        x.ravel().tolist(),
        charge=engine.charge,
        multiplicity=engine.multiplicity,
        methods=["rsirfo_block_fsb"],
        exact_hessian_interval=10,
        criteria="tight",
        mode_kill=True,
    )
    x_min, energy, _ = converged_point(name, end, engine)
    imaginary = imaginary_wavenumbers(engine, x_min)
    if imaginary:
        listed = ", ".join(f"{w:.1f}" for w in imaginary)
        raise WorkflowError(f"{name}: {len(imaginary)} imaginary modes ({listed} cm^-1)")
    return x_min, energy


# --- step 6 ---
def bonds_alike(symbols: list[str], a: np.ndarray, b: np.ndarray) -> bool:
    """Whether two structures have the same bonds (their bond graphs agree up to a
    permutation of equivalent atoms)."""
    found = compare_structures(symbols, np.ravel(a).tolist(), np.ravel(b).tolist())
    return bool(found["same_bonds"])


def same_structure(symbols: list[str], a: np.ndarray, b: np.ndarray) -> bool:
    """The same bonds and, after matching and superposing, an RMSD below 0.01 Bohr."""
    found = compare_structures(symbols, np.ravel(a).tolist(), np.ravel(b).tolist())
    return bool(found["same_bonds"]) and float(found["rmsd"]) < 1.0e-2


@dataclass
class Outcome:
    """The numbers of a finished workflow (Hartree, cm^-1)."""

    reactant: float
    transition_state: float
    product: float
    imaginary_wavenumber: float
    gradients: int
    hessians: int

    def report(self) -> dict[str, float | int]:
        """Energies in Hartree, the barrier and the reaction energy also in kcal/mol."""
        kcal = _core.HARTREE_TO_KCAL_MOL
        barrier = self.transition_state - self.reactant
        reaction = self.product - self.reactant
        return {
            "reactant_hartree": self.reactant,
            "transition_state_hartree": self.transition_state,
            "product_hartree": self.product,
            "barrier_hartree": barrier,
            "barrier_kcal_mol": barrier * kcal,
            "reaction_energy_hartree": reaction,
            "reaction_energy_kcal_mol": reaction * kcal,
            "imaginary_wavenumber_cm1": self.imaginary_wavenumber,
            "gradients": self.gradients,
            "hessians": self.hessians,
        }


# --- the workflow ---
def autots(input_file: Path, out: Path, engine_name: str) -> dict[str, float | int]:
    """The six steps on the structure of ``input_file``; the numbers of the reaction."""
    symbols, x0, charge, multiplicity = read_structure(input_file)
    engine = CountedEngine(engine_name, symbols, charge, multiplicity)

    # Step 1: push C (atom 1) and O (atom 6) together with 150 kJ/mol.
    pushed = push_reactants(engine, x0, 150.0, [1], [6])
    pushes = [f"push {k}" for k in range(len(pushed))]
    write_frames(out / "step1_push.xyz", symbols, pushed, pushes)
    reactant_ref, product_ref = x0, pushed[-1]
    print(f"step 1: {len(pushed)} structures from the reactants to the pushed product")

    # Step 2: the chain on the surface alone; its maxima are the candidates.
    images, energies = relax_path(engine, pushed, spacing_angstrom=0.5, iterations=30)
    labels = [f"image {k} E={e:.10f}" for k, e in enumerate(energies)]
    write_frames(out / "step2_chain.xyz", symbols, images, labels)
    chosen = candidates(energies, top_n=3)
    print(f"step 2: {len(images)} images, candidates {chosen}")

    # Steps 3 and 4: each candidate to a saddle point; only a first-order saddle goes on.
    found: list[tuple[np.ndarray, float, float]] = []
    for k, image in enumerate(chosen, start=1):
        name = f"step 3 (candidate {k}, image {image})"
        try:
            x_ts, e_ts = saddle_point(engine, images[image], name)
            wavenumber = first_order_saddle(engine, x_ts, name)
        except WorkflowError as reason:
            print(f"{reason}: not taken further")
            continue
        if any(abs(e - e_ts) < 1e-6 and same_structure(symbols, x, x_ts) for x, e, _ in found):
            print(f"{name}: the same transition state as an earlier candidate")
            continue
        print(f"{name}: E = {e_ts:.10f} Hartree, one imaginary mode, {wavenumber:.1f} cm^-1")
        found.append((x_ts, e_ts, wavenumber))
    if not found:
        raise WorkflowError("steps 3-4: no candidate converged to a first-order saddle point")

    # Steps 5 and 6: the IRC of each transition state; the one whose ends are the reactant
    # and the product is the reaction.
    for k, (x_ts, e_ts, wavenumber) in enumerate(found, start=1):
        sides = follow_irc(engine, x_ts)
        ends = [minimum(engine, x, f"step 5 (TS {k}, IRC end {j})") for j, x in enumerate(sides)]
        write_frames(out / f"ts{k}.xyz", symbols, [x_ts], [f"TS E={e_ts:.10f}"])
        comments = [f"end E={e:.10f}" for _, e in ends]
        write_frames(out / f"ts{k}_ends.xyz", symbols, [x for x, _ in ends], comments)
        reactant = [e for x, e in ends if bonds_alike(symbols, x, reactant_ref)]
        product = [e for x, e in ends if bonds_alike(symbols, x, product_ref)]
        if len(reactant) == 1 and len(product) == 1:
            numbers = Outcome(
                reactant[0], e_ts, product[0], wavenumber, engine.gradients, engine.hessians
            ).report()
            (out / "results.json").write_text(json.dumps(numbers, indent=2) + "\n")
            return numbers
        print(f"TS {k}: its IRC does not join the reactant with the product")
    raise WorkflowError("step 6: no transition state joins the reactant with the product")


def main(argv: list[str] | None = None) -> int:
    """The command line of the example; 0 when the workflow found the reaction."""
    parser = argparse.ArgumentParser(description="AutoTS from the Python building blocks")
    parser.add_argument("output", type=Path, help="directory for the results")
    parser.add_argument("--input", type=Path, default=HERE / "SN2.xyz")
    parser.add_argument("--engine", default="gfn1xtb", help="a built-in calculator")
    args = parser.parse_args(argv)
    try:
        args.output.mkdir(parents=True, exist_ok=True)
        numbers = autots(args.input, args.output, args.engine)
    except (WorkflowError, OSError, ValueError, RuntimeError) as reason:
        print(f"autots_by_hand: {reason}", file=sys.stderr)
        return 1
    for key, value in numbers.items():
        print(f"{key:28} {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
