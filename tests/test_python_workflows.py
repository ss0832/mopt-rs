"""The AutoTS workflow assembled from the Python building blocks
(`examples/python_autots/autots_by_hand.py`, `docs/source/python_workflows.md`): it runs, it
finds what `run_autots` finds on the same input, and the page shows its code.

MultiOptPy: `Wrapper/autots.py:AutoTSWorkflow` (the command compared with).

This file is part of mopt-py (the Python interface of mopt-rs).
"""

from __future__ import annotations

import csv
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from mopt_rs import _core
from mopt_rs.cli import run_counted
from mopt_rs.workflows import relax_band
from mopt_rs.xyz import read_xyz_frames

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "python_autots"
SCRIPT = EXAMPLE / "autots_by_hand.py"
PAGE = ROOT / "docs" / "source" / "python_workflows.md"

# Both runs end every optimization on the tight criteria (largest force component 1.5e-5
# Hartree/Bohr, RMS 1e-5). Two runs that stop near the same stationary point differ in
# energy by about g^2 / (2 k): with the softest curvature of these ion-molecule complexes
# (k ~ 1e-3 Hartree/Bohr^2) that is ~1e-7 Hartree. 1e-6 Hartree (6e-4 kcal/mol) on every
# energy is ten times that, while another stationary point differs by 0.05 kcal/mol or more
# (the rotation of the hydroxide in the reactant complex).
TOLERANCE_HARTREE = 1.0e-6


def example() -> ModuleType:
    spec = importlib.util.spec_from_file_location("autots_by_hand", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered before it runs: its dataclasses look their module up.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def last_frame(path: Path) -> np.ndarray:
    """The positions (Bohr) of the last structure of an xyz file of several."""
    return read_xyz_frames(path)[-1].positions_bohr


def command_result(work: Path, script: ModuleType, product: np.ndarray) -> dict[str, float]:
    """The transition state of `run_autots` whose ends are the reactant and the product,
    from `autots_summary.csv` and the optimized ends of its IRC."""
    symbols, x0, _, _ = script.read_structure(EXAMPLE / "SN2.xyz")
    with (work / "autots_summary.csv").open(encoding="utf-8") as table:
        rows = [r for r in csv.DictReader(table) if r["imaginary_modes"] == "1"]
    for row in rows:
        stem = Path(row["file"]).stem
        ends = []
        for j in (1, 2):
            end = work / f"{stem}_results" / f"endpoint_{j}_opt.xyz"
            ends.append((script.read_structure(end)[1], float(row[f"end{j}_hartree"])))
        reactant = [e for x, e in ends if script.bonds_alike(symbols, x, x0)]
        made = [e for x, e in ends if script.bonds_alike(symbols, x, product)]
        if len(reactant) == 1 and len(made) == 1:
            return {
                "transition_state_hartree": float(row["energy_hartree"]),
                "reactant_hartree": reactant[0],
                "product_hartree": made[0],
            }
    raise AssertionError("no transition state of run_autots joins the reactant and the product")


def test_the_example_finds_what_run_autots_finds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "by_hand"
    done = subprocess.run(
        [sys.executable, str(SCRIPT), str(out)], capture_output=True, text=True, timeout=600
    )
    assert done.returncode == 0, done.stdout + done.stderr
    found = json.loads((out / "results.json").read_text(encoding="utf-8"))
    # The script writes results.json only for a transition state with one imaginary mode
    # whose optimized IRC ends are minima with the bonds of the reactant and the product.
    assert found["imaginary_wavenumber_cm1"] < 0.0
    assert found["barrier_kcal_mol"] > 0.0 > found["reaction_energy_kcal_mol"]

    monkeypatch.chdir(tmp_path)
    arguments = [str(EXAMPLE / "SN2.xyz"), "-cfg", str(EXAMPLE / "autots_config.json")]
    code, counts = run_counted("run_autots", [*arguments, "--run_step4"])
    assert code == 0 and counts.gradients > 0
    script = example()
    product = last_frame(out / "step1_push.xyz")
    command = command_result(tmp_path / "autots_command", script, product)
    for key in ("transition_state_hartree", "reactant_hartree", "product_hartree"):
        assert abs(found[key] - command[key]) < TOLERANCE_HARTREE, (key, found[key], command[key])
    barrier = command["transition_state_hartree"] - command["reactant_hartree"]
    reaction = command["product_hartree"] - command["reactant_hartree"]
    assert abs(found["barrier_hartree"] - barrier) < 2 * TOLERANCE_HARTREE
    assert abs(found["reaction_energy_hartree"] - reaction) < 2 * TOLERANCE_HARTREE
    kcal = _core.HARTREE_TO_KCAL_MOL
    assert abs(found["barrier_kcal_mol"] - kcal * barrier) < 2 * kcal * TOLERANCE_HARTREE


def test_the_script_refuses_a_point_that_is_not_a_first_order_saddle() -> None:
    script = example()
    symbols, x0, charge, multiplicity = script.read_structure(EXAMPLE / "SN2.xyz")
    engine = script.CountedEngine("gfn1xtb", symbols, charge, multiplicity)
    # The reactant complex, minimized, has no imaginary mode at all.
    minimum, _ = script.minimum(engine, x0, "reactant")
    with pytest.raises(script.WorkflowError, match="0 imaginary modes"):
        script.first_order_saddle(engine, minimum, "reactant")


def test_candidates_are_the_maxima_of_the_chain_highest_first() -> None:
    script = example()
    assert script.candidates([0.0, 2.0, 1.0, 3.0, 0.5], top_n=3) == [3, 1]
    assert script.candidates([0.0, 2.0, 1.0, 3.0, 0.5], top_n=1) == [3]
    # No maximum: the highest inner image.
    assert script.candidates([0.0, 1.0, 2.0, 3.0], top_n=3) == [2]


def test_every_code_block_of_the_page_is_code_of_the_script() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    page = PAGE.read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)```", page, flags=re.DOTALL)
    assert len(blocks) >= 8
    for block in blocks:
        assert block in text, block


def test_relax_band_spaces_the_first_chain() -> None:
    def surface(x: list[float], image: int, hessian: bool) -> tuple[float, list[float], None]:
        return x[1] ** 2, [0.0, 2 * x[1]], None

    line = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]
    result = relax_band(surface, line, spacing=0.5, max_iterations=1)
    xs = np.asarray(result["images"])[:, 0]
    assert len(xs) == 7
    assert np.allclose(np.diff(xs), 0.5)
    with pytest.raises(ValueError, match="spacing"):
        relax_band(surface, line, spacing=0.0)


def test_the_criterion_of_an_imaginary_mode_is_that_of_the_library() -> None:
    # The mode kill and the IRC count a mode as imaginary below minus this curvature.
    assert _core.IMAGINARY_CURVATURE == 1e-8
