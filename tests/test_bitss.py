"""Binary-image transition state search from Python.

MultiOptPy: `ModelFunction/binary_image_ts_search_model_function.py:BITSSModelFunction`.

This file is part of mopt-py (the Python interface of mopt-rs).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

import mopt_rs
from mopt_rs import Bitss, NativeObjective, bitss_search
from mopt_rs.bitss import bitss_coefficients, bitss_function

MB_MINIMA = [(-0.558224, 1.441726), (0.623499, 0.028038), (-0.050011, 0.466694)]
MB_SADDLES = [(-0.822002, 0.624313), (0.212487, 0.292988)]


def test_built_in_surface_both_saddles() -> None:
    for first, second, saddle in [
        (MB_MINIMA[1], MB_MINIMA[2], MB_SADDLES[1]),
        (MB_MINIMA[0], MB_MINIMA[2], MB_SADDLES[0]),
        # Across the intermediate minimum: the higher of the two saddles.
        (MB_MINIMA[0], MB_MINIMA[1], MB_SADDLES[0]),
    ]:
        r = bitss_search("mb2d", first, second, max_iterations=2000)
        assert r.converged and r.end == "midpoint_gradient", r.end
        assert np.linalg.norm(r.transition_state - np.asarray(saddle)) < 5e-3
        assert r.value is not None
        assert r.stages >= 4 and r.iterations > 0
        assert r.gradient_evaluations > r.value_evaluations > 0
        # The images ended on either side of the estimate.
        assert np.allclose(0.5 * (r.first + r.second), r.transition_state)


def test_python_function_with_exact_hessians_and_callback() -> None:
    surface = NativeObjective("mb2d")
    calls = {"hessian": 0}

    def fun(x: np.ndarray) -> tuple[float, np.ndarray]:
        value, g, _ = surface.evaluate(x)
        return value, g

    def hessian(x: np.ndarray) -> np.ndarray:
        calls["hessian"] += 1
        h = surface.evaluate(x, need_hessian=True)[2]
        assert h is not None
        return h

    r = bitss_search(
        fun,
        MB_MINIMA[1],
        MB_MINIMA[2],
        methods="rsirfo_bfgs",
        exact_hessian_interval=5,
        hessian=hessian,
        max_iterations=2000,
    )
    assert r.converged
    assert calls["hessian"] > 0
    assert np.linalg.norm(r.transition_state - np.asarray(MB_SADDLES[1])) < 5e-3

    seen: list[dict[str, Any]] = []

    def stop(report: dict[str, Any]) -> bool:
        seen.append(report)
        return len(seen) == 3

    stopped = bitss_search(fun, MB_MINIMA[1], MB_MINIMA[2], callback=stop)
    assert stopped.end is None and not stopped.converged
    assert len(stopped.reports) == 3


def test_ask_tell_and_the_function_alone() -> None:
    surface = NativeObjective("mb2d")
    search = Bitss(list(MB_MINIMA[1]), list(MB_MINIMA[2]), methods=["lbfgs"], reduction=0.25)
    assert search.dimension == 2 and search.stage == 0 and search.end is None
    start = float(np.linalg.norm(np.subtract(MB_MINIMA[1], MB_MINIMA[2])))
    assert search.target_separation == pytest.approx(start)
    points = search.ask()
    assert points is not None
    # The first survey: both images with gradients and three points between them.
    assert [p["role"] for p in points] == ["first", "second", "between", "between", "between"]
    assert [p["share"] for p in points[2:]] == [0.25, 0.5, 0.75]
    assert [p["gradient"] for p in points] == [True, True, False, False, False]
    values: list[tuple[float, list[float] | None, list[float] | None]] = []
    gradients: list[list[float]] = []
    for p in points:
        value, g, _ = surface.evaluate(np.asarray(p["x"]))
        values.append((value, g.tolist() if p["gradient"] else None, None))
        gradients.append(g.tolist())
    report = search.tell(values)
    assert report["new_stage"] and report["stage"] == 1
    assert report["target_separation"] == pytest.approx(0.75 * start)
    assert report["end"] is None and "midpoint_energy" in report

    # The reported value and gradient are the BITSS function with the reported coefficients.
    value, gradient = bitss_function(
        report["kappa_e"],
        report["kappa_d"],
        report["target_separation"],
        report["first"],
        report["second"],
        (report["energies"][0], report["energies"][1]),
        gradients[0],
        gradients[1],
    )
    assert value == pytest.approx(report["value"], rel=1e-12)
    assert np.allclose(gradient, report["gradient"], rtol=1e-12, atol=1e-15)

    # Eqs. (4) and (5) of the paper.
    kappa_e, kappa_d = bitss_coefficients(10.0, 0.1, 2.0, 0.04, (0.03, 0.04))
    assert kappa_e == pytest.approx(125.0)
    assert kappa_d == pytest.approx(0.1)

    with pytest.raises(Exception, match="values for"):
        search.tell([])
    with pytest.raises(Exception, match="coincide"):
        Bitss([0.0, 0.0], [0.0, 0.0])
    with pytest.raises(ValueError, match="variables"):
        bitss_search("mb2d", [0.0, 0.0], [1.0])


def test_command_with_two_structure_files(tmp_path: Path) -> None:
    rise = 2.55 * 0.75**0.5
    base = [(0.0, 0.0, 0.0), (2.55, 0.0, 0.0), (1.275, rise, 0.0), (3.825, rise, 0.0)]

    def write(name: str, adatom: tuple[float, float, float]) -> str:
        lines = ["5", ""] + [f"Cu {x} {y} {z}" for x, y, z in [*base, adatom]]
        path = tmp_path / name
        path.write_text("\n".join(lines) + "\n")
        return str(path)

    start = write("adatom.xyz", (1.275, rise / 3.0, 2.08))
    other = write("other.xyz", (2.55, 2.0 * rise / 3.0, 2.08))
    results = mopt_rs.run_optmain(
        [start, "-os", "emt", "-opt", "lbfgs", "-fix", "1-4", "-mf", "bitss", other, "-ns", "2000"],
        console=False,
    )
    assert len(results) == 1
    job = results[0]
    assert job["optimized_flag"]
    guess = Path(job["folder_directory"]) / "adatom_ts_guess.xyz"
    assert guess.exists()
    assert len(job["element_list"]) == 5
    x, y, _ = (float(v) for v in guess.read_text().splitlines()[-1].split()[1:])
    # Above the middle of the edge the two triangles share.
    assert abs(x - 1.9125) < 0.02 and abs(y - 0.5 * rise) < 0.02
