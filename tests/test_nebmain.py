"""`nebmain` from Python: the command with the built-in objectives and calculators.

MultiOptPy: `neb.py:NEB`.

This file is part of mopt-py (the Python interface of mopt-rs).
"""

from __future__ import annotations

from pathlib import Path

import mopt_rs

HCN = """3
HCN
C 0.000 0.000 0.000
N 1.156 0.000 0.000
H -1.066 0.000 0.000
3
bridged
C 0.000 0.000 0.000
N 1.190 0.000 0.000
H 0.368 1.132 0.000
3
HNC
C 0.000 0.000 0.000
N 1.170 0.000 0.000
H 2.165 0.000 0.000
"""


def run_directory(base: Path) -> Path:
    runs = [p for p in base.iterdir() if p.is_dir() and "_NEB_" in p.name]
    assert len(runs) == 1
    return runs[0]


def test_a_chain_on_the_muller_brown_surface(tmp_path: Path) -> None:
    start = tmp_path / "start.txt"
    end = tmp_path / "end.txt"
    start.write_text("x -0.558\ny 1.442\n")
    end.write_text("x 0.623\ny 0.028\n")
    arguments = [str(start), str(end), "-os", "mb2d", "-p", "15", "-fe", "3", "-ns", "80"]
    code = mopt_rs._core.run_cli("nebmain", [*arguments, "-neb_opt", "tolerance=1e-6"])
    assert code == 0
    run = run_directory(tmp_path)
    energies = (run / "energy_plot.csv").read_text().splitlines()
    assert 10 < len(energies) < 60
    assert len(energies[-1].split(",")) == 17
    # The two saddles of the surface are the highest images of the last chain.
    last = sorted(float(v) for v in energies[-1].split(","))
    assert abs(last[-1] + 0.040665) < 1e-5
    for name in [
        "path_ITR_0_start/start_path.csv",
        "path_ITR_0_start/approx_TS_node.txt",
        "tmp_hessian_0.npy",
        "energy_profile.png",
        "chain_on_surface.svg",
        "mopt.log",
        "settings.json",
    ]:
        assert (run / name).is_file(), name
    assert "The chain is relaxed" in (run / "mopt.log").read_text()


def test_structures_with_fixed_ends_and_what_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "hcn.xyz"
    path.write_text(HCN)
    common = [str(path), "-os", "gfn0xtb", "-neb_opt", "figures=off"]
    assert mopt_rs._core.run_cli("nebmain", [*common, "-ns", "3", "-fe", "3"]) == 0
    run = run_directory(tmp_path)
    assert len((run / "energy_plot.csv").read_text().splitlines()) == 3
    first = (run / "path_ITR_0_hcn" / "hcn_0.xyz").read_text()
    last = (run / "path_ITR_2_hcn" / "hcn_0.xyz").read_text()
    assert first.splitlines()[0] == "3"
    assert first.splitlines()[1] == "0 1"
    # The fixed first image keeps its shape (the structures are turned onto each other).
    assert len(first.splitlines()) == len(last.splitlines()) == 5
    assert (run / "path_ITR_2_hcn" / "hcn_path.xyz").is_file()
    assert not (run / "energy_profile.png").exists()
    # Options that are not implemented yet end the command with the usage error code.
    assert mopt_rs._core.run_cli("nebmain", [*common, "-dneb"]) == 2
