"""Trajectory analysis from Python: frames in memory, the files of an ``mdmain`` run, and
the block-average statistics of a correlated series.

MultiOptPy: no counterpart (new in mopt-rs).

This file is part of mopt-py (the Python interface of mopt-rs).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from mopt_rs import TrajectoryAnalysis, _core
from mopt_rs.md import (
    analyze_md_run,
    autocorrelation,
    block_average,
    green_kubo,
    infrared_spectrum,
    power_spectrum,
)

LIGHT_CM_PER_FS = 2.99792458e-5


def fcc(a: float, k: int) -> list[float]:
    basis = [(0.0, 0.0, 0.0), (0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5)]
    out: list[float] = []
    for x in range(k):
        for y in range(k):
            for z in range(k):
                for b in basis:
                    out += [(x + b[0]) * a, (y + b[1]) * a, (z + b[2]) * a]
    return out


def test_frames_in_memory() -> None:
    # A crystal: twelve neighbours at a / sqrt 2.
    a = 3.6
    crystal = fcc(a, 3)
    n = len(crystal) // 3
    side = 3 * a
    box = [side, 0, 0, 0, side, 0, 0, 0, side]
    t = TrajectoryAnalysis(["Cu"] * n, [crystal, crystal], [0.0, 1.0], cells_angstrom=[box])
    assert len(t) == 2 and t.periodic and not t.has_velocities
    rdf = t.rdf()
    pair = rdf["pairs"]["Cu-Cu"]
    k = int((a / math.sqrt(2) + 0.05) / rdf["bin_width"])
    assert abs(pair["coordination"][k] - 12.0) < 1e-9
    # Two atoms vibrating at 1500 cm^-1.
    omega = 2 * math.pi * LIGHT_CM_PER_FS * 1500.0
    frames, velocities = [], []
    for step in range(3000):
        x = 0.05 * math.sin(omega * step)
        v = 0.05 * omega * math.cos(omega * step)
        frames.append([0.0, 0.0, -x, 0.0, 0.0, 0.74 + x])
        velocities.append([0.0, 0.0, -v, 0.0, 0.0, v])
    times = [float(s) for s in range(3000)]
    h2 = TrajectoryAnalysis(["H", "H"], frames, times, velocities_angstrom_per_fs=velocities)
    vdos = h2.vdos(max_wavenumber=4000.0)
    peak = vdos["wavenumbers"][int(np.argmax(vdos["total"]))]
    assert abs(peak - 1500.0) <= 2.0
    distance = h2.geometry([[0, 1]])["distance 1-2 (ang.)"]
    assert abs(min(distance) - 0.64) < 1e-6 and abs(max(distance) - 0.84) < 1e-6
    pca = h2.pca(2)
    assert pca["contributions"][0] > 0.999
    assert h2.fluctuations()["rmsd"][0] == 0.0
    assert len(h2.skip(1000.0)) == 2000
    with pytest.raises(ValueError):
        h2.geometry([[0]])


def test_correlations_spectra_and_green_kubo() -> None:
    rng = np.random.default_rng(3)
    # The fast and the direct estimators agree.
    x = np.sin(0.21 * np.arange(500)) + 0.2 * rng.standard_normal(500)
    fast = autocorrelation(x.tolist(), 100)
    direct = autocorrelation(x.tolist(), 100, direct=True)
    assert np.allclose(fast, direct, atol=1e-10)
    # A cosine at 1500 cm^-1 sampled every 0.5 fs: the power spectrum peaks there.
    dt = 0.5
    omega = 2 * math.pi * LIGHT_CM_PER_FS * 1500.0
    series = np.cos(omega * dt * np.arange(4000)).tolist()
    for window in ["hann", "blackman"]:
        s = power_spectrum(series, dt, window=window)
        peak = s["wavenumbers"][int(np.argmax(s["values"]))]
        assert abs(peak - 1500.0) < 2 * (s["wavenumbers"][1] - s["wavenumbers"][0])
    ir = infrared_spectrum(autocorrelation(series, 2000), dt, correction="standard")
    assert max(ir["values"]) == pytest.approx(1.0)
    # Ornstein-Uhlenbeck with a correlation time of 20 fs: the integral is 20 fs.
    a = math.exp(-1.0 / 20.0)
    y, ou = 0.0, []
    for _ in range(100000):
        y = a * y + math.sqrt(1 - a * a) * rng.standard_normal()
        ou.append(y)
    fs_per_au = 0.024188843265857
    g = green_kubo([ou], 1.0, prefactor=fs_per_au, lags=300)
    assert abs(g["plateau_mean"] / 20.0 - 1.0) < 0.2
    assert g["value_error"] > 0.0


def test_block_averages_of_a_correlated_series() -> None:
    rng = np.random.default_rng(1)
    x, series = 0.0, []
    for _ in range(32768):
        x = 0.8 * x + rng.standard_normal()
        series.append(x)
    s = block_average(series)
    # AR(1): statistical inefficiency (1 + phi) / (1 - phi) = 9.
    assert s["converged"] and 6.0 < s["inefficiency"] < 12.0
    assert s["error"] > 2.5 * s["naive_error"]


def test_the_files_of_an_mdmain_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    cluster = tmp_path / "ar7.xyz"
    shell = [
        (0.0, 0.0, 0.0),
        (3.8, 0.0, 0.0),
        (-3.8, 0.0, 0.0),
        (0.0, 3.8, 0.0),
        (0.0, -3.8, 0.0),
        (0.0, 0.0, 3.8),
        (0.0, 0.0, -3.8),
    ]
    cluster.write_text(
        f"{len(shell)}\n\n" + "".join(f"Ar {x} {y} {z}\n" for x, y, z in shell),
        encoding="utf-8",
    )
    arguments = [str(cluster), "-os", "lj", "-mt", "bussi", "-time", "400", "-ts", "4fs"]
    options = ["-temp", "30", "-md_opt", "seed=2", "write_every=2", "-analysis", "all"]
    options += ["atoms=1,2", "figures=svg"]
    assert _core.run_cli("mdmain", arguments + options) == 0
    run = next(p for p in tmp_path.rglob("ar7_MD_ASE_*") if p.is_dir())
    out = run / "trajectory_analysis"
    for name in ["statistics.csv", "rdf.csv", "msd.csv", "vdos.csv", "rmsd.csv", "pca.csv"]:
        assert (out / name).exists(), name
    assert (out / "rdf.svg").exists() and (out / "geometry_1.svg").exists()
    t = TrajectoryAnalysis.from_file(run)
    assert t.has_velocities and len(t) == 201
    assert t.times_fs[1] == pytest.approx(8.0)
    again = tmp_path / "again"
    written = analyze_md_run(run, ["rdf", "statistics", "skip=100"], out_dir=again)
    assert {Path(p).name for p in written} == {
        "rdf.csv",
        "rdf.png",
        "statistics.csv",
        "blocking.csv",
    }
    temperature = next(
        line
        for line in (again / "statistics.csv").read_text(encoding="utf-8").splitlines()
        if line.startswith("temperature_K")
    )
    mean = float(temperature.split(",")[3])
    assert 10.0 < mean < 60.0
