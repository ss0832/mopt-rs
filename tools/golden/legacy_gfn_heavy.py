"""Reference Hessians of MultiOptPy's GFN-FF-like and GFN0-like model Hessians for elements
beyond H, C, N, O, F, S, Cl, Br, I (``legacy_gfnff`` and ``legacy_gfn0xtb`` of mopt-rs).

Usage (a Python with numpy and scipy, MultiOptPy 1.21.2 as a source tree):

    python -I tools/golden/legacy_gfn_heavy.py <multioptpy_root> <output_json>

For each molecule, ``ApproxHessian().main`` of MultiOptPy computes the Hessian (Hartree/Bohr^2)
with the fixed gradient of ``tools/golden/model_hessians.py``. The molecules cover the element
classes the two models take from their tables (MultiOptPy's ``element_params`` of
``Parameters/gfnff_param.py``: H to Rn; ``rad`` and ``en`` of ``Parameters/gfn0xtb_param.py``:
H to Xe): alkali metal, main group of the third to sixth periods, transition metals of the
fourth to sixth periods, lanthanide, noble gas.

This file is part of mopt-rs.
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

import contextlib
import io
import json
import platform
import sys

import numpy as np

BOHR = 0.52917721067

MOLECULES = {
    "lif": (["Li", "F"], [[0.0, 0.0, 0.0], [1.56, 0.0, 0.0]]),
    "phosphine": (
        ["P", "H", "H", "H"],
        [[0.0, 0.0, 0.0], [1.42, 0.0, 0.0], [-0.27, 1.39, 0.0], [-0.27, -0.45, 1.32]],
    ),
    "titanium_tetrachloride": (
        ["Ti", "Cl", "Cl", "Cl", "Cl"],
        [
            [0.0, 0.0, 0.0],
            [1.25, 1.25, 1.25],
            [1.25, -1.25, -1.25],
            [-1.25, 1.25, -1.25],
            [-1.25, -1.25, 1.25],
        ],
    ),
    "stannane": (
        ["Sn", "H", "H", "H", "H"],
        [
            [0.0, 0.0, 0.0],
            [0.98, 0.98, 0.98],
            [0.98, -0.98, -0.98],
            [-0.98, 0.98, -0.98],
            [-0.98, -0.98, 0.98],
        ],
    ),
    "xenon_water": (
        ["Xe", "O", "H", "H"],
        [[0.0, 0.0, 0.0], [3.80, 0.0, 0.0], [4.39, 0.76, 0.0], [4.39, -0.76, 0.0]],
    ),
    "mercury_dichloride": (
        ["Hg", "Cl", "Cl"],
        [[0.0, 0.0, 0.0], [2.28, 0.0, 0.0], [-2.28, 0.0, 0.0]],
    ),
    "lanthanum_trichloride": (
        ["La", "Cl", "Cl", "Cl"],
        [[0.0, 0.0, 0.0], [2.62, 0.0, 0.3], [-1.31, 2.27, 0.3], [-1.31, -2.27, 0.3]],
    ),
}

CASES = {
    "gfnff": list(MOLECULES),
    "gfn0xtb": ["lif", "phosphine", "titanium_tetrachloride", "stannane", "xenon_water"],
}


def gradient_for(n):
    rng = np.random.default_rng(20261110 + n)
    g = rng.normal(scale=2e-3, size=(n, 3))
    return g - g.mean(axis=0)


def main():
    root, out = sys.argv[1], sys.argv[2]
    sys.path.insert(0, root)
    from multioptpy.ModelHessian.approx_hessian import ApproxHessian

    data = {
        "source": "MultiOptPy 1.21.2 ModelHessian.approx_hessian.ApproxHessian.main",
        "generator": "tools/golden/legacy_gfn_heavy.py",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "units": "positions Bohr, gradient Hartree/Bohr, Hessian Hartree/Bohr^2 (row-major)",
        "cases": [],
    }
    for model, mols in CASES.items():
        for mol in mols:
            symbols, ang = MOLECULES[mol]
            x = np.asarray(ang, dtype=float) / BOHR
            g = gradient_for(len(symbols))
            with contextlib.redirect_stdout(io.StringIO()):
                h = ApproxHessian().main(x.copy(), list(symbols), g.copy(), model)
            data["cases"].append(
                {
                    "molecule": mol,
                    "model": model,
                    "symbols": symbols,
                    "positions_bohr": x.ravel().tolist(),
                    "gradient": g.ravel().tolist(),
                    "hessian": np.asarray(h, dtype=float).ravel().tolist(),
                }
            )
            print(mol, model, file=sys.stderr)
    with open(out, "wb") as f:
        f.write((json.dumps(data, indent=1) + "\n").encode("utf-8"))


if __name__ == "__main__":
    main()
