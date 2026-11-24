"""Reference model Hessians from the original MultiOptPy (the ``legacy_`` models of mopt-rs).

Usage (inside the reference environment created by setup_ref_env.sh):

    python tools/golden/model_hessians.py <multioptpy_root> <output_json>

For every model name MultiOptPy's ``ApproxHessian.main`` accepts (and the ``ts``, ``sr``, and
``clip`` modifiers), the Hessian (Hartree/Bohr^2) is computed for a few molecules with a fixed
gradient and written as JSON together with the inputs.

This file is part of mopt-rs.
"""

import contextlib
import io
import json
import platform
import sys

import numpy as np

BOHR = 0.52917721067

MOLECULES = {
    # MultiOptPy's own test molecule (approx_hessian.test).
    "h2o2": (
        ["O", "O", "H", "H"],
        [
            [1.607230637, 0.000000000, -4.017111134],
            [1.607230637, 0.463701826, -2.637210910],
            [2.429229637, 0.052572461, -2.324941515],
            [0.785231637, -0.516274287, -4.017735703],
        ],
    ),
    "methanol": (
        ["C", "O", "H", "H", "H", "H"],
        [
            [-0.046, 0.663, 0.000],
            [-0.046, -0.756, 0.000],
            [-1.086, 0.976, 0.000],
            [0.437, 1.077, 0.889],
            [0.437, 1.077, -0.889],
            [0.866, -1.048, 0.000],
        ],
    ),
    "ethene_hcl": (
        ["C", "C", "H", "H", "H", "H", "Cl", "H"],
        [
            [0.000, 0.000, 0.667],
            [0.000, 0.000, -0.667],
            [0.000, 0.923, 1.238],
            [0.000, -0.923, 1.238],
            [0.000, 0.923, -1.238],
            [0.000, -0.923, -1.238],
            [3.100, 0.000, 0.300],
            [1.850, 0.000, 0.250],
        ],
    ),
    # Hydrogen fluoride next to water in a bent arrangement (MultiOptPy's GFN-FF-like model
    # detects F-H...O only when the D->H / H->A angle exceeds 120 degrees).
    "hf_water": (
        ["F", "H", "O", "H", "H"],
        [
            [0.000, 0.000, 0.000],
            [0.920, 0.000, 0.000],
            [-0.300, 1.750, 0.000],
            [-0.300, 2.710, 0.000],
            [-1.206, 1.750, 0.320],
        ],
    ),
    # A short H-C#N (the GFN0-like model's cyano branch needs both bonds within its radii).
    "hcn": (
        ["H", "C", "N"],
        [[0.000, 0.000, -1.040], [0.000, 0.000, 0.000], [0.000, 0.000, 1.150]],
    ),
    # S, F, Br, I, N in one molecule each.
    "fluoromethanethiol": (
        ["C", "F", "S", "H", "H", "H"],
        [
            [0.000, 0.000, 0.000],
            [1.350, 0.000, 0.000],
            [-0.600, 1.650, 0.000],
            [-1.940, 1.750, 0.000],
            [-0.360, -0.500, 0.890],
            [-0.360, -0.500, -0.890],
        ],
    ),
    "bromoiodomethylamine": (
        ["C", "Br", "I", "N", "H", "H", "H"],
        [
            [0.000, 0.000, 0.000],
            [1.940, 0.000, 0.000],
            [-0.700, 2.000, 0.000],
            [-0.500, -0.700, 1.200],
            [-1.500, -0.700, 1.300],
            [-0.200, -1.600, 1.600],
            [-0.500, -0.600, -0.900],
        ],
    ),
}

BASES = [
    "gfnff",
    "gfn0xtb",
    "fischerd3old",
    "fischerd3",
    "fischerd4",
    "schlegeld3",
    "schlegeld4",
    "schlegel",
    "swartd3",
    "swartd4",
    "swartd2",
    "swart",
    "lindh2007d3",
    "lindh2007d4",
    "lindh2007",
    "lindh",
    "fischer",
    "morse",
]
MODIFIED = ["fischerd3_ts", "lindh2007d3_sr", "swartd3_clip", "schlegel_ts_sr"]


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
        "python": platform.python_version(),
        "numpy": np.__version__,
        "units": "positions Bohr, gradient Hartree/Bohr, Hessian Hartree/Bohr^2 (row-major)",
        "cases": [],
    }
    for mol, (symbols, ang) in MOLECULES.items():
        x = np.asarray(ang, dtype=float) / BOHR
        g = gradient_for(len(symbols))
        for name in BASES + MODIFIED:
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    h = ApproxHessian().main(x.copy(), list(symbols), g.copy(), name)
                h = np.asarray(h, dtype=float)
                case = {"hessian": h.ravel().tolist()}
            except Exception as e:  # noqa: BLE001 - recorded as data
                case = {"error": f"{type(e).__name__}: {e}"}
            case.update(
                {
                    "molecule": mol,
                    "model": name,
                    "symbols": symbols,
                    "positions_bohr": x.ravel().tolist(),
                    "gradient": g.ravel().tolist(),
                }
            )
            data["cases"].append(case)
            print(mol, name, "error" if "error" in case else "ok", file=sys.stderr)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f)


if __name__ == "__main__":
    main()
