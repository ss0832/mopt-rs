#!/usr/bin/env bash
# This file is part of mopt-rs.
# Create the Python environment used to run the reference MultiOptPy implementation.
#
# The environment lives outside the repository (on the Linux file system for speed). MultiOptPy
# 1.21.2 pins numpy 2.2 / scipy 1.13 / torch 2.6, which have no wheels for the Python available
# here; the newest compatible versions are installed instead and recorded in the generated data.
set -euo pipefail

VENV="${MOPT_REF_VENV:-$HOME/.cache/mopt-venv-ref}"
if [ ! -x "$VENV/bin/python" ]; then
    python3 -m venv "$VENV"
fi
"$VENV/bin/python" -m pip install -q --upgrade pip
"$VENV/bin/python" -m pip install -q numpy scipy matplotlib ase
"$VENV/bin/python" -m pip install -q torch --index-url https://download.pytorch.org/whl/cpu
"$VENV/bin/python" - <<'PY'
import ase, matplotlib, numpy, scipy, sys, torch
print("python", sys.version.split()[0])
print("numpy", numpy.__version__, "scipy", scipy.__version__, "torch", torch.__version__,
      "matplotlib", matplotlib.__version__, "ase", ase.__version__)
PY
