"""``python -m mopt_rs.calculators``: the engines of other packages, installed or how to install
them.

MultiOptPy: no counterpart (new in mopt-rs).

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

from mopt_rs.calculators.packages import report

if __name__ == "__main__":
    print(report())
