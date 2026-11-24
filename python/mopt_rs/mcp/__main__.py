"""``python -m mopt_rs.mcp [--root DIR] [--max-jobs N] [--allow-external]``.

MultiOptPy: no counterpart (new in mopt-rs).

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import sys

from mopt_rs.mcp import main

sys.exit(main())
