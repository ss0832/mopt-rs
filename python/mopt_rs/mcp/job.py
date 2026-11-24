"""The child side of a job of the Python server: ``python -P -m mopt_rs.mcp.job COMMAND
[ARGS...]`` runs the pip-installed command ``COMMAND`` (the same entry point of
``mopt_rs.cli``) in the job directory, with the engines of the Python package, and records the
exit code in ``mcp_exit_code`` there.

Only the commands of the tables run. ``-P`` keeps the job directory out of the module search
path, so that a file copied into a job cannot stand in for a module.

MultiOptPy: no counterpart (new in mopt-rs).

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

from mopt_rs.mcp.tables import load_commands

JOB_FILE = "mcp_job.json"
EXIT_FILE = "mcp_exit_code"


def main(argv: list[str]) -> int:
    """Run ``argv`` = ``COMMAND [ARGS...]``; returns the exit code."""
    names = [c["name"] for c in load_commands()]
    if not argv or argv[0] not in names:
        print(f"mopt_rs.mcp.job: give a command of mopt-rs ({', '.join(names)})", file=sys.stderr)
        return 2
    command, *rest = argv
    from mopt_rs import cli

    entry = getattr(cli, command)
    sys.argv = [command, *rest]
    code: int
    try:
        entry()
        code = 0
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        code = 1
    job_file = Path(JOB_FILE)
    if job_file.is_file() and not job_file.is_symlink():
        exit_file = Path(EXIT_FILE)
        if not exit_file.is_symlink():
            exit_file.write_text(f"{code}\n", encoding="utf-8")
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
