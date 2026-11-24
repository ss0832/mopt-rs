"""A Model Context Protocol (MCP) server for mopt-rs: ``python -m mopt_rs.mcp [--root DIR]``.

The same tools, resources and prompts as ``mopt mcp`` of the Rust binary, with the engines of
the Python package: every implemented command is a tool that starts a job or a task (through
the pip-installed command), the job tools follow, wait for, stop, cancel, continue and read
jobs, component tools answer at once (the commands and their schemas, the calculators, a single
point, model Hessians, normal modes, coordinates, first images, superposition, bond graphs,
dispersion, Hessian updates, units, elements), and the documentation, the files of run
directories and the status of jobs are resources that can be subscribed to. The tools are built from the
option tables in ``commands.json`` (written by ``tools/cli/gen_cli.py`` with
``crates/mopt-rs/src/cli/spec.rs``). Only the standard library is used besides the package.
See the documentation page ``mcp`` (``python -m mopt_rs.docs mcp``).

MultiOptPy: no counterpart (new in mopt-rs).

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from mopt_rs.mcp.server import Server

__all__ = ["Server", "main"]


def main(argv: list[str] | None = None) -> int:
    """``python -m mopt_rs.mcp``; returns the exit code."""
    parser = argparse.ArgumentParser(
        prog="python -m mopt_rs.mcp",
        description="Model Context Protocol server of mopt-rs over stdio.",
    )
    parser.add_argument("--root", default=".", help="the directory the server may read and write")
    parser.add_argument(
        "--max-jobs", type=int, default=4, help="jobs of this server that may run at a time"
    )
    parser.add_argument(
        "--allow-external",
        action="store_true",
        help="allow settings that start programs outside mopt-rs or load code (refused by default)",
    )
    parser.add_argument(
        "--full-schemas",
        action="store_true",
        help="list every option of every command in tools/list (default: the compact schemas; "
        "command_help gives the options)",
    )
    parser.add_argument(
        "--page-size", type=int, default=100, help="entries per page of the lists (default: 100)"
    )
    args = parser.parse_args(argv)
    if args.max_jobs < 1:
        parser.error("--max-jobs needs a number of at least 1")
    if not 1 <= args.page_size <= 1000:
        parser.error("--page-size needs a number from 1 to 1000")
    # Standard output belongs to the protocol: everything else that writes there (the
    # compiled core, the engines, print) goes to standard error.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", newline="\n")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    try:
        server = Server(
            Path(args.root),
            allow_external=args.allow_external,
            max_jobs=args.max_jobs,
            full_schemas=args.full_schemas,
            page_size=args.page_size,
        )
    except ValueError as e:
        print(f"mopt mcp: {e}", file=sys.stderr)
        return 2
    # Relative paths that engines of a single point open are seen from the root.
    os.chdir(server.sandbox.root)
    server.serve(sys.stdin.buffer, protocol)
    return 0
