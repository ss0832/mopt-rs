"""``python -m mopt_rs.docs [NAME | --path]``: list, print, or locate the documentation.

MultiOptPy: no counterpart (new in mopt-rs).

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import sys

from mopt_rs import docs


def main(argv: list[str]) -> int:
    """Exit code of the command."""
    if not argv:
        print(f"Documents of mopt-rs in {docs.path()}:")
        for name in docs.names():
            print(f"  {name}")
        print("Print one with: python -m mopt_rs.docs NAME")
        return 0
    if argv[0] in ("--path", "-p"):
        print(docs.path())
        return 0
    try:
        text = docs.read(argv[0])
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 2
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
