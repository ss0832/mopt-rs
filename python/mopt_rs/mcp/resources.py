"""Resources: the documentation pages (the copy installed with the package) and the files of
run directories under the root (as ``crates/mopt-rs/src/mcp/resources.rs``).

- ``mopt://docs/NAME``: a page of ``mopt_rs.docs``;
- ``mopt://root/PATH``: a file (or a directory listing) under the root. Text files are returned
  as text, other files as Base64 up to ``READ_LIMIT`` bytes;
- ``mopt://jobs/JOB_ID``: the status of a job (answered by the server).

Resources can be subscribed to: :func:`check_watch` checks a URI as :func:`read` would, and
:func:`fingerprint` tells when it changed (the metadata of what it resolves to now, which must
still lie inside the root).

MultiOptPy: no counterpart (new in mopt-rs).

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote

from mopt_rs.mcp import run_files
from mopt_rs.mcp.paths import Sandbox, ToolError, refused_path_text
from mopt_rs.mcp.protocol import INVALID_PARAMS, RpcError

DOCS_PREFIX = "mopt://docs/"
ROOT_PREFIX = "mopt://root/"
JOBS_PREFIX = "mopt://jobs/"
READ_LIMIT = 8 << 20
LIST_LIMIT = 1000
_SCAN_DEPTH = 8
_SCAN_DIRECTORIES = 20_000
_SKIPPED = {"target", "node_modules", "__pycache__"}


class ResourceNotFound(LookupError):
    """No such resource."""


def root_uri(rel: str) -> str:
    """``rel`` (relative to the root) as a resource URI."""
    return ROOT_PREFIX + quote(rel, safe="/-._~")


def docs_dir() -> Path:
    """The documentation installed with the package."""
    import mopt_rs.docs

    return mopt_rs.docs.path()


def doc_pages(d: Path) -> list[tuple[str, Path]]:
    """``(name, path)`` of the pages, sorted."""
    return [(p.stem, p) for p in sorted(d.glob("*.md")) if p.is_file()]


def _title(path: Path) -> str | None:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return None


def run_directories_under(root: Path) -> list[Path]:
    """The run directories under the root (hidden directories and build output skipped)."""
    out: list[Path] = []
    level = [root]
    visited = 0
    for _ in range(_SCAN_DEPTH):
        following: list[Path] = []
        for d in level:
            for child in run_files.sorted_entries(d):
                if child.is_symlink() or not child.is_dir():
                    continue
                if child.name.startswith(".") or child.name in _SKIPPED:
                    continue
                visited += 1
                if visited > _SCAN_DIRECTORIES:
                    return out
                (out if run_files.is_run_directory(child) else following).append(child)
        level = following
    return out


def list_resources(sandbox: Sandbox, docs: Path | None) -> list[dict[str, Any]]:
    """The documentation pages, then the files of the run directories."""
    out: list[dict[str, Any]] = []
    if docs is not None:
        for name, path in doc_pages(docs):
            out.append(
                {
                    "uri": DOCS_PREFIX + name,
                    "name": f"{name}.md",
                    "title": _title(path) or name,
                    "description": "documentation of mopt-rs",
                    "mimeType": "text/markdown",
                    "size": path.stat().st_size,
                }
            )
    for run in run_directories_under(sandbox.root):
        files, _ = run_files.list_files(run, LIST_LIMIT)
        for f, size in files:
            if len(out) >= LIST_LIMIT:
                return out
            rel = sandbox.relative(f)
            out.append(
                {
                    "uri": root_uri(rel),
                    "name": rel,
                    "mimeType": run_files.mime_type(f),
                    "size": size,
                }
            )
    return out


def templates() -> list[dict[str, Any]]:
    """The resource templates."""
    return [
        {
            "uriTemplate": ROOT_PREFIX + "{+path}",
            "name": "file under the root",
            "description": "Any file or directory under the root of the server, by its path "
            "relative to the root.",
        },
        {
            "uriTemplate": DOCS_PREFIX + "{name}",
            "name": "documentation page",
            "description": "A page of the documentation of mopt-rs, by its file name without .md.",
            "mimeType": "text/markdown",
        },
        {
            "uriTemplate": JOBS_PREFIX + "{job_id}",
            "name": "status of a job",
            "description": "The status of a job as job_status gives it, by its job_id; subscribe "
            "to it to hear when the job changes.",
            "mimeType": "application/json",
        },
    ]


def _root_path(uri: str) -> str:
    """The path relative to the root of ``mopt://root/...`` (decoded once, checked by its
    text); raises :class:`ToolError`."""
    if not uri.startswith(ROOT_PREFIX):
        raise ToolError(f"{uri}: not a file under the root")
    encoded = uri[len(ROOT_PREFIX) :]
    if any(e in encoded.lower() for e in ("%2e", "%2f", "%5c", "%00")):
        raise ToolError(f"{uri}: encoded dots, slashes, backslashes and NUL characters are refused")
    try:
        rel = unquote(encoded, errors="strict") or "."
    except UnicodeDecodeError:
        raise ToolError(f"{uri}: not a valid URI") from None
    why = refused_path_text(rel)
    if why is not None:
        raise ToolError(why)
    return rel


def check_watch(sandbox: Sandbox, docs: Path | None, uri: str) -> None:
    """Whether ``uri`` can be subscribed to (raises :class:`ToolError`): a page of ``docs``, or
    a path under the root checked as :func:`read` checks it (it may not exist yet; when it
    exists it must lie inside the root). Job URIs are checked by the server."""
    if uri.startswith(DOCS_PREFIX):
        name = uri[len(DOCS_PREFIX) :].removesuffix(".md")
        if docs is None or name not in [n for n, _ in doc_pages(docs)]:
            raise ToolError(f"no documentation page '{name}'")
        return
    rel = _root_path(uri)
    target = sandbox.root / rel
    if os.path.lexists(target):
        try:
            resolved = target.resolve(strict=True)
        except OSError:
            return
        if not sandbox.contains(resolved):
            raise ToolError(f"'{rel}' is outside the root of the server")


def fingerprint(sandbox: Sandbox, uri: str) -> str:
    """What tells that the resource ``uri`` changed; fixed for a path that resolves outside the
    root now, so that nothing about what lies outside is told."""
    if uri.startswith(DOCS_PREFIX):
        return "page"
    try:
        rel = _root_path(uri)
    except ToolError:
        return "refused"
    try:
        resolved = (sandbox.root / rel).resolve(strict=True)
    except OSError:
        return "absent"
    if not sandbox.contains(resolved):
        return "outside"
    try:
        st = resolved.stat()
    except OSError:
        return "absent"
    entries = 0
    if resolved.is_dir():
        try:
            entries = sum(1 for _ in zip(range(100_000), resolved.iterdir(), strict=False))
        except OSError:
            entries = 0
    return f"{resolved.is_dir()}:{st.st_size}:{st.st_mtime_ns}:{entries}"


def read(sandbox: Sandbox, docs: Path | None, uri: str) -> list[dict[str, Any]]:
    """The contents of ``uri``; raises :class:`ResourceNotFound` or :class:`RpcError`."""
    if uri.startswith(DOCS_PREFIX):
        name = uri[len(DOCS_PREFIX) :].removesuffix(".md")
        if docs is None:
            raise ResourceNotFound(f"{uri}: no documentation is installed")
        for page, path in doc_pages(docs):
            if page == name:
                return [
                    {
                        "uri": uri,
                        "mimeType": "text/markdown",
                        "text": path.read_text(encoding="utf-8"),
                    }
                ]
        raise ResourceNotFound(f"no documentation page '{name}'")
    if not uri.startswith(ROOT_PREFIX):
        raise ResourceNotFound(f"{uri}: resources are {DOCS_PREFIX}NAME and {ROOT_PREFIX}PATH")
    encoded = uri[len(ROOT_PREFIX) :]
    if any(e in encoded.lower() for e in ("%2e", "%2f", "%5c", "%00")):
        raise ResourceNotFound(
            f"{uri}: encoded dots, slashes, backslashes and NUL characters are refused"
        )
    # Decoded once; the result is checked as any path of a client.
    try:
        rel = unquote(encoded, errors="strict") or "."
    except UnicodeDecodeError:
        raise ResourceNotFound(f"{uri}: not a valid URI") from None
    try:
        path = sandbox.existing(rel)
    except ToolError as e:
        raise ResourceNotFound(str(e)) from None
    if path.is_dir():
        entries = []
        for entry in run_files.sorted_entries(path):
            r = sandbox.relative(entry)
            entries.append(
                {
                    "path": r,
                    "uri": root_uri(r),
                    "directory": entry.is_dir() and not entry.is_symlink(),
                    "size": entry.lstat().st_size,
                }
            )
        listing = json.dumps({"entries": entries}, indent=2)
        return [{"uri": uri, "mimeType": "application/json", "text": listing}]
    if not path.is_file():
        raise ResourceNotFound(f"{rel} is not a regular file")
    size = path.stat().st_size
    if size > READ_LIMIT:
        raise RpcError(
            INVALID_PARAMS,
            f"{rel} has {size} bytes, more than the {READ_LIMIT} bytes read through MCP",
            {"uri": uri, "size": size},
        )
    try:
        opened, _ = sandbox.open_regular(path)
    except ToolError as e:
        raise ResourceNotFound(str(e)) from None
    with opened:
        data = opened.read(READ_LIMIT + 1)
    if len(data) > READ_LIMIT:
        raise RpcError(INVALID_PARAMS, f"{rel} grew beyond the {READ_LIMIT} bytes read through MCP")
    mime = run_files.mime_type(path)
    binary = (mime.startswith("image/") and mime != "image/svg+xml") or mime in (
        "application/pdf",
        "application/vnd.sqlite3",
    )
    if not binary:
        try:
            content = data.decode("utf-8")
        except UnicodeDecodeError:
            pass
        else:
            shown = "text/plain" if mime == "application/octet-stream" else mime
            return [{"uri": uri, "mimeType": shown, "text": content}]
    return [{"uri": uri, "mimeType": mime, "blob": base64.b64encode(data).decode("ascii")}]
