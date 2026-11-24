"""JSON-RPC 2.0 messages of the Model Context Protocol, the protocol versions, and the error
codes (the same as ``crates/mopt-rs/src/mcp/protocol.rs``).

Two eras are served: the stateless revision ``2026-07-28`` (every request carries the protocol
version and the client capabilities in ``params._meta``; ``server/discover``; ``resultType`` in
every result), and the revisions with the ``initialize`` handshake (``2025-11-25`` back to
``2024-11-05``).

MultiOptPy: no counterpart (new in mopt-rs).

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

#: The stateless revision.
MODERN_VERSION = "2026-07-28"
#: The revisions with the ``initialize`` handshake, newest first.
LEGACY_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")

META_PROTOCOL_VERSION = "io.modelcontextprotocol/protocolVersion"
META_CLIENT_CAPABILITIES = "io.modelcontextprotocol/clientCapabilities"
META_SERVER_INFO = "io.modelcontextprotocol/serverInfo"

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
UNSUPPORTED_PROTOCOL_VERSION = -32022
LEGACY_RESOURCE_NOT_FOUND = -32002


def supported_versions() -> list[str]:
    """Every protocol version the server accepts, newest first."""
    return [MODERN_VERSION, *LEGACY_VERSIONS]


def negotiate_legacy(requested: str | None) -> str:
    """The version answered to an ``initialize`` with ``requested``."""
    return requested if requested in LEGACY_VERSIONS else LEGACY_VERSIONS[0]


@dataclass
class RpcError(Exception):
    """A JSON-RPC error."""

    code: int
    message: str
    data: Any = None

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.data is not None:
            out["data"] = self.data
        return out


def invalid_params(message: str, data: Any = None) -> RpcError:
    """``-32602``."""
    return RpcError(INVALID_PARAMS, message, data)


def unsupported_version(requested: str) -> RpcError:
    """``UnsupportedProtocolVersion`` for ``requested``."""
    return RpcError(
        UNSUPPORTED_PROTOCOL_VERSION,
        "Unsupported protocol version",
        {"supported": supported_versions(), "requested": requested},
    )


def result_response(request_id: Any, result: Any) -> dict[str, Any]:
    """The response carrying ``result``."""
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def error_response(request_id: Any, error: RpcError) -> dict[str, Any]:
    """The error response (without an id when it could not be read: ``request_id`` None)."""
    out: dict[str, Any] = {"jsonrpc": "2.0"}
    if request_id is not None:
        out["id"] = request_id
    out["error"] = error.to_json()
    return out


def _valid_id(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, str):
        return True
    return isinstance(value, int) or (isinstance(value, float) and value.is_integer())


@dataclass
class Request:
    """A request: id, method, parameters."""

    id: Any
    method: str
    params: dict[str, Any]


@dataclass
class Notification:
    """A notification: method, parameters."""

    method: str
    params: dict[str, Any]


#: The longest message the server reads.
MAX_MESSAGE_BYTES = 16 << 20
#: The deepest nesting of arrays and objects in a message.
MAX_NESTING = 64


def nests_deeper_than(line: str, limit: int) -> bool:
    """Whether arrays and objects in ``line`` nest deeper than ``limit`` (strings skipped)."""
    depth = 0
    in_string = escaped = False
    for ch in line:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "[{":
            depth += 1
            if depth > limit:
                return True
        elif ch in "]}":
            depth = max(0, depth - 1)
    return False


def parse_message(line: str) -> Request | Notification | dict[str, Any] | None:
    """The message of one line, an error response to send, or None for a response from the
    client (the server sends no requests)."""
    if nests_deeper_than(line, MAX_NESTING):
        return error_response(
            None,
            RpcError(INVALID_REQUEST, f"Invalid request: nested deeper than {MAX_NESTING} levels"),
        )
    try:
        value = json.loads(line)
    except (ValueError, RecursionError) as e:
        return error_response(None, RpcError(PARSE_ERROR, f"Parse error: {e}"))

    def invalid(request_id: Any, why: str) -> dict[str, Any]:
        rid = request_id if _valid_id(request_id) else None
        return error_response(rid, RpcError(INVALID_REQUEST, f"Invalid request: {why}"))

    if not isinstance(value, dict):
        why = (
            "batches are not supported; send one message per line"
            if isinstance(value, list)
            else "a message is a JSON object"
        )
        return invalid(None, why)
    request_id = value.get("id")
    if value.get("jsonrpc") != "2.0":
        return invalid(request_id, 'jsonrpc must be "2.0"')
    if "method" not in value:
        if "result" in value or "error" in value:
            return None
        return invalid(request_id, "no method")
    method = value["method"]
    if not isinstance(method, str):
        return invalid(request_id, "the method is a string")
    params = value.get("params")
    if params is None:
        params = {}
    if not isinstance(params, dict):
        return invalid(request_id, "params is an object")
    if "id" not in value:
        return Notification(method, params)
    if not _valid_id(request_id):
        return invalid(None, "the id is a string or an integer, never null")
    return Request(request_id, method, params)
