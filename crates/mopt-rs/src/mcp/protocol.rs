//! JSON-RPC 2.0 messages of the Model Context Protocol, the protocol versions, and the error
//! codes.
//!
//! Two eras of the protocol are served (a "dual-era" server in the words of the
//! specification):
//!
//! - the stateless revision [`MODERN_VERSION`] (`2026-07-28`): every request carries the
//!   protocol version and the client capabilities in `params._meta`
//!   (`io.modelcontextprotocol/protocolVersion`, `io.modelcontextprotocol/clientCapabilities`),
//!   `server/discover` reports what the server supports, and every result carries
//!   `resultType`;
//! - the revisions with the `initialize` handshake ([`LEGACY_VERSIONS`], `2025-11-25` back to
//!   `2024-11-05`): `initialize` negotiates the version for the rest of the process,
//!   `notifications/initialized` follows, `ping` answers `{}`.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::BTreeMap;

use crate::core_infra::json::JsonValue;

/// The stateless revision (per-request `_meta`, `server/discover`).
pub const MODERN_VERSION: &str = "2026-07-28";
/// The revisions with the `initialize` handshake, newest first.
pub const LEGACY_VERSIONS: [&str; 4] = ["2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"];

/// `_meta` key of the protocol version of a request.
pub const META_PROTOCOL_VERSION: &str = "io.modelcontextprotocol/protocolVersion";
/// `_meta` key of the client capabilities of a request.
pub const META_CLIENT_CAPABILITIES: &str = "io.modelcontextprotocol/clientCapabilities";
/// `_meta` key of the server identity of a result.
pub const META_SERVER_INFO: &str = "io.modelcontextprotocol/serverInfo";

/// JSON-RPC: invalid JSON.
pub const PARSE_ERROR: i64 = -32700;
/// JSON-RPC: not a valid request object.
pub const INVALID_REQUEST: i64 = -32600;
/// JSON-RPC: unknown method.
pub const METHOD_NOT_FOUND: i64 = -32601;
/// JSON-RPC: invalid parameters (also: resource not found, from `2026-07-28` on).
pub const INVALID_PARAMS: i64 = -32602;
/// JSON-RPC: internal error.
pub const INTERNAL_ERROR: i64 = -32603;
/// MCP `2026-07-28`: the requested protocol version is not supported.
pub const UNSUPPORTED_PROTOCOL_VERSION: i64 = -32022;
/// MCP up to `2025-11-25`: resource not found.
pub const LEGACY_RESOURCE_NOT_FOUND: i64 = -32002;

/// Every protocol version the server accepts, newest first.
pub fn supported_versions() -> Vec<&'static str> {
    let mut out = vec![MODERN_VERSION];
    out.extend(LEGACY_VERSIONS);
    out
}

/// How a request is to be answered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Era {
    /// The stateless revision: `resultType` and the cache fields in results.
    Modern,
    /// A revision with the handshake (the negotiated or the declared version).
    Legacy(String),
}

impl Era {
    /// The era of a protocol version the server supports.
    pub fn of_version(version: &str) -> Option<Self> {
        if version == MODERN_VERSION {
            Some(Self::Modern)
        } else if LEGACY_VERSIONS.contains(&version) {
            Some(Self::Legacy(version.to_string()))
        } else {
            None
        }
    }
}

/// The version a server with the handshake answers to an `initialize` with `requested`: the
/// same when it is supported, else the newest revision with the handshake.
pub fn negotiate_legacy(requested: Option<&str>) -> &'static str {
    match requested {
        Some(v) => LEGACY_VERSIONS
            .iter()
            .find(|s| **s == v)
            .copied()
            .unwrap_or(LEGACY_VERSIONS[0]),
        None => LEGACY_VERSIONS[0],
    }
}

/// A JSON-RPC error.
#[derive(Debug, Clone, PartialEq)]
pub struct RpcError {
    /// Error code.
    pub code: i64,
    /// Message.
    pub message: String,
    /// Further information.
    pub data: Option<JsonValue>,
}

impl RpcError {
    /// An error without data.
    pub fn new(code: i64, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            data: None,
        }
    }

    /// `-32602`, invalid parameters.
    pub fn invalid_params(message: impl Into<String>) -> Self {
        Self::new(INVALID_PARAMS, message)
    }

    /// The error with `data`.
    #[must_use]
    pub fn with_data(mut self, data: JsonValue) -> Self {
        self.data = Some(data);
        self
    }

    /// The error `UnsupportedProtocolVersion` for `requested`.
    pub fn unsupported_version(requested: &str) -> Self {
        Self::new(UNSUPPORTED_PROTOCOL_VERSION, "Unsupported protocol version").with_data(object([
            (
                "supported",
                JsonValue::Array(supported_versions().into_iter().map(text).collect()),
            ),
            ("requested", text(requested)),
        ]))
    }

    fn to_json(&self) -> JsonValue {
        let mut map = BTreeMap::new();
        map.insert("code".to_string(), JsonValue::Number(self.code as f64));
        map.insert("message".to_string(), text(&self.message));
        if let Some(d) = &self.data {
            map.insert("data".to_string(), d.clone());
        }
        JsonValue::Object(map)
    }
}

/// One message read from the client.
#[derive(Debug, Clone, PartialEq)]
pub enum Incoming {
    /// A request: id, method, parameters (an object; empty when absent).
    Request(JsonValue, String, JsonValue),
    /// A notification: method, parameters.
    Notification(String, JsonValue),
    /// A response to a request of the server (the server sends none; ignored).
    Response,
}

/// The longest message the server reads (longer lines are answered with `-32600`).
pub const MAX_MESSAGE_BYTES: usize = 16 << 20;
/// The deepest nesting of arrays and objects in a message (deeper ones are answered with
/// `-32600` before they are parsed, so that the parser cannot exhaust the stack).
pub const MAX_NESTING: usize = 64;

/// Whether arrays and objects in `line` nest deeper than `limit` (strings are skipped).
pub fn nests_deeper_than(line: &str, limit: usize) -> bool {
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for b in line.bytes() {
        if in_string {
            match (escaped, b) {
                (true, _) => escaped = false,
                (false, b'\\') => escaped = true,
                (false, b'"') => in_string = false,
                _ => {}
            }
            continue;
        }
        match b {
            b'"' => in_string = true,
            b'[' | b'{' => {
                depth += 1;
                if depth > limit {
                    return true;
                }
            }
            b']' | b'}' => depth = depth.saturating_sub(1),
            _ => {}
        }
    }
    false
}

/// Parse one line: the message, or the error response to send (`-32700`, `-32600`).
pub fn parse_message(line: &str) -> Result<Incoming, JsonValue> {
    if nests_deeper_than(line, MAX_NESTING) {
        return Err(error_response(
            None,
            &RpcError::new(
                INVALID_REQUEST,
                format!("Invalid request: nested deeper than {MAX_NESTING} levels"),
            ),
        ));
    }
    let value = JsonValue::parse(line).map_err(|e| {
        error_response(
            None,
            &RpcError::new(PARSE_ERROR, format!("Parse error: {e}")),
        )
    })?;
    let invalid = |id: Option<&JsonValue>, why: &str| {
        error_response(
            id,
            &RpcError::new(INVALID_REQUEST, format!("Invalid request: {why}")),
        )
    };
    let Some(map) = value.as_object() else {
        let why = if value.as_array().is_some() {
            "batches are not supported; send one message per line"
        } else {
            "a message is a JSON object"
        };
        return Err(invalid(None, why));
    };
    let id = map.get("id");
    if map.get("jsonrpc").and_then(JsonValue::as_str) != Some("2.0") {
        return Err(invalid(
            id.filter(|i| valid_id(i)),
            "jsonrpc must be \"2.0\"",
        ));
    }
    let Some(method) = map.get("method") else {
        if map.contains_key("result") || map.contains_key("error") {
            return Ok(Incoming::Response);
        }
        return Err(invalid(id.filter(|i| valid_id(i)), "no method"));
    };
    let Some(method) = method.as_str() else {
        return Err(invalid(
            id.filter(|i| valid_id(i)),
            "the method is a string",
        ));
    };
    let params = match map.get("params") {
        None | Some(JsonValue::Null) => JsonValue::empty_object(),
        Some(p @ JsonValue::Object(_)) => p.clone(),
        Some(_) => return Err(invalid(id.filter(|i| valid_id(i)), "params is an object")),
    };
    match id {
        None => Ok(Incoming::Notification(method.to_string(), params)),
        Some(i) if valid_id(i) => Ok(Incoming::Request(i.clone(), method.to_string(), params)),
        Some(_) => Err(invalid(
            None,
            "the id is a string or an integer, never null",
        )),
    }
}

fn valid_id(id: &JsonValue) -> bool {
    match id {
        JsonValue::String(_) => true,
        JsonValue::Number(n) => n.is_finite() && n.fract() == 0.0,
        _ => false,
    }
}

/// The response carrying `result` for the request `id`.
pub fn result_response(id: &JsonValue, result: JsonValue) -> JsonValue {
    object([
        ("jsonrpc", text("2.0")),
        ("id", id.clone()),
        ("result", result),
    ])
}

/// The error response for the request `id` (without an id when it could not be read).
pub fn error_response(id: Option<&JsonValue>, error: &RpcError) -> JsonValue {
    let mut map = BTreeMap::new();
    map.insert("jsonrpc".to_string(), text("2.0"));
    if let Some(i) = id {
        map.insert("id".to_string(), i.clone());
    }
    map.insert("error".to_string(), error.to_json());
    JsonValue::Object(map)
}

/// A JSON string.
pub fn text(s: impl AsRef<str>) -> JsonValue {
    JsonValue::String(s.as_ref().to_string())
}

/// A JSON object of `pairs`.
pub fn object<const N: usize>(pairs: [(&str, JsonValue); N]) -> JsonValue {
    JsonValue::Object(pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}

/// A JSON number of a count.
pub fn count(n: usize) -> JsonValue {
    JsonValue::Number(n as f64)
}

/// The value of `key` in the object `value`, as a string.
pub fn get_str<'a>(value: &'a JsonValue, key: &str) -> Option<&'a str> {
    value.get(key).and_then(JsonValue::as_str)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn messages_are_classified_and_bad_ones_answered() {
        let r = parse_message(r#"{"jsonrpc":"2.0","id":1,"method":"tools/list"}"#).unwrap();
        assert_eq!(
            r,
            Incoming::Request(
                JsonValue::Number(1.0),
                "tools/list".into(),
                JsonValue::empty_object()
            )
        );
        let n = parse_message(r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#);
        assert!(matches!(n, Ok(Incoming::Notification(m, _)) if m == "notifications/initialized"));
        let e = parse_message("{not json").unwrap_err();
        assert_eq!(
            e.get("error").and_then(|e| e.get("code")),
            Some(&JsonValue::Number(-32700.0))
        );
        assert!(e.get("id").is_none());
        let e = parse_message(r#"[{"jsonrpc":"2.0","id":1,"method":"ping"}]"#).unwrap_err();
        assert_eq!(
            e.get("error").and_then(|e| e.get("code")),
            Some(&JsonValue::Number(-32600.0))
        );
        let e = parse_message(r#"{"jsonrpc":"2.0","id":null,"method":"ping"}"#).unwrap_err();
        assert_eq!(
            e.get("error").and_then(|e| e.get("code")),
            Some(&JsonValue::Number(-32600.0))
        );
        let e =
            parse_message(r#"{"jsonrpc":"2.0","id":"a","method":"x","params":[1]}"#).unwrap_err();
        assert_eq!(e.get("id"), Some(&text("a")));
        // Deep nesting is refused before parsing; brackets inside strings do not count.
        let deep = "[".repeat(100_000);
        let e = parse_message(&deep).unwrap_err();
        assert_eq!(
            e.get("error").and_then(|e| e.get("code")),
            Some(&JsonValue::Number(-32600.0))
        );
        let quoted = format!(
            r#"{{"jsonrpc":"2.0","method":"x","params":{{"a":"{}"}}}}"#,
            "[".repeat(1000)
        );
        assert!(parse_message(&quoted).is_ok());
    }

    #[test]
    fn versions_are_negotiated() {
        assert_eq!(negotiate_legacy(Some("2025-06-18")), "2025-06-18");
        assert_eq!(negotiate_legacy(Some("2026-07-28")), "2025-11-25");
        assert_eq!(negotiate_legacy(Some("1.0")), "2025-11-25");
        assert_eq!(Era::of_version(MODERN_VERSION), Some(Era::Modern));
        assert_eq!(Era::of_version("1900-01-01"), None);
        let e = RpcError::unsupported_version("1900-01-01");
        assert_eq!(e.code, -32022);
        let data = e.data.unwrap();
        assert_eq!(get_str(&data, "requested"), Some("1900-01-01"));
    }
}
