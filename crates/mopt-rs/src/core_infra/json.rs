//! RFC 8259 compliant recursive-descent JSON parser and pretty-printing serializer.
//!
//! MultiOptPy: no counterpart (new in mopt-rs) (MultiOptPy uses Python's `json`).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::BTreeMap;
use std::fmt::Write as _;

use crate::error::{MoptError, MoptResult};

/// In-memory representation of any RFC 8259 JSON value.
#[derive(Debug, Clone, PartialEq)]
pub enum JsonValue {
    /// JSON `null`.
    Null,
    /// JSON `true` or `false`.
    Bool(bool),
    /// JSON numeric value stored as `f64`.
    Number(f64),
    /// JSON UTF-8 string.
    String(String),
    /// JSON ordered list.
    Array(Vec<JsonValue>),
    /// JSON key-value map.
    Object(BTreeMap<String, JsonValue>),
}

impl JsonValue {
    /// Parse a JSON document from a string slice.
    pub fn parse(input: &str) -> MoptResult<Self> {
        let mut parser = JsonParser::new(input);
        let value = parser.parse_value()?;
        parser.skip_whitespace();
        if !parser.is_eof() {
            return Err(MoptError::parse(format!(
                "Trailing characters in JSON at byte offset {}",
                parser.pos
            )));
        }
        Ok(value)
    }

    /// Serialize this JSON value into a compact single-line string.
    pub fn to_string_compact(&self) -> String {
        let mut out = String::new();
        self.write_fmt(&mut out, None, 0);
        out
    }

    /// Serialize this JSON value into a human-readable indented string (2 spaces per level).
    pub fn to_string_pretty(&self) -> String {
        let mut out = String::new();
        self.write_fmt(&mut out, Some(2), 0);
        out
    }

    /// Construct an empty JSON object.
    pub fn empty_object() -> Self {
        Self::Object(BTreeMap::new())
    }

    /// Return `true` if `self` is `JsonValue::Null`.
    pub fn is_null(&self) -> bool {
        matches!(self, Self::Null)
    }

    /// Extract boolean value if `Self::Bool`.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Extract `f64` number if `Self::Number`.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Number(n) => Some(*n),
            _ => None,
        }
    }

    /// Extract `i64` integer if `Self::Number`.
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Self::Number(n) if n.is_finite() => Some(*n as i64),
            _ => None,
        }
    }

    /// Extract `usize` if `Self::Number` and non-negative.
    pub fn as_usize(&self) -> Option<usize> {
        match self {
            Self::Number(n) if n.is_finite() && *n >= 0.0 => Some(*n as usize),
            _ => None,
        }
    }

    /// Extract `&str` if `Self::String`.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Extract slice if `Self::Array`.
    pub fn as_array(&self) -> Option<&[JsonValue]> {
        match self {
            Self::Array(arr) => Some(arr.as_slice()),
            _ => None,
        }
    }

    /// Extract object map if `Self::Object`.
    pub fn as_object(&self) -> Option<&BTreeMap<String, JsonValue>> {
        match self {
            Self::Object(map) => Some(map),
            _ => None,
        }
    }

    /// Mutable reference to object map if `Self::Object`.
    pub fn as_object_mut(&mut self) -> Option<&mut BTreeMap<String, JsonValue>> {
        match self {
            Self::Object(map) => Some(map),
            _ => None,
        }
    }

    /// Look up a key on a `Self::Object`.
    pub fn get(&self, key: &str) -> Option<&JsonValue> {
        self.as_object()?.get(key)
    }

    /// Extract an array of strings (converting numbers/booleans to string if needed for config compatibility).
    pub fn as_string_vec(&self) -> Option<Vec<String>> {
        match self {
            Self::Array(items) => {
                let mut out = Vec::with_capacity(items.len());
                for item in items {
                    match item {
                        Self::String(s) => out.push(s.clone()),
                        Self::Number(n) => {
                            if n.fract() == 0.0 && n.abs() < (i64::MAX as f64) {
                                out.push(format!("{}", *n as i64));
                            } else {
                                out.push(format!("{n}"));
                            }
                        }
                        Self::Bool(b) => out.push(b.to_string()),
                        _ => return None,
                    }
                }
                Some(out)
            }
            Self::String(s) => Some(vec![s.clone()]),
            _ => None,
        }
    }

    fn write_fmt(&self, out: &mut String, indent: Option<usize>, depth: usize) {
        match self {
            Self::Null => out.push_str("null"),
            Self::Bool(true) => out.push_str("true"),
            Self::Bool(false) => out.push_str("false"),
            Self::Number(n) => {
                if !n.is_finite() {
                    out.push_str("null");
                } else if n.fract() == 0.0 && n.abs() < 1e15 {
                    let _ = write!(out, "{}", *n as i64);
                } else {
                    let _ = write!(out, "{n}");
                }
            }
            Self::String(s) => write_escaped_str(out, s),
            Self::Array(items) => {
                if items.is_empty() {
                    out.push_str("[]");
                    return;
                }
                out.push('[');
                for (idx, item) in items.iter().enumerate() {
                    if idx > 0 {
                        out.push(',');
                    }
                    if let Some(ind) = indent {
                        out.push('\n');
                        for _ in 0..((depth + 1) * ind) {
                            out.push(' ');
                        }
                    }
                    item.write_fmt(out, indent, depth + 1);
                }
                if let Some(ind) = indent {
                    out.push('\n');
                    for _ in 0..(depth * ind) {
                        out.push(' ');
                    }
                }
                out.push(']');
            }
            Self::Object(map) => {
                if map.is_empty() {
                    out.push_str("{}");
                    return;
                }
                out.push('{');
                for (idx, (k, v)) in map.iter().enumerate() {
                    if idx > 0 {
                        out.push(',');
                    }
                    if let Some(ind) = indent {
                        out.push('\n');
                        for _ in 0..((depth + 1) * ind) {
                            out.push(' ');
                        }
                    }
                    write_escaped_str(out, k);
                    if indent.is_some() {
                        out.push_str(": ");
                    } else {
                        out.push(':');
                    }
                    v.write_fmt(out, indent, depth + 1);
                }
                if let Some(ind) = indent {
                    out.push('\n');
                    for _ in 0..(depth * ind) {
                        out.push(' ');
                    }
                }
                out.push('}');
            }
        }
    }
}

fn write_escaped_str(out: &mut String, s: &str) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\x08' => out.push_str("\\b"),
            '\x0c' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

struct JsonParser<'a> {
    bytes: &'a [u8],
    src: &'a str,
    pos: usize,
}

impl<'a> JsonParser<'a> {
    fn new(src: &'a str) -> Self {
        let stripped = src.strip_prefix('\u{feff}').unwrap_or(src);
        Self {
            bytes: stripped.as_bytes(),
            src: stripped,
            pos: 0,
        }
    }

    fn is_eof(&self) -> bool {
        self.pos >= self.bytes.len()
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn skip_whitespace(&mut self) {
        while let Some(b) = self.peek() {
            if matches!(b, b' ' | b'\t' | b'\n' | b'\r') {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn parse_value(&mut self) -> MoptResult<JsonValue> {
        self.skip_whitespace();
        let b = self
            .peek()
            .ok_or_else(|| MoptError::parse("Unexpected end of JSON input"))?;
        match b {
            b'n' => self.parse_literal(b"null", JsonValue::Null),
            b't' => self.parse_literal(b"true", JsonValue::Bool(true)),
            b'f' => self.parse_literal(b"false", JsonValue::Bool(false)),
            b'"' => self.parse_string().map(JsonValue::String),
            b'[' => self.parse_array(),
            b'{' => self.parse_object(),
            b'-' | b'0'..=b'9' => self.parse_number(),
            other => Err(MoptError::parse(format!(
                "Unexpected character '{}' at byte {}",
                other as char, self.pos
            ))),
        }
    }

    fn parse_literal(&mut self, expected: &[u8], val: JsonValue) -> MoptResult<JsonValue> {
        if self.bytes.get(self.pos..self.pos + expected.len()) == Some(expected) {
            self.pos += expected.len();
            Ok(val)
        } else {
            Err(MoptError::parse(format!(
                "Invalid JSON literal at byte {}",
                self.pos
            )))
        }
    }

    fn parse_number(&mut self) -> MoptResult<JsonValue> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        match self.peek() {
            Some(b'0') => self.pos += 1,
            Some(b'1'..=b'9') => {
                while matches!(self.peek(), Some(b'0'..=b'9')) {
                    self.pos += 1;
                }
            }
            _ => return Err(MoptError::parse(format!("Invalid number at byte {start}"))),
        }
        if self.peek() == Some(b'.') {
            self.pos += 1;
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(MoptError::parse(format!(
                    "Missing fractional digits at byte {}",
                    self.pos
                )));
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.pos += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(MoptError::parse(format!(
                    "Missing exponent digits at byte {}",
                    self.pos
                )));
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }
        let slice = &self.src[start..self.pos];
        let num = slice
            .parse::<f64>()
            .map_err(|e| MoptError::parse(format!("Invalid float '{slice}': {e}")))?;
        Ok(JsonValue::Number(num))
    }

    fn parse_hex4(&mut self) -> MoptResult<u16> {
        let mut val: u16 = 0;
        for _ in 0..4 {
            let b = self
                .peek()
                .ok_or_else(|| MoptError::parse("Truncated \\uXXXX escape"))?;
            self.pos += 1;
            let digit = match b {
                b'0'..=b'9' => (b - b'0') as u16,
                b'a'..=b'f' => (b - b'a' + 10) as u16,
                b'A'..=b'F' => (b - b'A' + 10) as u16,
                _ => return Err(MoptError::parse("Invalid hex digit in \\uXXXX escape")),
            };
            val = (val << 4) | digit;
        }
        Ok(val)
    }

    fn parse_string(&mut self) -> MoptResult<String> {
        if self.peek() != Some(b'"') {
            return Err(MoptError::parse("Expected '\"' at start of JSON string"));
        }
        self.pos += 1;
        let mut out = String::new();
        while !self.is_eof() {
            let rest = &self.src[self.pos..];
            let mut chars = rest.chars();
            let ch = chars
                .next()
                .ok_or_else(|| MoptError::parse("Unexpected EOF in string"))?;
            self.pos += ch.len_utf8();
            match ch {
                '"' => return Ok(out),
                '\\' => {
                    let esc = self
                        .peek()
                        .ok_or_else(|| MoptError::parse("Unexpected EOF after '\\'"))?;
                    self.pos += 1;
                    match esc {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\x08'),
                        b'f' => out.push('\x0c'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let code = self.parse_hex4()?;
                            if (0xD800..=0xDBFF).contains(&code) {
                                if self.bytes.get(self.pos..self.pos + 2) != Some(b"\\u") {
                                    return Err(MoptError::parse(
                                        "Unpaired UTF-16 high surrogate in JSON string",
                                    ));
                                }
                                self.pos += 2;
                                let low = self.parse_hex4()?;
                                if !(0xDC00..=0xDFFF).contains(&low) {
                                    return Err(MoptError::parse(
                                        "Invalid UTF-16 low surrogate in JSON string",
                                    ));
                                }
                                let combined = 0x10000
                                    + (((code as u32 - 0xD800) << 10) | (low as u32 - 0xDC00));
                                let c = char::from_u32(combined).ok_or_else(|| {
                                    MoptError::parse("Invalid Unicode scalar from surrogate pair")
                                })?;
                                out.push(c);
                            } else {
                                let c = char::from_u32(code as u32).ok_or_else(|| {
                                    MoptError::parse("Invalid Unicode scalar in \\uXXXX")
                                })?;
                                out.push(c);
                            }
                        }
                        other => {
                            return Err(MoptError::parse(format!(
                                "Invalid escape sequence '\\{}'",
                                other as char
                            )))
                        }
                    }
                }
                c if (c as u32) < 0x20 => {
                    return Err(MoptError::parse(
                        "Unescaped control character in JSON string",
                    ))
                }
                c => out.push(c),
            }
        }
        Err(MoptError::parse("Unterminated JSON string"))
    }

    fn parse_array(&mut self) -> MoptResult<JsonValue> {
        self.pos += 1; // '['
        self.skip_whitespace();
        let mut items = Vec::new();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(JsonValue::Array(items));
        }
        loop {
            items.push(self.parse_value()?);
            self.skip_whitespace();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b']') => {
                    self.pos += 1;
                    return Ok(JsonValue::Array(items));
                }
                _ => {
                    return Err(MoptError::parse(format!(
                        "Expected ',' or ']' in JSON array at byte {}",
                        self.pos
                    )))
                }
            }
        }
    }

    fn parse_object(&mut self) -> MoptResult<JsonValue> {
        self.pos += 1; // '{'
        self.skip_whitespace();
        let mut map = BTreeMap::new();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(JsonValue::Object(map));
        }
        loop {
            self.skip_whitespace();
            let key = self.parse_string()?;
            self.skip_whitespace();
            if self.peek() != Some(b':') {
                return Err(MoptError::parse(format!(
                    "Expected ':' after object key at byte {}",
                    self.pos
                )));
            }
            self.pos += 1;
            let val = self.parse_value()?;
            map.insert(key, val);
            self.skip_whitespace();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b'}') => {
                    self.pos += 1;
                    return Ok(JsonValue::Object(map));
                }
                _ => {
                    return Err(MoptError::parse(format!(
                        "Expected ',' or '}}' in JSON object at byte {}",
                        self.pos
                    )))
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_roundtrip_and_types() {
        let raw = r#"{
            "work_dir": "autots_run",
            "top_n_candidates": 3,
            "active": true,
            "active_atoms": null,
            "step1_settings": {
                "manual_AFIR": ["150.0", "1-3", "5"],
                "temperature_K": 298.15
            }
        }"#;
        let parsed = JsonValue::parse(raw).expect("valid json");
        assert_eq!(
            parsed.get("work_dir").and_then(JsonValue::as_str),
            Some("autots_run")
        );
        assert_eq!(
            parsed.get("top_n_candidates").and_then(JsonValue::as_usize),
            Some(3)
        );
        assert_eq!(
            parsed.get("active").and_then(JsonValue::as_bool),
            Some(true)
        );
        assert!(parsed.get("active_atoms").is_some_and(JsonValue::is_null));

        let pretty = parsed.to_string_pretty();
        let reparsed = JsonValue::parse(&pretty).expect("reparse pretty");
        assert_eq!(parsed, reparsed);
    }
}
