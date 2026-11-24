//! SQLite database files without a library: writing a whole database of tables, and reading
//! the rows of the tables of any database file in the rollback-journal format (also one written
//! or changed by SQLite itself).
//!
//! The file format is the published one (<https://www.sqlite.org/fileformat2.html>): pages of
//! 4096 bytes, the 100-byte header on page 1, the schema table `sqlite_schema` rooted on page 1,
//! and every table a B-tree of table pages (interior pages 0x05, leaf pages 0x0d) whose cells
//! hold records (a header of serial types, then the values), with the part of a payload that
//! does not fit on its page in a chain of overflow pages. The writer builds every table bottom
//! up, leaves first and then the interior levels, with the rows in the order of their rowids;
//! it writes no indexes, so a table is declared with `INTEGER PRIMARY KEY` as its only key (an
//! alias of the rowid, which needs no index). SQLite opens the files, and `PRAGMA
//! integrity_check` finds them sound (tested from Python's `sqlite3`).
//!
//! The reader follows the B-trees of the tables (interior and leaf pages, overflow chains); it
//! does not read indexes, the write-ahead log of a database in WAL mode (whose changes since the
//! last checkpoint it does not see), or tables `WITHOUT ROWID`.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use crate::error::{MoptError, MoptResult};

/// The page size of the files written.
pub const PAGE_SIZE: usize = 4096;

/// A value of a column.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// NULL.
    Null,
    /// A signed integer.
    Integer(i64),
    /// A float.
    Real(f64),
    /// UTF-8 text.
    Text(String),
    /// Bytes.
    Blob(Vec<u8>),
}

impl Value {
    /// The integer, or a real that is a whole number.
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Self::Integer(i) => Some(*i),
            Self::Real(f) if f.fract() == 0.0 && f.is_finite() => Some(*f as i64),
            _ => None,
        }
    }

    /// The number (an integer as a float).
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Integer(i) => Some(*i as f64),
            Self::Real(f) => Some(*f),
            _ => None,
        }
    }

    /// The text.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Text(s) => Some(s),
            _ => None,
        }
    }

    /// The bytes.
    pub fn as_blob(&self) -> Option<&[u8]> {
        match self {
            Self::Blob(b) => Some(b),
            _ => None,
        }
    }
}

/// A table to write: its name, its `CREATE TABLE` statement, and its rows as `(rowid,
/// values)` (the column that is the `INTEGER PRIMARY KEY` holds `Null`: SQLite keeps it as the
/// rowid).
#[derive(Debug, Clone, PartialEq)]
pub struct Table {
    /// The name.
    pub name: String,
    /// The statement that creates it.
    pub sql: String,
    /// The rows, in any order (they are written in the order of their rowids).
    pub rows: Vec<(i64, Vec<Value>)>,
}

fn put_varint(out: &mut Vec<u8>, v: u64) {
    if v > 0x00ff_ffff_ffff_ffff {
        let mut buf = [0u8; 9];
        buf[8] = v as u8;
        let mut x = v >> 8;
        for b in buf[..8].iter_mut().rev() {
            *b = ((x & 0x7f) as u8) | 0x80;
            x >>= 7;
        }
        out.extend_from_slice(&buf);
        return;
    }
    let mut groups = Vec::with_capacity(9);
    let mut x = v;
    loop {
        groups.push((x & 0x7f) as u8);
        x >>= 7;
        if x == 0 {
            break;
        }
    }
    groups.reverse();
    let last = groups.len() - 1;
    for (k, g) in groups.iter_mut().enumerate() {
        if k < last {
            *g |= 0x80;
        }
    }
    out.extend_from_slice(&groups);
}

fn varint_len(v: u64) -> usize {
    let mut out = Vec::with_capacity(9);
    put_varint(&mut out, v);
    out.len()
}

fn get_varint(b: &[u8]) -> Option<(u64, usize)> {
    let mut v = 0u64;
    for i in 0..8 {
        let c = *b.get(i)?;
        v = (v << 7) | u64::from(c & 0x7f);
        if c & 0x80 == 0 {
            return Some((v, i + 1));
        }
    }
    let c = *b.get(8)?;
    Some(((v << 8) | u64::from(c), 9))
}

/// The serial type and the body of `value`.
fn serial(value: &Value) -> (u64, Vec<u8>) {
    match value {
        Value::Null => (0, Vec::new()),
        Value::Integer(0) => (8, Vec::new()),
        Value::Integer(1) => (9, Vec::new()),
        Value::Integer(i) => {
            let i = *i;
            let (t, n) = if (-128..=127).contains(&i) {
                (1, 1)
            } else if (-32768..=32767).contains(&i) {
                (2, 2)
            } else if (-8_388_608..=8_388_607).contains(&i) {
                (3, 3)
            } else if (-2_147_483_648..=2_147_483_647).contains(&i) {
                (4, 4)
            } else if (-140_737_488_355_328..=140_737_488_355_327).contains(&i) {
                (5, 6)
            } else {
                (6, 8)
            };
            (t, i.to_be_bytes()[8 - n..].to_vec())
        }
        Value::Real(f) => (7, f.to_bits().to_be_bytes().to_vec()),
        Value::Text(s) => (13 + 2 * s.len() as u64, s.as_bytes().to_vec()),
        Value::Blob(b) => (12 + 2 * b.len() as u64, b.clone()),
    }
}

/// The record of `values`.
fn record(values: &[Value]) -> Vec<u8> {
    let parts: Vec<(u64, Vec<u8>)> = values.iter().map(serial).collect();
    let types: usize = parts.iter().map(|(t, _)| varint_len(*t)).sum();
    // The size of the header counts its own varint.
    let mut size = types + 1;
    while varint_len(size as u64) + types != size {
        size = varint_len(size as u64) + types;
    }
    let mut out = Vec::new();
    put_varint(&mut out, size as u64);
    for (t, _) in &parts {
        put_varint(&mut out, *t);
    }
    for (_, body) in parts {
        out.extend_from_slice(&body);
    }
    out
}

/// The values of a record.
fn decode_record(payload: &[u8]) -> MoptResult<Vec<Value>> {
    let bad = || MoptError::parse("SQLite: a record that cannot be read");
    let (size, mut at) = get_varint(payload).ok_or_else(bad)?;
    let size = size as usize;
    let mut types = Vec::new();
    while at < size {
        let (t, n) = get_varint(payload.get(at..).ok_or_else(bad)?).ok_or_else(bad)?;
        types.push(t);
        at += n;
    }
    let mut body = size;
    let mut out = Vec::with_capacity(types.len());
    for t in types {
        let take = |n: usize, body: &mut usize| -> MoptResult<&[u8]> {
            let part = payload.get(*body..*body + n).ok_or_else(bad)?;
            *body += n;
            Ok(part)
        };
        let int = |bytes: &[u8]| {
            let mut v: i64 = if bytes.first().is_some_and(|b| b & 0x80 != 0) {
                -1
            } else {
                0
            };
            for &b in bytes {
                v = (v << 8) | i64::from(b);
            }
            v
        };
        let value = match t {
            0 => Value::Null,
            1..=4 => Value::Integer(int(take(t as usize, &mut body)?)),
            5 => Value::Integer(int(take(6, &mut body)?)),
            6 => Value::Integer(int(take(8, &mut body)?)),
            7 => {
                let b = take(8, &mut body)?;
                let mut a = [0u8; 8];
                a.copy_from_slice(b);
                Value::Real(f64::from_bits(u64::from_be_bytes(a)))
            }
            8 => Value::Integer(0),
            9 => Value::Integer(1),
            t if t >= 12 && t % 2 == 0 => {
                Value::Blob(take(((t - 12) / 2) as usize, &mut body)?.to_vec())
            }
            t if t >= 13 => {
                let b = take(((t - 13) / 2) as usize, &mut body)?;
                Value::Text(String::from_utf8_lossy(b).into_owned())
            }
            _ => return Err(bad()),
        };
        out.push(value);
    }
    Ok(out)
}

/// How much of a payload of `p` bytes stays on a table leaf page of `usable` bytes.
fn local_size(p: usize, usable: usize) -> usize {
    let x = usable - 35;
    if p <= x {
        return p;
    }
    let m = ((usable - 12) * 32 / 255) - 23;
    let k = m + ((p - m) % (usable - 4));
    if k <= x {
        k
    } else {
        m
    }
}

/// The pages of a database being written (page 1 is index 0).
struct Pages {
    pages: Vec<Vec<u8>>,
}

impl Pages {
    fn add(&mut self, page: Vec<u8>) -> u32 {
        self.pages.push(page);
        self.pages.len() as u32
    }

    /// The cell of a leaf for `rowid` and `payload`, with its overflow pages written.
    fn leaf_cell(&mut self, rowid: i64, payload: &[u8]) -> Vec<u8> {
        let usable = PAGE_SIZE;
        let local = local_size(payload.len(), usable);
        let mut cell = Vec::with_capacity(local + 22);
        put_varint(&mut cell, payload.len() as u64);
        put_varint(&mut cell, rowid as u64);
        cell.extend_from_slice(&payload[..local]);
        if local < payload.len() {
            // The chain of overflow pages, written in order: each holds the number of the
            // next one (0 for the last) and up to `usable - 4` bytes.
            let rest = &payload[local..];
            let chunk = usable - 4;
            let count = rest.len().div_ceil(chunk);
            let first = self.pages.len() as u32 + 1;
            for k in 0..count {
                let mut page = vec![0u8; PAGE_SIZE];
                let next = if k + 1 < count {
                    first + k as u32 + 1
                } else {
                    0
                };
                page[..4].copy_from_slice(&next.to_be_bytes());
                let part = &rest[k * chunk..((k + 1) * chunk).min(rest.len())];
                page[4..4 + part.len()].copy_from_slice(part);
                self.add(page);
            }
            cell.extend_from_slice(&first.to_be_bytes());
        }
        cell
    }
}

/// A B-tree page of `kind` (0x0d leaf, 0x05 interior) with `cells` from `start` (100 on page 1);
/// `right` is the right-most child of an interior page.
fn btree_page(kind: u8, cells: &[Vec<u8>], start: usize, right: u32, page: &mut [u8]) {
    let header = if kind == 0x05 { 12 } else { 8 };
    let mut end = PAGE_SIZE;
    let mut pointers = Vec::with_capacity(cells.len());
    for c in cells {
        end -= c.len();
        page[end..end + c.len()].copy_from_slice(c);
        pointers.push(end as u16);
    }
    page[start] = kind;
    page[start + 1..start + 3].copy_from_slice(&0u16.to_be_bytes());
    page[start + 3..start + 5].copy_from_slice(&(cells.len() as u16).to_be_bytes());
    let content = if end == 65536 { 0 } else { end as u16 };
    page[start + 5..start + 7].copy_from_slice(&content.to_be_bytes());
    page[start + 7] = 0;
    if kind == 0x05 {
        page[start + 8..start + 12].copy_from_slice(&right.to_be_bytes());
    }
    for (k, p) in pointers.iter().enumerate() {
        let at = start + header + 2 * k;
        page[at..at + 2].copy_from_slice(&p.to_be_bytes());
    }
}

/// Write the B-tree of `rows` (sorted by rowid) and return its root page.
fn write_tree(pages: &mut Pages, rows: &[(i64, Vec<u8>)]) -> u32 {
    // Leaves, filled in order.
    let mut level: Vec<(u32, i64)> = Vec::new();
    let mut cells: Vec<Vec<u8>> = Vec::new();
    let mut used = 8;
    let mut last = 0;
    let flush =
        |pages: &mut Pages, cells: &mut Vec<Vec<u8>>, level: &mut Vec<(u32, i64)>, last: i64| {
            let mut page = vec![0u8; PAGE_SIZE];
            btree_page(0x0d, cells, 0, 0, &mut page);
            let n = pages.add(page);
            level.push((n, last));
            cells.clear();
        };
    for (rowid, payload) in rows {
        let cell = pages.leaf_cell(*rowid, payload);
        if used + cell.len() + 2 > PAGE_SIZE && !cells.is_empty() {
            flush(pages, &mut cells, &mut level, last);
            used = 8;
        }
        used += cell.len() + 2;
        cells.push(cell);
        last = *rowid;
    }
    if !cells.is_empty() || level.is_empty() {
        flush(pages, &mut cells, &mut level, last);
    }
    // Interior levels until one page is left: a cell per child but the last, which is the
    // right-most pointer; the key of a cell is the largest rowid of its child.
    while level.len() > 1 {
        let mut next: Vec<(u32, i64)> = Vec::new();
        let mut group: Vec<(u32, i64)> = Vec::new();
        let mut used = 12;
        let close = |pages: &mut Pages, group: &mut Vec<(u32, i64)>, next: &mut Vec<(u32, i64)>| {
            let Some(&(right, key)) = group.last() else {
                return;
            };
            let cells: Vec<Vec<u8>> = group[..group.len() - 1]
                .iter()
                .map(|&(child, k)| {
                    let mut c = child.to_be_bytes().to_vec();
                    put_varint(&mut c, k as u64);
                    c
                })
                .collect();
            let mut page = vec![0u8; PAGE_SIZE];
            btree_page(0x05, &cells, 0, right, &mut page);
            let n = pages.add(page);
            next.push((n, key));
            group.clear();
        };
        for &(child, key) in &level {
            let size = 4 + varint_len(key as u64) + 2;
            if used + size > PAGE_SIZE && group.len() > 1 {
                close(pages, &mut group, &mut next);
                used = 12;
            }
            used += size;
            group.push((child, key));
        }
        close(pages, &mut group, &mut next);
        level = next;
    }
    level.first().map_or(0, |&(n, _)| n)
}

/// The bytes of a database with `tables`.
pub fn write_database(tables: &[Table]) -> MoptResult<Vec<u8>> {
    let mut pages = Pages {
        pages: vec![vec![0u8; PAGE_SIZE]],
    };
    let mut schema = Vec::new();
    for (k, t) in tables.iter().enumerate() {
        let mut rows: Vec<(i64, Vec<u8>)> = t
            .rows
            .iter()
            .map(|(rowid, values)| (*rowid, record(values)))
            .collect();
        rows.sort_by_key(|(r, _)| *r);
        if rows.windows(2).any(|w| w[0].0 == w[1].0) {
            return Err(MoptError::config(format!(
                "SQLite: table {} has two rows of one rowid",
                t.name
            )));
        }
        let root = write_tree(&mut pages, &rows);
        let values = vec![
            Value::Text("table".to_string()),
            Value::Text(t.name.clone()),
            Value::Text(t.name.clone()),
            Value::Integer(i64::from(root)),
            Value::Text(t.sql.clone()),
        ];
        schema.push((k as i64 + 1, record(&values)));
    }
    // The schema on page 1, after the header of the file.
    let cells: Vec<Vec<u8>> = schema
        .iter()
        .map(|(rowid, payload)| pages.leaf_cell(*rowid, payload))
        .collect();
    let size: usize = 100 + 8 + cells.iter().map(|c| c.len() + 2).sum::<usize>();
    if size > PAGE_SIZE {
        return Err(MoptError::config(
            "SQLite: the schema of the tables does not fit on the first page",
        ));
    }
    let count = pages.pages.len() as u32;
    let page = &mut pages.pages[0];
    btree_page(0x0d, &cells, 100, 0, page);
    let h = &mut page[..100];
    h[..16].copy_from_slice(b"SQLite format 3\0");
    h[16..18].copy_from_slice(&(PAGE_SIZE as u16).to_be_bytes());
    h[18] = 1;
    h[19] = 1;
    h[20] = 0;
    h[21] = 64;
    h[22] = 32;
    h[23] = 32;
    h[24..28].copy_from_slice(&1u32.to_be_bytes());
    h[28..32].copy_from_slice(&count.to_be_bytes());
    h[40..44].copy_from_slice(&1u32.to_be_bytes());
    h[44..48].copy_from_slice(&4u32.to_be_bytes());
    h[56..60].copy_from_slice(&1u32.to_be_bytes());
    h[92..96].copy_from_slice(&1u32.to_be_bytes());
    h[96..100].copy_from_slice(&3_045_000u32.to_be_bytes());
    Ok(pages.pages.concat())
}

/// A database file being read.
pub struct Database {
    bytes: Vec<u8>,
    page_size: usize,
    usable: usize,
}

/// A table of a database: its name, root page, and statement.
#[derive(Debug, Clone, PartialEq)]
pub struct TableInfo {
    /// The name.
    pub name: String,
    /// The root page.
    pub root: u32,
    /// The statement that created it.
    pub sql: String,
}

impl Database {
    /// The database of the file contents `bytes`.
    pub fn open(bytes: Vec<u8>) -> MoptResult<Self> {
        if bytes.len() < 100 || &bytes[..16] != b"SQLite format 3\0" {
            return Err(MoptError::parse("not an SQLite database file"));
        }
        let raw = usize::from(u16::from_be_bytes([bytes[16], bytes[17]]));
        let page_size = if raw == 1 { 65536 } else { raw };
        if page_size < 512 || !page_size.is_power_of_two() {
            return Err(MoptError::parse("SQLite: the page size cannot be read"));
        }
        let usable = page_size - usize::from(bytes[20]);
        Ok(Self {
            bytes,
            page_size,
            usable,
        })
    }

    fn page(&self, n: u32) -> MoptResult<&[u8]> {
        let n = n as usize;
        if n == 0 {
            return Err(MoptError::parse("SQLite: a page number 0"));
        }
        self.bytes
            .get((n - 1) * self.page_size..n * self.page_size)
            .ok_or_else(|| MoptError::parse(format!("SQLite: page {n} is beyond the file")))
    }

    /// The payload of the cell at `at` of a leaf page, with its overflow chain.
    fn leaf_payload(&self, page: &[u8], at: usize) -> MoptResult<(i64, Vec<u8>)> {
        let bad = || MoptError::parse("SQLite: a cell that cannot be read");
        let (size, n1) = get_varint(page.get(at..).ok_or_else(bad)?).ok_or_else(bad)?;
        let (rowid, n2) = get_varint(page.get(at + n1..).ok_or_else(bad)?).ok_or_else(bad)?;
        let size = size as usize;
        let start = at + n1 + n2;
        let local = local_size(size, self.usable);
        let mut payload = page.get(start..start + local).ok_or_else(bad)?.to_vec();
        if local < size {
            let pointer = page.get(start + local..start + local + 4).ok_or_else(bad)?;
            let mut next = u32::from_be_bytes([pointer[0], pointer[1], pointer[2], pointer[3]]);
            let mut guard = 0;
            while payload.len() < size {
                guard += 1;
                if next == 0 || guard > self.bytes.len() / self.page_size + 1 {
                    return Err(MoptError::parse("SQLite: an overflow chain is broken"));
                }
                let p = self.page(next)?;
                let take = (size - payload.len()).min(self.usable - 4);
                payload.extend_from_slice(&p[4..4 + take]);
                next = u32::from_be_bytes([p[0], p[1], p[2], p[3]]);
            }
        }
        Ok((rowid as i64, payload))
    }

    /// The rows of the table B-tree rooted at `root`, in the order of their rowids.
    pub fn rows_at(&self, root: u32) -> MoptResult<Vec<(i64, Vec<Value>)>> {
        let mut out = Vec::new();
        let mut stack = vec![root];
        let mut visited = 0usize;
        while let Some(n) = stack.pop() {
            visited += 1;
            if visited > self.bytes.len() / self.page_size + 1 {
                return Err(MoptError::parse("SQLite: a B-tree with a loop"));
            }
            let page = self.page(n)?;
            let start = if n == 1 { 100 } else { 0 };
            let kind = page[start];
            let cells = usize::from(u16::from_be_bytes([page[start + 3], page[start + 4]]));
            let header = match kind {
                0x0d => 8,
                0x05 => 12,
                other => {
                    return Err(MoptError::parse(format!(
                        "SQLite: page {n} is not a page of a table (type {other:#x})"
                    )))
                }
            };
            let pointer = |k: usize| -> MoptResult<usize> {
                let at = start + header + 2 * k;
                page.get(at..at + 2)
                    .map(|b| usize::from(u16::from_be_bytes([b[0], b[1]])))
                    .ok_or_else(|| MoptError::parse("SQLite: a cell pointer beyond the page"))
            };
            if kind == 0x0d {
                for k in 0..cells {
                    let (rowid, payload) = self.leaf_payload(page, pointer(k)?)?;
                    out.push((rowid, decode_record(&payload)?));
                }
            } else {
                let right = u32::from_be_bytes([
                    page[start + 8],
                    page[start + 9],
                    page[start + 10],
                    page[start + 11],
                ]);
                // Children pushed so that the left-most is read first.
                stack.push(right);
                for k in (0..cells).rev() {
                    let at = pointer(k)?;
                    let b = page
                        .get(at..at + 4)
                        .ok_or_else(|| MoptError::parse("SQLite: a cell beyond the page"))?;
                    stack.push(u32::from_be_bytes([b[0], b[1], b[2], b[3]]));
                }
            }
        }
        Ok(out)
    }

    /// The tables of the database (from `sqlite_schema`).
    pub fn tables(&self) -> MoptResult<Vec<TableInfo>> {
        Ok(self
            .rows_at(1)?
            .into_iter()
            .filter_map(|(_, v)| {
                (v.first()?.as_str()? == "table").then(|| TableInfo {
                    name: v.get(1).and_then(Value::as_str).unwrap_or("").to_string(),
                    root: v.get(3).and_then(Value::as_i64).unwrap_or(0) as u32,
                    sql: v.get(4).and_then(Value::as_str).unwrap_or("").to_string(),
                })
            })
            .collect())
    }

    /// The rows of the table `name` (`None` when there is no such table).
    pub fn rows(&self, name: &str) -> MoptResult<Option<Vec<(i64, Vec<Value>)>>> {
        let Some(t) = self.tables()?.into_iter().find(|t| t.name == name) else {
            return Ok(None);
        };
        if t.root == 0 {
            return Ok(Some(Vec::new()));
        }
        self.rows_at(t.root).map(Some)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn varints_and_records_go_there_and_back() {
        for v in [
            0u64,
            1,
            127,
            128,
            240,
            2287,
            67823,
            1 << 35,
            u64::MAX >> 9,
            u64::MAX,
        ] {
            let mut b = Vec::new();
            put_varint(&mut b, v);
            assert_eq!(get_varint(&b), Some((v, b.len())), "{v}");
        }
        let values = vec![
            Value::Null,
            Value::Integer(0),
            Value::Integer(1),
            Value::Integer(-5),
            Value::Integer(40_000),
            Value::Integer(-3_000_000_000),
            Value::Integer(i64::MIN),
            Value::Real(-1.25e-7),
            Value::Text("EQ0 énergie".to_string()),
            Value::Blob(vec![0, 1, 255]),
        ];
        assert_eq!(decode_record(&record(&values)).unwrap(), values);
    }

    #[test]
    fn a_database_of_many_and_long_rows_is_read_back() {
        let big = "x".repeat(10_000);
        let rows: Vec<(i64, Vec<Value>)> = (1..=3000)
            .map(|i| {
                let text = if i % 500 == 0 {
                    big.clone()
                } else {
                    format!("row {i}")
                };
                (
                    i,
                    vec![Value::Null, Value::Text(text), Value::Real(i as f64 * 0.5)],
                )
            })
            .collect();
        let tables = vec![
            Table {
                name: "t".to_string(),
                sql: "CREATE TABLE t(id INTEGER PRIMARY KEY, name TEXT, x REAL)".to_string(),
                rows: rows.clone(),
            },
            Table {
                name: "empty".to_string(),
                sql: "CREATE TABLE empty(id INTEGER PRIMARY KEY)".to_string(),
                rows: Vec::new(),
            },
        ];
        let bytes = write_database(&tables).unwrap();
        assert_eq!(bytes.len() % PAGE_SIZE, 0);
        // A copy for checks with SQLite itself (tests/test_sqlite_files.py).
        let _ = std::fs::write(std::env::temp_dir().join("mopt_sqlite_unit.db"), &bytes);
        let db = Database::open(bytes).unwrap();
        let names: Vec<String> = db.tables().unwrap().into_iter().map(|t| t.name).collect();
        assert_eq!(names, vec!["t", "empty"]);
        assert_eq!(db.rows("t").unwrap().unwrap(), rows);
        assert_eq!(db.rows("empty").unwrap().unwrap(), vec![]);
        assert!(db.rows("none").unwrap().is_none());
        assert!(Database::open(b"not a database".to_vec()).is_err());
    }

    #[test]
    fn a_file_written_and_changed_by_sqlite_is_read() {
        // tools/golden/sqlite_fixture.py: pages of 1024 bytes, rows deleted and updated, an
        // index, rows on overflow pages.
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/golden/sqlite_written_by_sqlite3.db"
        );
        let db = Database::open(std::fs::read(path).unwrap()).unwrap();
        let names: Vec<String> = db.tables().unwrap().into_iter().map(|t| t.name).collect();
        assert_eq!(names, vec!["nodes", "notes"]);
        let rows = db.rows("nodes").unwrap().unwrap();
        let expected: Vec<i64> = (1..=2000).filter(|i| i % 7 != 0).collect();
        assert_eq!(rows.iter().map(|(r, _)| *r).collect::<Vec<_>>(), expected);
        for (i, v) in &rows {
            let label = if i % 3 != 0 {
                format!("EQ{i}")
            } else {
                format!("TS{i}")
            };
            let label = if i % 11 == 0 {
                format!("{label}_updated")
            } else {
                label
            };
            assert_eq!(v[0], Value::Null, "{i}");
            assert_eq!(v[1].as_str(), Some(label.as_str()), "{i}");
            assert_eq!(v[2].as_f64(), Some(-0.001 * *i as f64), "{i}");
            let n = if i % 250 == 0 { 3000 } else { 8 };
            let data: Vec<u8> = (0..n).map(|k| ((i * k) % 256) as u8).collect();
            assert_eq!(v[3].as_blob(), Some(data.as_slice()), "{i}");
        }
        let notes = db.rows("notes").unwrap().unwrap();
        assert_eq!(notes[1].1[1].as_str().map(|s| s.chars().count()), Some(600));
        assert_eq!(notes[2].1[1], Value::Null);
    }
}
