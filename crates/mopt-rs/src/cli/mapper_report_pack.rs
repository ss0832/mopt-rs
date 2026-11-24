//! What the page of a network holds packed: the structures and the paths beyond those it has
//! as text, so that every one can be shown and saved from the page without making it several
//! times as large.
//!
//! Frames of one set of atoms (an xyz structure, the frames of a path) are written compactly
//! before they are compressed: the elements once, the titles of the frames, and the
//! coordinates as whole numbers of 10^-5 Angstrom, those of the first frame as they are and
//! those of every later frame as the difference from the frame before, each as a variable-length
//! number (a difference along a path is small, so most take two bytes). Anything else (the
//! frames of plain variables) is compressed as the text it is. The result is a raw DEFLATE
//! stream in base64, which the script of the page unpacks ([`MAGIC`] tells the two apart).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::fmt::Write as _;

/// The first bytes of frames written compactly.
pub const MAGIC: &[u8] = b"MOPTXYZ1\n";

/// Coordinates are kept to this many units per Angstrom (five decimals).
const UNITS: f64 = 1e5;

/// The largest coordinate that is written compactly (Angstrom); beyond it the text is kept.
const LARGEST: f64 = 1e7;

fn push_number(out: &mut Vec<u8>, value: i64) {
    // Zigzag, then seven bits a byte, the lowest first.
    let mut v = ((value << 1) ^ (value >> 63)) as u64;
    loop {
        let byte = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

/// The xyz frames `text` written compactly; `None` when it is no such text (frames of one
/// number of atoms with the same elements, `symbol x y z` on every atom line).
pub fn compact(text: &str) -> Option<Vec<u8>> {
    let lines: Vec<&str> = text.lines().collect();
    let mut elements: Vec<&str> = Vec::new();
    let mut titles: Vec<&str> = Vec::new();
    let mut frames: Vec<Vec<i64>> = Vec::new();
    let mut at = 0;
    while at < lines.len() {
        if lines[at].trim().is_empty() {
            at += 1;
            continue;
        }
        let n: usize = lines[at].trim().parse().ok()?;
        if n == 0 || at + 2 + n > lines.len() {
            return None;
        }
        let mut coordinates = Vec::with_capacity(3 * n);
        for (k, line) in lines[at + 2..at + 2 + n].iter().enumerate() {
            let fields: Vec<&str> = line.split_whitespace().collect();
            let &[symbol, x, y, z] = fields.as_slice() else {
                return None;
            };
            if frames.is_empty() {
                elements.push(symbol);
            } else if elements.get(k) != Some(&symbol) {
                return None;
            }
            for value in [x, y, z] {
                let v: f64 = value.parse().ok()?;
                if !v.is_finite() || v.abs() > LARGEST {
                    return None;
                }
                coordinates.push((v * UNITS).round() as i64);
            }
        }
        if !frames.is_empty() && elements.len() != n {
            return None;
        }
        titles.push(lines[at + 1]);
        frames.push(coordinates);
        at += 2 + n;
    }
    if frames.is_empty() {
        return None;
    }
    let mut out = MAGIC.to_vec();
    let mut head = String::new();
    let _ = writeln!(head, "{}", elements.len());
    let _ = writeln!(head, "{}", elements.join(" "));
    let _ = writeln!(head, "{}", frames.len());
    for title in &titles {
        // A title is one line; a byte 0 ends the head.
        let _ = writeln!(head, "{}", title.replace('\0', " "));
    }
    out.extend_from_slice(head.as_bytes());
    out.push(0);
    for (k, frame) in frames.iter().enumerate() {
        for (i, &v) in frame.iter().enumerate() {
            let before = if k == 0 { 0 } else { frames[k - 1][i] };
            push_number(&mut out, v - before);
        }
    }
    Some(out)
}

/// The text of frames written by [`compact`] (the coordinates with five decimals), as the
/// script of the page makes it; `None` when the bytes are no such frames.
pub fn expanded(bytes: &[u8]) -> Option<String> {
    let rest = bytes.strip_prefix(MAGIC)?;
    let end = rest.iter().position(|&b| b == 0)?;
    let head = std::str::from_utf8(&rest[..end]).ok()?;
    let mut lines = head.lines();
    let n: usize = lines.next()?.trim().parse().ok()?;
    let elements: Vec<&str> = lines.next()?.split(' ').collect();
    let count: usize = lines.next()?.trim().parse().ok()?;
    let titles: Vec<&str> = lines.collect();
    if elements.len() != n || titles.len() != count {
        return None;
    }
    let mut numbers = rest[end + 1..].iter();
    let mut next = || -> Option<i64> {
        let (mut value, mut shift) = (0_u64, 0_u32);
        loop {
            let byte = *numbers.next()?;
            value |= u64::from(byte & 0x7f).checked_shl(shift)?;
            if byte & 0x80 == 0 {
                break;
            }
            shift += 7;
        }
        Some(((value >> 1) as i64) ^ -((value & 1) as i64))
    };
    let mut out = String::new();
    let mut frame = vec![0_i64; 3 * n];
    for title in titles {
        let _ = write!(out, "{n}\n{title}\n");
        for (a, element) in elements.iter().enumerate() {
            for k in 0..3 {
                frame[3 * a + k] += next()?;
            }
            let _ = writeln!(
                out,
                "{element} {:.5} {:.5} {:.5}",
                frame[3 * a] as f64 / UNITS,
                frame[3 * a + 1] as f64 / UNITS,
                frame[3 * a + 2] as f64 / UNITS
            );
        }
    }
    Some(out)
}

/// `text` packed for the page: compact when it is xyz frames, as it is otherwise; a raw
/// DEFLATE stream in base64.
pub fn packed(text: &str) -> String {
    let bytes = compact(text).unwrap_or_else(|| text.as_bytes().to_vec());
    super::autots_static::base64_bytes(&crate::core_infra::viz::deflate::deflate_fixed(&bytes))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    const PATH: &str = "3\nbackward energy=-1.5\nO 0.00000 0.00000 0.11779\nH 0.00000 0.75545 \
                        -0.47116\nH 0.00000 -0.75545 -0.47116\n3\nTS energy=-1.4\nO 0.00100 \
                        0.00000 0.12000\nH 0.00000 0.76000 -0.47000\nH -0.00200 -0.75000 \
                        -0.48000\n";

    #[test]
    fn frames_come_back_as_they_were_to_five_decimals() {
        let text = PATH;
        let bytes = compact(text).unwrap();
        assert!(bytes.starts_with(MAGIC));
        assert_eq!(expanded(&bytes).unwrap(), text);
        // More decimals than five are rounded, and other spacing is that of the page.
        let loose = "2\n\nAr   0.1234567  -1.0   2\nAr 3.80 0 0\n";
        assert_eq!(
            expanded(&compact(loose).unwrap()).unwrap(),
            "2\n\nAr 0.12346 -1.00000 2.00000\nAr 3.80000 0.00000 0.00000\n"
        );
    }

    #[test]
    fn a_path_takes_a_few_bytes_per_atom() {
        // 40 frames of 12 atoms that move by a few hundredths of an Angstrom a frame.
        let mut text = String::new();
        for f in 0..40 {
            let _ = writeln!(text, "12\nframe {f}");
            for a in 0..12 {
                let (a, f) = (f64::from(a), f64::from(f));
                let _ = writeln!(
                    text,
                    "C {:.5} {:.5} {:.5}",
                    1.4 * a + 0.013 * f,
                    0.7 * a.sin() - 0.021 * f,
                    0.3 * a.cos() + 0.008 * f * (a + 1.0).sin()
                );
            }
        }
        let bytes = compact(&text).unwrap();
        assert_eq!(expanded(&bytes).unwrap(), text);
        let atom_lines = 40 * 12;
        // The numbers alone: at most two bytes a coordinate after the first frame.
        assert!(bytes.len() < 7 * atom_lines + 400, "{}", bytes.len());
        // In the page (compressed, in base64): less than two fifths of the text.
        assert!(
            5 * packed(&text).len() < 2 * text.len(),
            "{} against {}",
            packed(&text).len(),
            text.len()
        );
    }

    #[test]
    fn what_is_no_xyz_text_is_kept_as_text() {
        // Frames of plain variables, frames of different atoms, an atom line with a field
        // more: none is written compactly.
        for text in [
            "2\nframe\nx 0.5\ny 1.5\n",
            "1\na\nH 0 0 0\n1\nb\nHe 0 0 0\n",
            "1\na\nH 0 0 0 0.3\n",
            "",
        ] {
            assert!(compact(text).is_none(), "{text:?}");
        }
        assert!(expanded(b"3\nwater\n").is_none());
        // Negative numbers and large ones come back.
        let far = "1\nfar\nXe -12345.67890 0.00001 -0.00001\n";
        assert_eq!(expanded(&compact(far).unwrap()).unwrap(), far);
    }
}
