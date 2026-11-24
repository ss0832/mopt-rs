//! PNG (RFC 2083) encoder for 8-bit RGB images.
//!
//! MultiOptPy: no counterpart (new in mopt-rs) (MultiOptPy saves PNG through matplotlib).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use super::checksum::crc32;
use super::deflate::zlib_compress;

/// PNG file signature.
pub const PNG_SIGNATURE: [u8; 8] = [137, 80, 78, 71, 13, 10, 26, 10];

/// Encode a row-major RGB8 buffer (`width * height * 3` bytes) as PNG.
///
/// Each scanline uses filter type 1 (Sub), which turns horizontal runs into zero bytes that
/// compress well.
pub fn encode_rgb8(width: u32, height: u32, rgb: &[u8]) -> Vec<u8> {
    let row_bytes = width as usize * 3;
    debug_assert_eq!(rgb.len(), row_bytes * height as usize);
    let mut filtered = Vec::with_capacity((row_bytes + 1) * height as usize);
    for row in rgb.chunks_exact(row_bytes.max(1)).take(height as usize) {
        filtered.push(1u8);
        for (k, &v) in row.iter().enumerate() {
            let left = if k >= 3 { row[k - 3] } else { 0 };
            filtered.push(v.wrapping_sub(left));
        }
    }

    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&height.to_be_bytes());
    // bit depth 8, color type 2 (RGB), deflate, adaptive filtering, no interlace
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);

    let idat = zlib_compress(&filtered);
    let mut png = Vec::with_capacity(idat.len() + 64);
    png.extend_from_slice(&PNG_SIGNATURE);
    write_chunk(&mut png, b"IHDR", &ihdr);
    write_chunk(&mut png, b"IDAT", &idat);
    write_chunk(&mut png, b"IEND", &[]);
    png
}

fn write_chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    let start = out.len();
    out.extend_from_slice(kind);
    out.extend_from_slice(data);
    let crc = crc32(&out[start..]);
    out.extend_from_slice(&crc.to_be_bytes());
}

#[cfg(test)]
mod tests {
    use super::super::deflate::inflate::zlib_decompress;
    use super::*;

    /// Parse chunks, verify CRCs, and undo the Sub filter.
    fn decode(png: &[u8]) -> (u32, u32, Vec<u8>) {
        assert_eq!(&png[..8], &PNG_SIGNATURE);
        let mut pos = 8;
        let (mut w, mut h, mut idat) = (0, 0, Vec::new());
        loop {
            let len = u32::from_be_bytes(png[pos..pos + 4].try_into().unwrap()) as usize;
            let kind = &png[pos + 4..pos + 8];
            let data = &png[pos + 8..pos + 8 + len];
            let crc = u32::from_be_bytes(png[pos + 8 + len..pos + 12 + len].try_into().unwrap());
            assert_eq!(crc, crc32(&png[pos + 4..pos + 8 + len]));
            match kind {
                b"IHDR" => {
                    w = u32::from_be_bytes(data[0..4].try_into().unwrap());
                    h = u32::from_be_bytes(data[4..8].try_into().unwrap());
                    assert_eq!(&data[8..], &[8, 2, 0, 0, 0]);
                }
                b"IDAT" => idat.extend_from_slice(data),
                b"IEND" => break,
                _ => panic!("unexpected chunk"),
            }
            pos += 12 + len;
        }
        let raw = zlib_decompress(&idat);
        let row = w as usize * 3;
        let mut rgb = Vec::new();
        for line in raw.chunks_exact(row + 1) {
            assert_eq!(line[0], 1);
            let start = rgb.len();
            for k in 0..row {
                let left = if k >= 3 { rgb[start + k - 3] } else { 0u8 };
                rgb.push(line[1 + k].wrapping_add(left));
            }
        }
        (w, h, rgb)
    }

    #[test]
    fn png_round_trip() {
        let (w, h) = (37u32, 11u32);
        let rgb: Vec<u8> = (0..w * h * 3).map(|i| (i * 7 % 251) as u8).collect();
        let (dw, dh, back) = decode(&encode_rgb8(w, h, &rgb));
        assert_eq!((dw, dh), (w, h));
        assert_eq!(back, rgb);
    }
}
