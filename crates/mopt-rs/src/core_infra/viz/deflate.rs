//! zlib (RFC 1950) streams compressed with DEFLATE (RFC 1951) fixed-Huffman blocks.
//!
//! The encoder uses LZ77 with a hash-chain match finder over a 32 KiB window and emits a single
//! fixed-Huffman block. Plot rasters are dominated by long runs of identical pixels, for which
//! this already gives large savings without dynamic Huffman tables.
//!
//! MultiOptPy: no counterpart (new in mopt-rs) (PNG compression; MultiOptPy uses matplotlib).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use super::checksum::adler32;

const WINDOW: usize = 32 * 1024;
const MIN_MATCH: usize = 3;
const MAX_MATCH: usize = 258;
const HASH_BITS: u32 = 15;
const MAX_CHAIN: usize = 64;

/// Length code table: (base length, extra bits) for codes 257..=285.
const LENGTH_BASE: [(u16, u8); 29] = [
    (3, 0),
    (4, 0),
    (5, 0),
    (6, 0),
    (7, 0),
    (8, 0),
    (9, 0),
    (10, 0),
    (11, 1),
    (13, 1),
    (15, 1),
    (17, 1),
    (19, 2),
    (23, 2),
    (27, 2),
    (31, 2),
    (35, 3),
    (43, 3),
    (51, 3),
    (59, 3),
    (67, 4),
    (83, 4),
    (99, 4),
    (115, 4),
    (131, 5),
    (163, 5),
    (195, 5),
    (227, 5),
    (258, 0),
];

/// Distance code table: (base distance, extra bits) for codes 0..=29.
const DIST_BASE: [(u16, u8); 30] = [
    (1, 0),
    (2, 0),
    (3, 0),
    (4, 0),
    (5, 1),
    (7, 1),
    (9, 2),
    (13, 2),
    (17, 3),
    (25, 3),
    (33, 4),
    (49, 4),
    (65, 5),
    (97, 5),
    (129, 6),
    (193, 6),
    (257, 7),
    (385, 7),
    (513, 8),
    (769, 8),
    (1025, 9),
    (1537, 9),
    (2049, 10),
    (3073, 10),
    (4097, 11),
    (6145, 11),
    (8193, 12),
    (12289, 12),
    (16385, 13),
    (24577, 13),
];

/// LSB-first bit writer as required by DEFLATE.
struct BitWriter {
    out: Vec<u8>,
    acc: u64,
    nbits: u32,
}

impl BitWriter {
    fn new(out: Vec<u8>) -> Self {
        Self {
            out,
            acc: 0,
            nbits: 0,
        }
    }

    fn put(&mut self, value: u32, nbits: u32) {
        self.acc |= u64::from(value) << self.nbits;
        self.nbits += nbits;
        while self.nbits >= 8 {
            self.out.push((self.acc & 0xFF) as u8);
            self.acc >>= 8;
            self.nbits -= 8;
        }
    }

    /// Write a Huffman code, which DEFLATE stores most-significant bit first.
    fn put_code(&mut self, code: u32, len: u32) {
        let mut rev = 0;
        for i in 0..len {
            rev |= ((code >> i) & 1) << (len - 1 - i);
        }
        self.put(rev, len);
    }

    fn finish(mut self) -> Vec<u8> {
        if self.nbits > 0 {
            self.out.push((self.acc & 0xFF) as u8);
        }
        self.out
    }
}

/// Fixed-Huffman literal/length code for symbol `sym` (RFC 1951, 3.2.6).
fn fixed_litlen_code(sym: u16) -> (u32, u32) {
    let s = u32::from(sym);
    match sym {
        0..=143 => (0x30 + s, 8),
        144..=255 => (0x190 + (s - 144), 9),
        256..=279 => (s - 256, 7),
        _ => (0xC0 + (s - 280), 8),
    }
}

fn emit_literal(w: &mut BitWriter, byte: u8) {
    let (code, len) = fixed_litlen_code(u16::from(byte));
    w.put_code(code, len);
}

fn emit_match(w: &mut BitWriter, length: usize, distance: usize) {
    let li = LENGTH_BASE
        .iter()
        .rposition(|&(base, _)| usize::from(base) <= length)
        .unwrap_or(0);
    let (lbase, lextra) = LENGTH_BASE[li];
    let (code, len) = fixed_litlen_code(257 + li as u16);
    w.put_code(code, len);
    w.put((length - usize::from(lbase)) as u32, u32::from(lextra));

    let di = DIST_BASE
        .iter()
        .rposition(|&(base, _)| usize::from(base) <= distance)
        .unwrap_or(0);
    let (dbase, dextra) = DIST_BASE[di];
    w.put_code(di as u32, 5);
    w.put((distance - usize::from(dbase)) as u32, u32::from(dextra));
}

#[inline]
fn hash3(data: &[u8], i: usize) -> usize {
    let v = u32::from(data[i]) << 16 | u32::from(data[i + 1]) << 8 | u32::from(data[i + 2]);
    (v.wrapping_mul(0x9E37_79B1) >> (32 - HASH_BITS)) as usize
}

/// Compress `data` into a raw DEFLATE stream (one final fixed-Huffman block).
pub fn deflate_fixed(data: &[u8]) -> Vec<u8> {
    deflate_into(Vec::with_capacity(data.len() / 4 + 16), data)
}

fn deflate_into(out: Vec<u8>, data: &[u8]) -> Vec<u8> {
    let mut w = BitWriter::new(out);
    w.put(1, 1); // BFINAL
    w.put(1, 2); // BTYPE = 01 (fixed Huffman)

    let mut head = vec![usize::MAX; 1 << HASH_BITS];
    let mut prev = vec![usize::MAX; data.len()];
    let mut i = 0;
    let insert = |head: &mut Vec<usize>, prev: &mut Vec<usize>, pos: usize| {
        if pos + MIN_MATCH <= data.len() {
            let h = hash3(data, pos);
            prev[pos] = head[h];
            head[h] = pos;
        }
    };

    while i < data.len() {
        let mut best_len = 0;
        let mut best_dist = 0;
        if i + MIN_MATCH <= data.len() {
            let max_len = MAX_MATCH.min(data.len() - i);
            let mut cand = head[hash3(data, i)];
            let mut chain = 0;
            while cand != usize::MAX && i - cand <= WINDOW && chain < MAX_CHAIN {
                let len = data[cand..]
                    .iter()
                    .zip(&data[i..i + max_len])
                    .take_while(|(a, b)| a == b)
                    .count();
                if len > best_len {
                    best_len = len;
                    best_dist = i - cand;
                    if len == max_len {
                        break;
                    }
                }
                cand = prev[cand];
                chain += 1;
            }
        }
        if best_len >= MIN_MATCH {
            emit_match(&mut w, best_len, best_dist);
            for p in i..i + best_len {
                insert(&mut head, &mut prev, p);
            }
            i += best_len;
        } else {
            emit_literal(&mut w, data[i]);
            insert(&mut head, &mut prev, i);
            i += 1;
        }
    }
    let (code, len) = fixed_litlen_code(256);
    w.put_code(code, len);
    w.finish()
}

/// Compress `data` into a zlib stream (2-byte header, DEFLATE body, Adler-32 trailer).
pub fn zlib_compress(data: &[u8]) -> Vec<u8> {
    // CMF = 0x78 (deflate, 32 KiB window); FLG = 0x5E (level 2) so that 0x785E % 31 == 0.
    let mut out = Vec::with_capacity(data.len() / 4 + 32);
    out.extend_from_slice(&[0x78, 0x5E]);
    let mut out = deflate_into(out, data);
    out.extend_from_slice(&adler32(data).to_be_bytes());
    out
}

#[cfg(test)]
pub(crate) mod inflate {
    //! Minimal inflater (stored + fixed-Huffman blocks) used to verify the encoder.

    use super::{DIST_BASE, LENGTH_BASE};

    struct BitReader<'a> {
        data: &'a [u8],
        pos: usize,
        bit: u32,
    }

    impl BitReader<'_> {
        fn bit(&mut self) -> u32 {
            let b = (u32::from(self.data[self.pos]) >> self.bit) & 1;
            self.bit += 1;
            if self.bit == 8 {
                self.bit = 0;
                self.pos += 1;
            }
            b
        }

        fn bits(&mut self, n: u32) -> u32 {
            (0..n).fold(0, |acc, i| acc | (self.bit() << i))
        }

        fn huff(&mut self, n: u32) -> u32 {
            (0..n).fold(0, |acc, _| (acc << 1) | self.bit())
        }

        fn litlen(&mut self) -> u32 {
            let mut code = self.huff(7);
            if code <= 0x17 {
                return code + 256;
            }
            code = (code << 1) | self.bit();
            if (0x30..=0xBF).contains(&code) {
                return code - 0x30;
            }
            if (0xC0..=0xC7).contains(&code) {
                return code - 0xC0 + 280;
            }
            code = (code << 1) | self.bit();
            code - 0x190 + 144
        }
    }

    /// Decode a zlib stream, checking the header and Adler-32 trailer.
    pub fn zlib_decompress(z: &[u8]) -> Vec<u8> {
        assert_eq!(z[0] & 0x0F, 8);
        assert_eq!((u16::from(z[0]) << 8 | u16::from(z[1])) % 31, 0);
        let mut r = BitReader {
            data: &z[2..],
            pos: 0,
            bit: 0,
        };
        let mut out: Vec<u8> = Vec::new();
        loop {
            let last = r.bits(1);
            match r.bits(2) {
                0 => {
                    if r.bit != 0 {
                        r.bit = 0;
                        r.pos += 1;
                    }
                    let len = u16::from_le_bytes([r.data[r.pos], r.data[r.pos + 1]]) as usize;
                    r.pos += 4;
                    out.extend_from_slice(&r.data[r.pos..r.pos + len]);
                    r.pos += len;
                }
                1 => loop {
                    let sym = r.litlen();
                    if sym < 256 {
                        out.push(sym as u8);
                    } else if sym == 256 {
                        break;
                    } else {
                        let (lb, le) = LENGTH_BASE[(sym - 257) as usize];
                        let len = usize::from(lb) + r.bits(u32::from(le)) as usize;
                        let (db, de) = DIST_BASE[r.huff(5) as usize];
                        let dist = usize::from(db) + r.bits(u32::from(de)) as usize;
                        let start = out.len() - dist;
                        for k in 0..len {
                            out.push(out[start + k]);
                        }
                    }
                },
                t => panic!("unsupported block type {t}"),
            }
            if last == 1 {
                break;
            }
        }
        let trailer = &z[z.len() - 4..];
        assert_eq!(
            u32::from_be_bytes(trailer.try_into().unwrap()),
            super::adler32(&out)
        );
        out
    }
}

#[cfg(test)]
mod tests {
    use super::inflate::zlib_decompress;
    use super::*;

    #[test]
    fn round_trip_various_inputs() {
        let mut noisy = Vec::new();
        let mut x = 12345u32;
        for _ in 0..70_000 {
            x = x.wrapping_mul(1_103_515_245).wrapping_add(12345);
            noisy.push((x >> 24) as u8);
        }
        let cases: Vec<Vec<u8>> = vec![
            Vec::new(),
            b"a".to_vec(),
            b"abcabcabcabcabcabcabcabc".to_vec(),
            vec![255u8; 100_000],
            (0..=255u8).cycle().take(80_000).collect(),
            noisy,
        ];
        for data in cases {
            let z = zlib_compress(&data);
            assert_eq!(zlib_decompress(&z), data);
        }
    }

    #[test]
    fn runs_compress_well() {
        let z = zlib_compress(&vec![7u8; 1_000_000]);
        assert!(z.len() < 20_000, "compressed size {}", z.len());
    }
}
