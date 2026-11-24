//! Matrices of real numbers in NumPy's `.npy` format: what MultiOptPy keeps its Hessians
//! in between iterations (`tmp_hessian_<i>.npy`), so that such files can be exchanged with
//! MultiOptPy and read with `numpy.load`.
//!
//! Written: version 1.0, little-endian 64-bit floats (`<f8`), C order, two dimensions.
//! Read: versions 1 to 3, `<f8` (or `=f8` / `|f8` on a little-endian machine's files), C or
//! Fortran order, two dimensions.
//!
//! MultiOptPy: `numpy.save` / `numpy.load` in `Optimizer/rfo_neb.py` and others.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::fs;
use std::path::Path;

use faer::Mat;

use crate::error::{MoptError, MoptResult};

/// The first bytes of every `.npy` file.
const MAGIC: &[u8; 6] = b"\x93NUMPY";
/// The header and what precedes it are padded to a multiple of this.
const ALIGNMENT: usize = 64;

/// `matrix` as the bytes of a `.npy` file.
pub fn npy_bytes(matrix: &Mat<f64>) -> Vec<u8> {
    let mut header = format!(
        "{{'descr': '<f8', 'fortran_order': False, 'shape': ({}, {}), }}",
        matrix.nrows(),
        matrix.ncols()
    );
    // Magic (6), version (2), length of the header (2), the header, and a line feed.
    let unpadded = MAGIC.len() + 4 + header.len() + 1;
    header.push_str(&" ".repeat((ALIGNMENT - unpadded % ALIGNMENT) % ALIGNMENT));
    header.push('\n');
    let mut bytes = Vec::with_capacity(unpadded + 8 * matrix.nrows() * matrix.ncols() + ALIGNMENT);
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&[1, 0]);
    bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    for r in 0..matrix.nrows() {
        for c in 0..matrix.ncols() {
            bytes.extend_from_slice(&matrix[(r, c)].to_le_bytes());
        }
    }
    bytes
}

/// The value of `key` in the dictionary of a header, up to the next comma outside
/// parentheses.
fn header_value<'a>(header: &'a str, key: &str) -> Option<&'a str> {
    let start = header.find(&format!("'{key}'"))? + key.len() + 2;
    let rest = header[start..].trim_start().strip_prefix(':')?.trim_start();
    let mut depth = 0_i32;
    let end = rest
        .char_indices()
        .find(|(_, c)| {
            match c {
                '(' => depth += 1,
                ')' => depth -= 1,
                _ => {}
            }
            (*c == ',' && depth == 0) || *c == '}'
        })
        .map_or(rest.len(), |(k, _)| k);
    Some(rest[..end].trim())
}

/// The matrix in the bytes of a `.npy` file.
pub fn matrix_from_npy(bytes: &[u8]) -> MoptResult<Mat<f64>> {
    let bad = |what: &str| MoptError::parse(format!("not a .npy file of a real matrix: {what}"));
    if bytes.len() < 10 || &bytes[..6] != MAGIC {
        return Err(bad("the first bytes are not those of a .npy file"));
    }
    let (length, start): (usize, usize) = match bytes[6] {
        1 => (usize::from(u16::from_le_bytes([bytes[8], bytes[9]])), 10),
        2 | 3 if bytes.len() >= 12 => (
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            12,
        ),
        version => return Err(bad(&format!("format version {version}"))),
    };
    let data = start
        .checked_add(length)
        .filter(|end| *end <= bytes.len())
        .ok_or_else(|| bad("the header is longer than the file"))?;
    let header =
        std::str::from_utf8(&bytes[start..data]).map_err(|_| bad("the header is not text"))?;
    let kind = header_value(header, "descr")
        .ok_or_else(|| bad("no 'descr' in the header"))?
        .trim_matches(|c| c == '\'' || c == '"');
    if !matches!(kind, "<f8" | "=f8" | "|f8") {
        return Err(bad(&format!("numbers of the type {kind} (expected <f8)")));
    }
    let fortran = header_value(header, "fortran_order") == Some("True");
    let shape: Vec<usize> = header_value(header, "shape")
        .ok_or_else(|| bad("no 'shape' in the header"))?
        .trim_matches(|c| c == '(' || c == ')')
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| part.parse::<usize>())
        .collect::<Result<_, _>>()
        .map_err(|_| bad("a shape that is not made of whole numbers"))?;
    let [rows, columns] = shape[..] else {
        return Err(bad(&format!("{} dimensions (expected 2)", shape.len())));
    };
    let count = rows
        .checked_mul(columns)
        .filter(|count| count.checked_mul(8) == Some(bytes.len() - data))
        .ok_or_else(|| bad("the size of the data does not match the shape"))?;
    let values: Vec<f64> = bytes[data..data + 8 * count]
        .chunks_exact(8)
        .map(|chunk| {
            let mut number = [0_u8; 8];
            number.copy_from_slice(chunk);
            f64::from_le_bytes(number)
        })
        .collect();
    Ok(Mat::from_fn(rows, columns, |r, c| {
        if fortran {
            values[c * rows + r]
        } else {
            values[r * columns + c]
        }
    }))
}

/// Write `matrix` to the `.npy` file `path`.
pub fn write_npy_matrix(path: impl AsRef<Path>, matrix: &Mat<f64>) -> MoptResult<()> {
    let path = path.as_ref();
    fs::write(path, npy_bytes(matrix))
        .map_err(|e| MoptError::Io(format!("{}: {e}", path.display())))
}

/// The matrix in the `.npy` file `path`.
pub fn read_npy_matrix(path: impl AsRef<Path>) -> MoptResult<Mat<f64>> {
    let path = path.as_ref();
    let bytes = fs::read(path).map_err(|e| MoptError::Io(format!("{}: {e}", path.display())))?;
    matrix_from_npy(&bytes).map_err(|e| e.context(path.display()))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn a_matrix_comes_back_as_it_was_written() {
        let matrix = Mat::from_fn(3, 2, |r, c| (r as f64 + 1.0) / (c as f64 + 3.0) - 0.5);
        let bytes = npy_bytes(&matrix);
        // What numpy.save writes: the magic string, version 1.0, a header that ends with
        // a line feed on a multiple of 64 bytes, then the numbers row by row.
        assert_eq!(&bytes[..8], b"\x93NUMPY\x01\x00");
        let length = usize::from(u16::from_le_bytes([bytes[8], bytes[9]]));
        assert_eq!((10 + length) % 64, 0);
        assert_eq!(bytes[10 + length - 1], b'\n');
        let header = std::str::from_utf8(&bytes[10..10 + length]).unwrap();
        assert!(header.starts_with("{'descr': '<f8', 'fortran_order': False, 'shape': (3, 2), }"));
        assert_eq!(bytes.len(), 10 + length + 8 * 6);
        assert_eq!(
            &bytes[10 + length + 8..10 + length + 16],
            &matrix[(0, 1)].to_le_bytes()
        );
        assert_eq!(matrix_from_npy(&bytes).unwrap(), matrix);

        let dir = std::env::temp_dir().join(format!("mopt_npy_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("matrix.npy");
        write_npy_matrix(&path, &matrix).unwrap();
        assert_eq!(read_npy_matrix(&path).unwrap(), matrix);
        assert!(read_npy_matrix(dir.join("missing.npy")).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn other_files_numpy_writes_are_read_or_refused() {
        // Fortran order in a version 2.0 file: the numbers column by column.
        let header = "{'descr': '<f8', 'fortran_order': True, 'shape': (2, 3), }\n";
        let mut bytes = b"\x93NUMPY\x02\x00".to_vec();
        bytes.extend_from_slice(&(header.len() as u32).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        for value in [1.0_f64, 4.0, 2.0, 5.0, 3.0, 6.0] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let matrix = matrix_from_npy(&bytes).unwrap();
        assert_eq!((matrix.nrows(), matrix.ncols()), (2, 3));
        assert_eq!((matrix[(0, 2)], matrix[(1, 0)]), (3.0, 4.0));

        // Not a .npy file, other numbers, other dimensions, a truncated file.
        assert!(matrix_from_npy(b"plain text").is_err());
        let with = |header: &str, numbers: usize| {
            let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
            bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
            bytes.extend_from_slice(header.as_bytes());
            bytes.extend(std::iter::repeat_n(0_u8, 8 * numbers));
            bytes
        };
        let integers = "{'descr': '<i8', 'fortran_order': False, 'shape': (1, 1), }\n";
        assert!(matrix_from_npy(&with(integers, 1)).is_err());
        let vector = "{'descr': '<f8', 'fortran_order': False, 'shape': (4,), }\n";
        assert!(matrix_from_npy(&with(vector, 4)).is_err());
        let square = "{'descr': '<f8', 'fortran_order': False, 'shape': (2, 2), }\n";
        assert!(matrix_from_npy(&with(square, 3)).is_err());
        assert!(matrix_from_npy(&with(square, 4)).is_ok());
        assert!(matrix_from_npy(&with("{'shape': (2, 2), }\n", 4)).is_err());
        assert!(matrix_from_npy(b"\x93NUMPY\x09\x00\x00\x00").is_err());
    }
}
