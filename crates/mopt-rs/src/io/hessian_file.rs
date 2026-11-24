//! Hessians in Cartesian coordinates read from files of several kinds.
//!
//! [`read_hessian_file`] tells the kind from the content:
//!
//! - a NumPy file (`numpy.save`) of a square matrix;
//! - an ORCA `.hess` file (the block `$hessian`);
//! - a Gaussian formatted checkpoint file (`Cartesian Force Constants`, the lower
//!   triangle);
//! - a Gaussian output (`Force constants in Cartesian coordinates`, printed by a
//!   frequency job with `#P`);
//! - numbers only, separated by blanks, commas, or line ends (lines that begin with `#`
//!   are skipped): a full square matrix row by row, or its lower triangle row by row.
//!   A count of numbers that fits both is read as the square.
//!
//! All are taken as Hartree/Bohr^2 with the atoms in the order of the structure, three
//! coordinates per atom, which is what ORCA and Gaussian write; a NumPy or text file has
//! to be in these units.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::path::Path;

use faer::Mat;

use super::npy::read_npy_matrix;
use crate::error::{MoptError, MoptResult};

/// The first bytes of a NumPy file.
const NPY_MAGIC: &[u8] = b"\x93NUMPY";

/// A number as programs write it (Fortran's `D` exponents included).
fn number(token: &str) -> Option<f64> {
    token
        .replace(['D', 'd'], "E")
        .parse::<f64>()
        .ok()
        .filter(|value| value.is_finite())
}

/// `n` for `count` elements of a lower triangle with its diagonal.
fn triangle_order(count: usize) -> Option<usize> {
    let n = (((8 * count + 1) as f64).sqrt() as usize).saturating_sub(1) / 2;
    (n > 0 && n * (n + 1) / 2 == count).then_some(n)
}

/// The symmetric matrix of the lower triangle `values`, row by row.
fn from_triangle(values: &[f64], n: usize) -> Mat<f64> {
    let mut h = Mat::<f64>::zeros(n, n);
    let mut k = 0;
    for r in 0..n {
        for c in 0..=r {
            h[(r, c)] = values[k];
            h[(c, r)] = values[k];
            k += 1;
        }
    }
    h
}

/// A matrix printed in blocks of columns from line `start` on: a line of column numbers,
/// then lines of a row number and the values of that row in those columns (all of them,
/// or those of the lower triangle). `base` is the first number (0: ORCA, 1: Gaussian).
fn from_blocks(lines: &[&str], start: usize, base: usize, what: &str) -> MoptResult<Mat<f64>> {
    let mut entries: Vec<(usize, usize, f64)> = Vec::new();
    let mut columns: Vec<usize> = Vec::new();
    let mut order = 0;
    for line in lines.iter().skip(start) {
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.is_empty() {
            continue;
        }
        let whole: Option<Vec<usize>> = tokens.iter().map(|t| t.parse::<usize>().ok()).collect();
        if let Some(numbers) = whole {
            // A line of column numbers (a row of whole numbers has a value beside them).
            if numbers.iter().any(|c| *c < base) {
                break;
            }
            columns = numbers.iter().map(|c| c - base).collect();
            continue;
        }
        let row = tokens[0].parse::<usize>().ok().filter(|r| *r >= base);
        let values: Option<Vec<f64>> = tokens[1..].iter().map(|t| number(t)).collect();
        let (Some(row), Some(values)) = (row, values) else {
            break;
        };
        if columns.is_empty() || values.is_empty() || values.len() > columns.len() {
            break;
        }
        let row = row - base;
        for (column, value) in columns.iter().zip(&values) {
            entries.push((row, *column, *value));
            order = order.max(row + 1).max(column + 1);
        }
    }
    if order == 0 {
        return Err(MoptError::parse(format!("{what}: no rows of numbers")));
    }
    let mut h = Mat::<f64>::from_fn(order, order, |_, _| f64::NAN);
    for (r, c, value) in &entries {
        h[(*r, *c)] = *value;
    }
    for r in 0..order {
        for c in 0..order {
            if h[(r, c)].is_nan() {
                h[(r, c)] = h[(c, r)];
            }
            if h[(r, c)].is_nan() {
                return Err(MoptError::parse(format!(
                    "{what}: element ({}, {}) is missing",
                    r + 1,
                    c + 1
                )));
            }
        }
    }
    Ok(h)
}

/// The Hessian in `text` (every kind of the module but the NumPy file).
pub fn parse_hessian_text(text: &str) -> MoptResult<Mat<f64>> {
    let lines: Vec<&str> = text.lines().collect();
    let last = |found: &dyn Fn(&str) -> bool| lines.iter().rposition(|line| found(line));
    if let Some(at) = last(&|line| line.trim().eq_ignore_ascii_case("$hessian")) {
        let announced = lines
            .get(at + 1)
            .and_then(|line| line.split_whitespace().next())
            .and_then(|token| token.parse::<usize>().ok())
            .ok_or_else(|| MoptError::parse("the $hessian block has no dimension"))?;
        let h = from_blocks(&lines, at + 2, 0, "the $hessian block")?;
        if h.nrows() != announced {
            return Err(MoptError::parse(format!(
                "the $hessian block has {} rows, {announced} were announced",
                h.nrows()
            )));
        }
        return Ok(h);
    }
    if let Some(at) = last(&|line| line.starts_with("Cartesian Force Constants")) {
        let count = lines[at]
            .rsplit("N=")
            .next()
            .and_then(|rest| rest.trim().parse::<usize>().ok())
            .ok_or_else(|| MoptError::parse("'Cartesian Force Constants' without N="))?;
        let values: Vec<f64> = lines[at + 1..]
            .iter()
            .flat_map(|line| line.split_whitespace())
            .map_while(number)
            .take(count)
            .collect();
        let n = triangle_order(count).filter(|_| values.len() == count);
        return n.map(|n| from_triangle(&values, n)).ok_or_else(|| {
            MoptError::parse(format!(
                "'Cartesian Force Constants': {} of {count} numbers, which is no lower triangle",
                values.len()
            ))
        });
    }
    if let Some(at) = last(&|line| line.contains("Force constants in Cartesian coordinates")) {
        return from_blocks(&lines, at + 1, 1, "the Gaussian force constants");
    }
    let mut values = Vec::new();
    for line in lines
        .iter()
        .filter(|line| !line.trim_start().starts_with('#'))
    {
        for token in line
            .split(|c: char| c.is_whitespace() || c == ',' || c == ';')
            .filter(|token| !token.is_empty())
        {
            values.push(number(token).ok_or_else(|| {
                MoptError::parse(format!(
                    "'{token}' is not a number (a Hessian as numbers only, an ORCA .hess \
                     file, a Gaussian .fchk file or output, or a NumPy file is read)"
                ))
            })?);
        }
    }
    let count = values.len();
    let square = (count as f64).sqrt().round() as usize;
    if square > 0 && square * square == count {
        return Ok(Mat::from_fn(square, square, |r, c| values[r * square + c]));
    }
    triangle_order(count)
        .map(|n| from_triangle(&values, n))
        .ok_or_else(|| {
            MoptError::parse(format!(
                "{count} numbers are neither a square matrix nor a lower triangle"
            ))
        })
}

/// The Hessian of the file `path` (Hartree/Bohr^2; the kinds of the module documentation).
pub fn read_hessian_file(path: impl AsRef<Path>) -> MoptResult<Mat<f64>> {
    let path = path.as_ref();
    let bytes =
        std::fs::read(path).map_err(|e| MoptError::Io(format!("{}: {e}", path.display())))?;
    if bytes.starts_with(NPY_MAGIC) {
        return read_npy_matrix(path);
    }
    let text = String::from_utf8_lossy(&bytes);
    parse_hessian_text(&text).map_err(|e| e.context(path.display()))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::io::npy::write_npy_matrix;

    /// A symmetric matrix of order 7 (more columns than one block of five or six).
    fn sample() -> Mat<f64> {
        Mat::from_fn(7, 7, |r, c| {
            let (lo, hi) = (r.min(c) as f64, r.max(c) as f64);
            0.5 / (1.0 + hi - lo) - 0.01 * lo * hi
        })
    }

    fn same(a: &Mat<f64>, b: &Mat<f64>, tolerance: f64) {
        assert_eq!((a.nrows(), a.ncols()), (b.nrows(), b.ncols()));
        for r in 0..a.nrows() {
            for c in 0..a.ncols() {
                assert!((a[(r, c)] - b[(r, c)]).abs() < tolerance, "({r}, {c})");
            }
        }
    }

    #[test]
    fn every_kind_of_file_gives_the_same_matrix() {
        let h = sample();
        let n = h.nrows();
        let dir = std::env::temp_dir().join(format!("mopt_hessian_file_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        // NumPy.
        let npy = dir.join("h.npy");
        write_npy_matrix(&npy, &h).unwrap();
        same(&read_hessian_file(&npy).unwrap(), &h, 1e-15);

        // Numbers only: the square with a comment and commas, and the lower triangle.
        let mut square = String::from("# Hartree/Bohr^2\n");
        let mut triangle = String::new();
        for r in 0..n {
            for c in 0..n {
                square.push_str(&format!("{:.12e}, ", h[(r, c)]));
                if c <= r {
                    triangle.push_str(&format!("{:.12e} ", h[(r, c)]));
                }
            }
            square.push('\n');
            triangle.push('\n');
        }
        let file = dir.join("h.txt");
        std::fs::write(&file, &square).unwrap();
        same(&read_hessian_file(&file).unwrap(), &h, 1e-12);
        same(&parse_hessian_text(&triangle).unwrap(), &h, 1e-12);

        // ORCA: full rows in blocks of five columns, numbers from zero.
        let mut orca = String::from("$orca_hessian_file\n\n$hessian\n7\n");
        for block in (0..n).step_by(5) {
            let columns: Vec<usize> = (block..(block + 5).min(n)).collect();
            let header: Vec<String> = columns.iter().map(|c| format!("{c:>19}")).collect();
            orca.push_str(&format!("{}\n", header.join("")));
            for r in 0..n {
                let row: Vec<String> = columns
                    .iter()
                    .map(|c| format!("{:>19.10E}", h[(r, *c)]))
                    .collect();
                orca.push_str(&format!("{r:>5}{}\n", row.join("")));
            }
        }
        orca.push_str("\n$vibrational_frequencies\n7\n    0        0.000000\n");
        same(&parse_hessian_text(&orca).unwrap(), &h, 1e-10);

        // Gaussian output: the lower triangle in blocks of five columns, numbers from
        // one, D exponents.
        let mut gaussian = String::from(" Force constants in Cartesian coordinates: \n");
        for block in (0..n).step_by(5) {
            let columns: Vec<usize> = (block..(block + 5).min(n)).collect();
            let header: Vec<String> = columns.iter().map(|c| format!("{:>14}", c + 1)).collect();
            gaussian.push_str(&format!("      {}\n", header.join("")));
            for r in block..n {
                let row: Vec<String> = columns
                    .iter()
                    .filter(|c| **c <= r)
                    .map(|c| format!("{:>14.6E}", h[(r, *c)]).replace('E', "D"))
                    .collect();
                gaussian.push_str(&format!("{:>7}{}\n", r + 1, row.join("")));
            }
        }
        gaussian.push_str(" Final forces over variables, Energy=-1.0D+00:\n");
        same(&parse_hessian_text(&gaussian).unwrap(), &h, 1e-6);

        // Gaussian formatted checkpoint file: the lower triangle, five numbers per line.
        let mut values = Vec::new();
        for r in 0..n {
            for c in 0..=r {
                values.push(h[(r, c)]);
            }
        }
        let mut fchk = format!(
            "Number of atoms                            I                3\n\
             Cartesian Force Constants                  R   N={:>12}\n",
            values.len()
        );
        for line in values.chunks(5) {
            let row: Vec<String> = line.iter().map(|v| format!("{v:>16.8E}")).collect();
            fchk.push_str(&format!("{}\n", row.join("")));
        }
        fchk.push_str("Dipole Moment                              R   N=           3\n");
        same(&parse_hessian_text(&fchk).unwrap(), &h, 1e-8);

        // What is no Hessian.
        assert!(parse_hessian_text("1.0 2.0 3.0 4.0 5.0").is_err());
        assert!(parse_hessian_text("1.0 two 3.0 4.0").is_err());
        assert!(parse_hessian_text("").is_err());
        assert!(read_hessian_file(dir.join("none.npy")).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
