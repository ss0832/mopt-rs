//! What the report of `run_autots` shows where its script does not run: a drawing of the
//! first frame of every structure (SVG, written by mopt-rs) and downloads as links.
//!
//! The report draws its structures with a script of its own, so they can be turned,
//! played, and restyled. Where a page is shown without running scripts (some previews,
//! a mail program, printing), that would leave empty boxes and buttons that do nothing.
//! The drawing of this module stands in the box until the script replaces it, and a
//! download is a link that holds its file (`data:` URL), which needs no script.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::fmt::Write as _;

use crate::domain::{Element, Units};
use crate::parameters::covalent::covalent_radius_bohr;

/// The colours of the elements after the CPK convention (as in Jmol), the same as the
/// script of the report.
fn color(symbol: &str) -> &'static str {
    match symbol {
        "H" => "#e8e8e8",
        "C" => "#505050",
        "N" => "#3050f8",
        "O" => "#ff0d0d",
        "F" => "#90e050",
        "S" => "#ffd02a",
        "Cl" => "#1ff01f",
        "Br" => "#a62929",
        "I" => "#940094",
        "P" => "#ff8000",
        "Si" => "#f0c8a0",
        "B" => "#ffb5b5",
        "Cu" => "#c88033",
        "Ag" => "#c0c0c0",
        "Au" => "#ffd123",
        "Pd" => "#006985",
        "Pt" => "#d0d0e0",
        "Ni" => "#50d050",
        "Fe" => "#e06633",
        "Al" => "#bfa6a6",
        _ => "#ff1493",
    }
}

/// The covalent radius of `symbol` in Angstrom (1 Angstrom for what is not an element).
fn radius(symbol: &str) -> f64 {
    Element::parse_token(symbol).map_or(1.0, |e| covalent_radius_bohr(e) * Units::BOHR_TO_ANGSTROM)
}

/// The atoms of the first frame of `xyz` (Angstrom).
fn first_frame(xyz: &str) -> Vec<(String, [f64; 3])> {
    let mut lines = xyz.lines().skip_while(|l| l.trim().is_empty());
    let Some(n) = lines.next().and_then(|l| l.trim().parse::<usize>().ok()) else {
        return Vec::new();
    };
    lines
        .skip(1)
        .take(n)
        .filter_map(|line| {
            let f: Vec<&str> = line.split_whitespace().collect();
            let at = |k: usize| f.get(k).and_then(|v| v.parse::<f64>().ok());
            Some((f.first()?.to_string(), [at(1)?, at(2)?, at(3)?]))
        })
        .collect()
}

/// The axes along which the atoms spread most, least: the eigenvectors of their scatter
/// matrix by Jacobi rotations (3 x 3), largest first.
fn principal_axes(points: &[[f64; 3]], centre: [f64; 3]) -> [[f64; 3]; 3] {
    let mut a = [[0.0; 3]; 3];
    for p in points {
        let d = [p[0] - centre[0], p[1] - centre[1], p[2] - centre[2]];
        for i in 0..3 {
            for j in 0..3 {
                a[i][j] += d[i] * d[j];
            }
        }
    }
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..30 {
        for (p, q) in [(0, 1), (0, 2), (1, 2)] {
            if a[p][q].abs() < 1e-14 {
                continue;
            }
            let theta = 0.5 * (2.0 * a[p][q]).atan2(a[q][q] - a[p][p]);
            let (s, c) = theta.sin_cos();
            for row in &mut a {
                let (x, y) = (row[p], row[q]);
                row[p] = c * x - s * y;
                row[q] = s * x + c * y;
            }
            let (rp, rq) = (a[p], a[q]);
            a[p] = std::array::from_fn(|k| c * rp[k] - s * rq[k]);
            a[q] = std::array::from_fn(|k| s * rp[k] + c * rq[k]);
            for row in &mut v {
                let (x, y) = (row[p], row[q]);
                row[p] = c * x - s * y;
                row[q] = s * x + c * y;
            }
        }
    }
    let mut order = [0, 1, 2];
    order.sort_by(|&i, &j| a[j][j].total_cmp(&a[i][i]));
    order.map(|k| [v[0][k], v[1][k], v[2][k]])
}

/// A drawing of the first frame of `xyz`: the atoms seen along the axis of least spread
/// (balls in the colours of the elements, numbered from 1), the bonds by covalent radii.
/// Empty for a frame without atoms.
pub fn drawing(xyz: &str) -> String {
    let atoms = first_frame(xyz);
    if atoms.is_empty() {
        return String::new();
    }
    let n = atoms.len() as f64;
    let centre = [0, 1, 2].map(|k| atoms.iter().map(|(_, p)| p[k]).sum::<f64>() / n);
    let positions: Vec<[f64; 3]> = atoms.iter().map(|(_, p)| *p).collect();
    let [ex, ey, ez] = principal_axes(&positions, centre);
    let project = |p: &[f64; 3]| {
        let d = [p[0] - centre[0], p[1] - centre[1], p[2] - centre[2]];
        let dot = |e: &[f64; 3]| d[0] * e[0] + d[1] * e[1] + d[2] * e[2];
        (dot(&ex), dot(&ey), dot(&ez))
    };
    let flat: Vec<(f64, f64, f64)> = positions.iter().map(project).collect();
    let extent = flat
        .iter()
        .map(|(x, y, _)| x.abs().max(y.abs() * 640.0 / 400.0))
        .fold(1.0_f64, f64::max)
        + 0.8;
    let (width, height) = (640.0, 400.0);
    let scale = 0.47 * width / extent;
    let at = |k: usize| {
        (
            width / 2.0 + scale * flat[k].0,
            height / 2.0 - scale * flat[k].1,
        )
    };
    let mut svg = format!(
        "<svg class=\"static-view\" xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {width} \
         {height}\" role=\"img\" aria-label=\"The first frame ({} atoms)\">",
        atoms.len()
    );
    // Bonds first, then the atoms from the back to the front.
    let close = |i: usize, j: usize| {
        let d: f64 = (0..3)
            .map(|k| (positions[i][k] - positions[j][k]).powi(2))
            .sum::<f64>()
            .sqrt();
        d < 1.2 * (radius(&atoms[i].0) + radius(&atoms[j].0))
    };
    for i in 0..atoms.len() {
        for j in 0..i {
            if close(i, j) {
                let ((x1, y1), (x2, y2)) = (at(i), at(j));
                let _ = write!(
                    svg,
                    "<line x1=\"{x1:.1}\" y1=\"{y1:.1}\" x2=\"{x2:.1}\" y2=\"{y2:.1}\" \
                     stroke=\"#8a8f99\" stroke-width=\"{:.1}\" stroke-linecap=\"round\"/>",
                    (0.16 * scale).max(1.5)
                );
            }
        }
    }
    let mut order: Vec<usize> = (0..atoms.len()).collect();
    order.sort_by(|&a, &b| flat[a].2.total_cmp(&flat[b].2));
    let labels = atoms.len() <= 200;
    for k in order {
        let (x, y) = at(k);
        let r = (0.32 * radius(&atoms[k].0) * scale).max(2.5);
        let _ = write!(
            svg,
            "<circle cx=\"{x:.1}\" cy=\"{y:.1}\" r=\"{r:.1}\" fill=\"{}\" stroke=\"rgba(0,0,0,0.35)\" \
             stroke-width=\"1\"/>",
            color(&atoms[k].0)
        );
        if labels {
            let _ = write!(
                svg,
                "<text x=\"{x:.1}\" y=\"{:.1}\" text-anchor=\"middle\" font-size=\"11\" \
                 font-weight=\"600\" fill=\"#111111\" stroke=\"rgba(255,255,255,0.85)\" \
                 stroke-width=\"3\" paint-order=\"stroke\">{}</text>",
                y + 4.0,
                k + 1
            );
        }
    }
    svg.push_str("</svg>");
    svg
}

/// The bytes of `text` in base64 (RFC 4648, with padding).
fn base64(text: &str) -> String {
    base64_bytes(text.as_bytes())
}

/// `bytes` in base64 (RFC 4648, with padding).
pub fn base64_bytes(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b = [
            chunk[0],
            chunk.get(1).copied().unwrap_or(0),
            chunk.get(2).copied().unwrap_or(0),
        ];
        let n = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
        for k in 0..4 {
            if k <= chunk.len() {
                out.push(char::from(ALPHABET[((n >> (18 - 6 * k)) & 63) as usize]));
            } else {
                out.push('=');
            }
        }
    }
    out
}

/// A link that saves `text` as the file `name` (the file is in the link).
pub fn download_link(name: &str, text: &str, label: &str, class: &str) -> String {
    data_download(name, text, label, class, "chemical/x-xyz")
}

/// A self-contained download in the native format of the result.
pub(super) fn data_download(
    name: &str,
    text: &str,
    label: &str,
    class: &str,
    mime: &str,
) -> String {
    let safe: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || "._-".contains(c) {
                c
            } else {
                '_'
            }
        })
        .collect();
    format!(
        "<a class=\"{class}\" download=\"{safe}\" href=\"data:{mime};base64,{}\">{label}</a>",
        base64(text)
    )
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    const WATER: &str = "3\nwater\nO 0.0 0.0 0.0\nH 0.96 0.0 0.0\nH -0.24 0.93 0.0\n";

    #[test]
    fn a_drawing_of_the_first_frame() {
        let svg = drawing(&format!("{WATER}{WATER}"));
        assert!(svg.starts_with("<svg class=\"static-view\""));
        assert_eq!(svg.matches("<circle").count(), 3, "{svg}");
        assert_eq!(svg.matches("<line").count(), 2, "{svg}");
        assert!(svg.contains("#ff0d0d") && svg.contains(">3</text>"));
        assert!(drawing("").is_empty() && drawing("x\n").is_empty());
    }

    #[test]
    fn base64_as_the_standard_has_it() {
        assert_eq!(base64(""), "");
        assert_eq!(base64("f"), "Zg==");
        assert_eq!(base64("fo"), "Zm8=");
        assert_eq!(base64("foo"), "Zm9v");
        assert_eq!(base64("foobar"), "Zm9vYmFy");
        let link = download_link("a b.xyz", "foo", "Download", "dl");
        assert_eq!(
            link,
            "<a class=\"dl\" download=\"a_b.xyz\" href=\"data:chemical/x-xyz;base64,Zm9v\">Download</a>"
        );
    }
}
