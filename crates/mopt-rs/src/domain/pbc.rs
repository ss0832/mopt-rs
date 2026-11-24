//! Periodic Boundary Conditions (`PeriodicCell`) supporting orthorhombic and general triclinic
//! lattices, selective 1D/2D/3D periodicity, Minimum Image Convention (MIC), cutoff lattice-image
//! enumeration, and Extended XYZ `Lattice="..."` headers.
//!
//! MultiOptPy: `Utils/pbc.py`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use crate::domain::quantity::{Angstrom, Bohr, Degree, LengthDim, Quantity, Radian, Unit, Vector3};
use crate::domain::units::Units;
use crate::error::{MoptError, MoptResult};

/// General 1D/2D/3D periodic simulation cell stored internally in Bohr.
///
/// Columns of `h_bohr` represent lattice vectors `a`, `b`, `c`:
/// `r_cart = h_bohr * s_frac`, where `h_bohr[row][col]` has `row = x,y,z` and `col = a,b,c`.
#[derive(Debug, Clone, PartialEq)]
pub struct PeriodicCell {
    /// Column-major lattice matrix `H = [a, b, c]` in Bohr (`h_bohr[xyz][vec_idx]`).
    pub h_bohr: [[f64; 3]; 3],
    /// Exact inverse matrix `H^{-1}` in `Bohr^{-1}`.
    pub h_inv_bohr: [[f64; 3]; 3],
    /// Per-lattice-vector periodicity flags (`[p_a, p_b, p_c]`).
    pub periodic_axes: [bool; 3],
}

impl PeriodicCell {
    /// Construct from three lattice vectors `a`, `b`, `c` in any length unit.
    pub fn from_lattice_vectors<U: Unit<Dim = LengthDim>>(
        a: Vector3<U>,
        b: Vector3<U>,
        c: Vector3<U>,
        periodic_axes: [bool; 3],
    ) -> MoptResult<Self> {
        let [a, b, c] = [a, b, c].map(|v| v.to::<Bohr>().value());
        let h_bohr = [[a[0], b[0], c[0]], [a[1], b[1], c[1]], [a[2], b[2], c[2]]];
        let h_inv_bohr = invert_3x3(&h_bohr)?;
        Ok(Self {
            h_bohr,
            h_inv_bohr,
            periodic_axes,
        })
    }

    /// 3D orthorhombic box with side lengths `(lx, ly, lz)`.
    pub fn orthorhombic<U: Unit<Dim = LengthDim>>(
        lx: Quantity<U>,
        ly: Quantity<U>,
        lz: Quantity<U>,
    ) -> MoptResult<Self> {
        if lx.value() <= 0.0 || ly.value() <= 0.0 || lz.value() <= 0.0 {
            return Err(MoptError::config(format!(
                "orthorhombic box lengths must be positive, got ({lx}, {ly}, {lz})"
            )));
        }
        let z = 0.0;
        Self::from_lattice_vectors(
            Vector3::<U>::new([lx.value(), z, z]),
            Vector3::new([z, ly.value(), z]),
            Vector3::new([z, z, lz.value()]),
            [true, true, true],
        )
    }

    /// Triclinic cell from crystallographic parameters `(a, b, c)` and `(alpha, beta, gamma)`.
    pub fn from_crystallographic<U: Unit<Dim = LengthDim>>(
        lengths: [Quantity<U>; 3],
        angles: [Quantity<Degree>; 3],
        periodic_axes: [bool; 3],
    ) -> MoptResult<Self> {
        let [a, b, c] = lengths.map(Quantity::value);
        if a <= 0.0 || b <= 0.0 || c <= 0.0 {
            return Err(MoptError::config("cell lengths (a, b, c) must be positive"));
        }
        let [alpha, beta, gamma] = angles.map(|q| q.to::<Radian>().value());
        let (cos_a, cos_b, cos_g, sin_g) = (
            libm::cos(alpha),
            libm::cos(beta),
            libm::cos(gamma),
            libm::sin(gamma),
        );
        if sin_g.abs() < 1e-12 {
            return Err(MoptError::config("degenerate cell angle gamma"));
        }
        let cx = c * cos_b;
        let cy = c * (cos_a - cos_b * cos_g) / sin_g;
        let cz_sq = c * c - cx * cx - cy * cy;
        if cz_sq <= 1e-14 {
            return Err(MoptError::config(
                "invalid crystallographic cell angles producing non-positive volume",
            ));
        }
        Self::from_lattice_vectors(
            Vector3::<U>::new([a, 0.0, 0.0]),
            Vector3::new([b * cos_g, b * sin_g, 0.0]),
            Vector3::new([cx, cy, libm::sqrt(cz_sq)]),
            periodic_axes,
        )
    }

    /// Parse CLI `-pbc` specification:
    /// - `"Lx,Ly,Lz"` -> 3D orthorhombic in Angstroms
    /// - `"a,b,c,alpha,beta,gamma"` -> 3D crystallographic in Angstroms & degrees
    /// - `"ax,ay,az,bx,by,bz,cx,cy,cz"` -> 9-component lattice vectors in Angstroms
    ///
    /// Optional suffix `:TTF` / `:T,T,F` controls axis periodicity (e.g. `"12.0,12.0,25.0:TTF"` for 2D slab).
    pub fn parse_cli_spec(spec: &str) -> MoptResult<Self> {
        let trimmed = spec.trim();
        let (nums_part, axes_part) = match trimmed.split_once(':') {
            Some((n, p)) => (n.trim(), Some(p.trim())),
            None => (trimmed, None),
        };
        let periodic_axes = match axes_part {
            Some(p) => parse_periodic_flags(p)?,
            None => [true, true, true],
        };

        let vals: Vec<f64> = nums_part
            .split([',', ' '])
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|s| {
                s.parse::<f64>()
                    .map_err(|e| MoptError::parse(format!("Invalid PBC float '{s}': {e}")))
            })
            .collect::<MoptResult<Vec<f64>>>()?;

        match vals.len() {
            3 => {
                let mut cell = Self::orthorhombic(
                    Quantity::<Angstrom>::new(vals[0]),
                    Quantity::new(vals[1]),
                    Quantity::new(vals[2]),
                )?;
                cell.periodic_axes = periodic_axes;
                Ok(cell)
            }
            6 => Self::from_crystallographic(
                [vals[0], vals[1], vals[2]].map(Quantity::<Angstrom>::new),
                [vals[3], vals[4], vals[5]].map(Quantity::<Degree>::new),
                periodic_axes,
            ),
            9 => Self::from_lattice_vectors(
                Vector3::<Angstrom>::new([vals[0], vals[1], vals[2]]),
                Vector3::new([vals[3], vals[4], vals[5]]),
                Vector3::new([vals[6], vals[7], vals[8]]),
                periodic_axes,
            ),
            n => Err(MoptError::parse(format!(
                "Expected 3, 6, or 9 numeric values in PBC specification, got {n}"
            ))),
        }
    }

    /// Parse Extended XYZ comment header if `Lattice="ax ay az bx by bz cx cy cz"` is present.
    pub fn parse_extxyz_comment(comment: &str) -> Option<Self> {
        let lat_idx = comment.find("Lattice=\"")?;
        let after = &comment[lat_idx + "Lattice=\"".len()..];
        let end_q = after.find('"')?;
        let lat_str = &after[..end_q];
        let vals: Vec<f64> = lat_str
            .split_whitespace()
            .filter_map(|s| s.parse::<f64>().ok())
            .collect();
        if vals.len() != 9 {
            return None;
        }
        let mut pbc = [true, true, true];
        if let Some(p_idx) = comment.find("pbc=\"") {
            let p_after = &comment[p_idx + "pbc=\"".len()..];
            if let Some(p_end) = p_after.find('"') {
                if let Ok(flags) = parse_periodic_flags(&p_after[..p_end]) {
                    pbc = flags;
                }
            }
        }
        Self::from_lattice_vectors(
            Vector3::<Angstrom>::new([vals[0], vals[1], vals[2]]),
            Vector3::new([vals[3], vals[4], vals[5]]),
            Vector3::new([vals[6], vals[7], vals[8]]),
            pbc,
        )
        .ok()
    }

    /// Format Extended XYZ `Lattice="..." pbc="..."` metadata in Angstroms, with 15 decimals:
    /// a rounding of at most 5e-16 Angstrom per component, so that the values recomputed from
    /// the file stay within the 1e-10 every hard constraint is held to (a distance through a
    /// periodic image; also the volume of a cell of 10^4 Bohr^3, which 12 decimals would move by
    /// about 1e-9 Bohr^3).
    pub fn to_extxyz_header(&self) -> String {
        let s = Units::BOHR_TO_ANGSTROM;
        let a = self.lattice_vector(0).value();
        let b = self.lattice_vector(1).value();
        let c = self.lattice_vector(2).value();
        let f = |b_flag: bool| if b_flag { "T" } else { "F" };
        format!(
            "Lattice=\"{:.15} {:.15} {:.15} {:.15} {:.15} {:.15} {:.15} {:.15} {:.15}\" \
             pbc=\"{} {} {}\"",
            a[0] * s,
            a[1] * s,
            a[2] * s,
            b[0] * s,
            b[1] * s,
            b[2] * s,
            c[0] * s,
            c[1] * s,
            c[2] * s,
            f(self.periodic_axes[0]),
            f(self.periodic_axes[1]),
            f(self.periodic_axes[2]),
        )
    }

    /// Return `true` if at least one axis has periodic boundary conditions enabled.
    #[inline]
    pub fn is_any_periodic(&self) -> bool {
        self.periodic_axes[0] || self.periodic_axes[1] || self.periodic_axes[2]
    }

    /// Lattice vector `idx` (`0 = a`, `1 = b`, `2 = c`).
    #[inline]
    pub fn lattice_vector(&self, idx: usize) -> Vector3<Bohr> {
        Vector3::new([
            self.h_bohr[0][idx],
            self.h_bohr[1][idx],
            self.h_bohr[2][idx],
        ])
    }

    /// Signed cell determinant (`det(H)`) in `Bohr^3`.
    pub fn volume_bohr3(&self) -> f64 {
        det_3x3(&self.h_bohr).abs()
    }

    /// Convert Cartesian coordinates (Bohr) to fractional coordinates `s = H^{-1} r`.
    #[inline]
    pub fn cart_to_frac(&self, r_bohr: [f64; 3]) -> [f64; 3] {
        mat3_mul_vec(&self.h_inv_bohr, r_bohr)
    }

    /// Convert fractional coordinates `s` to Cartesian coordinates (Bohr) `r = H s`.
    #[inline]
    pub fn frac_to_cart(&self, s_frac: [f64; 3]) -> [f64; 3] {
        mat3_mul_vec(&self.h_bohr, s_frac)
    }

    /// Compute the constant lattice shift `L_shift` (in Bohr) such that
    /// `delta_mic = (r_i - r_j) + L_shift` satisfies the Minimum Image Convention.
    ///
    /// Because `L_shift` is piecewise constant (zero derivative away from cell half-boundaries),
    /// automatic differentiation (`Dual` / `HyperDual`) simply adds `L_shift` to `(r_i - r_j)`.
    #[inline]
    pub fn mic_lattice_shift_bohr(&self, r_i_bohr: [f64; 3], r_j_bohr: [f64; 3]) -> [f64; 3] {
        let raw = [
            r_i_bohr[0] - r_j_bohr[0],
            r_i_bohr[1] - r_j_bohr[1],
            r_i_bohr[2] - r_j_bohr[2],
        ];
        let s = self.cart_to_frac(raw);
        let n: [f64; 3] = std::array::from_fn(|k| {
            if self.periodic_axes[k] {
                -libm::round(s[k])
            } else {
                0.0
            }
        });
        let initial = self.frac_to_cart(n);
        let h = &self.h_bohr;
        if h[0][1] == 0.0
            && h[0][2] == 0.0
            && h[1][0] == 0.0
            && h[1][2] == 0.0
            && h[2][0] == 0.0
            && h[2][1] == 0.0
        {
            return initial;
        }
        // Use only the periodic subspace: a distant atom above a slab must not
        // enlarge the search radius by its distance perpendicular to the slab.
        let mut axes = [0; 3];
        let mut rank = 0;
        let mut q = [[0.0; 3]; 3];
        let mut upper = [[0.0; 3]; 3];
        let mut projected = [0.0; 3];
        let residual: [f64; 3] = std::array::from_fn(|i| raw[i] + initial[i]);
        // Use the same tie order for a pair and its reverse.
        let orientation = residual
            .iter()
            .find(|&&x| x != 0.0)
            .map_or(1.0, |x| x.signum());
        for axis in 0..3 {
            if !self.periodic_axes[axis] {
                continue;
            }
            axes[rank] = axis;
            let mut v = self.lattice_vector(axis).value();
            for j in 0..rank {
                upper[j][rank] = dot3(v, q[j]);
                for (x, u) in v.iter_mut().zip(q[j]) {
                    *x -= upper[j][rank] * u;
                }
            }
            let length = dot3(v, v).sqrt();
            upper[rank][rank] = length;
            q[rank] = v.map(|x| x / length);
            projected[rank] = orientation * dot3(residual, q[rank]);
            rank += 1;
        }
        let mut best = [0.0; 3];
        let mut radius2 = dot3(projected, projected);
        closest_translation(
            rank,
            &upper,
            projected,
            0.0,
            &mut [0.0; 3],
            &mut best,
            &mut radius2,
        );
        let mut image = n;
        for k in 0..rank {
            image[axes[k]] -= orientation * best[k];
        }
        self.frac_to_cart(image)
    }

    /// Minimum-image displacement `r_i - r_j` in Bohr.
    #[inline]
    pub fn mic_delta(&self, r_i_bohr: [f64; 3], r_j_bohr: [f64; 3]) -> [f64; 3] {
        let shift = self.mic_lattice_shift_bohr(r_i_bohr, r_j_bohr);
        [
            (r_i_bohr[0] - r_j_bohr[0]) + shift[0],
            (r_i_bohr[1] - r_j_bohr[1]) + shift[1],
            (r_i_bohr[2] - r_j_bohr[2]) + shift[2],
        ]
    }

    /// Minimum-image Euclidean distance `||r_i - r_j||_MIC` in Bohr.
    #[inline]
    pub fn mic_distance(&self, r_i_bohr: [f64; 3], r_j_bohr: [f64; 3]) -> f64 {
        let d = self.mic_delta(r_i_bohr, r_j_bohr);
        libm::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
    }

    /// Wrap a single Cartesian point (Bohr) into the primary `[0, 1)` unit cell along periodic axes.
    #[inline]
    pub fn wrap_point(&self, r_bohr: [f64; 3]) -> [f64; 3] {
        let mut s = self.cart_to_frac(r_bohr);
        for (v, periodic) in s.iter_mut().zip(self.periodic_axes) {
            if periodic {
                *v -= libm::floor(*v);
            }
        }
        self.frac_to_cart(s)
    }

    /// Perpendicular heights (`d_a, d_b, d_c`) between opposite faces of the unit cell in Bohr.
    pub fn perpendicular_widths_bohr(&self) -> [f64; 3] {
        // Row `k` of `H^{-1}` is the reciprocal vector `b_k` with `||b_k|| = 1 / d_k`.
        self.h_inv_bohr.map(|row| {
            let norm = libm::sqrt(row[0] * row[0] + row[1] * row[1] + row[2] * row[2]);
            if norm > 1e-16 {
                1.0 / norm
            } else {
                f64::INFINITY
            }
        })
    }

    /// Enumerate all Cartesian lattice translation vectors `L = n_a a + n_b b + n_c c` (in Bohr)
    /// that can bring two points in the unit cell within `cutoff_bohr`.
    /// Always includes `[0.0, 0.0, 0.0]` as the first element.
    pub fn lattice_shifts_within_cutoff(&self, cutoff_bohr: f64) -> Vec<[f64; 3]> {
        self.lattice_translations(self.translation_ranges(cutoff_bohr, 0, 1))
            .into_iter()
            .map(|(_, v)| v)
            .collect()
    }

    /// How many lattice translations along each axis a search within `cutoff_bohr` takes:
    /// `ceil(cutoff / d_k) + extra`, at least `minimum`, along the periodic axes with a finite
    /// width `d_k` ([`Self::perpendicular_widths_bohr`]); 0 along the others. With `extra = 1`
    /// every image of a point of the cell within the cutoff of another point of the cell is
    /// reached (two points of the cell differ by less than one cell along an axis).
    pub fn translation_ranges(&self, cutoff_bohr: f64, extra: i32, minimum: i32) -> [i32; 3] {
        let widths = self.perpendicular_widths_bohr();
        std::array::from_fn(|k| {
            if self.periodic_axes[k] && widths[k].is_finite() {
                (libm::ceil(cutoff_bohr / widths[k]) as i32 + extra).max(minimum)
            } else {
                0
            }
        })
    }

    /// The lattice translations `n` with `|n_k| <= ranges[k]` and their vectors (Bohr): the
    /// zero translation first, the others in lexicographic order of `n`. The one enumeration
    /// of the crate outside the xtb ports of GFN-FF.
    pub fn lattice_translations(&self, ranges: [i32; 3]) -> Vec<([i32; 3], [f64; 3])> {
        let mut shifts = vec![([0; 3], [0.0; 3])];
        for na in -ranges[0]..=ranges[0] {
            for nb in -ranges[1]..=ranges[1] {
                for nc in -ranges[2]..=ranges[2] {
                    if (na, nb, nc) != (0, 0, 0) {
                        shifts.push((
                            [na, nb, nc],
                            self.frac_to_cart([f64::from(na), f64::from(nb), f64::from(nc)]),
                        ));
                    }
                }
            }
        }
        shifts
    }
}

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Enumerate an upper triangular lattice inside the sphere of the best image.
/// Try the closest coefficient first, then expand on either side until pruned.
fn closest_translation(
    rank: usize,
    upper: &[[f64; 3]; 3],
    residual: [f64; 3],
    cost: f64,
    image: &mut [f64; 3],
    best: &mut [f64; 3],
    radius2: &mut f64,
) {
    if rank == 0 {
        if cost < *radius2 {
            *radius2 = cost;
            *best = *image;
        }
        return;
    }
    let k = rank - 1;
    let center = residual[k] / upper[k][k];
    let nearest = center.round();
    let sign = if center >= nearest { 1.0 } else { -1.0 };
    let mut trial = nearest;
    let mut stride = sign;
    loop {
        let next_cost = cost + (residual[k] - trial * upper[k][k]).powi(2);
        if next_cost >= *radius2 || !next_cost.is_finite() {
            break;
        }
        image[k] = trial;
        let mut remaining = residual;
        for i in 0..k {
            remaining[i] -= upper[i][k] * trial;
        }
        closest_translation(k, upper, remaining, next_cost, image, best, radius2);
        // In one dimension the nearest coefficient already minimizes the cost.
        if k == 0 || trial + stride == trial {
            break;
        }
        trial += stride;
        stride = -stride - stride.signum();
    }
}

fn parse_periodic_flags(spec: &str) -> MoptResult<[bool; 3]> {
    let tokens: Vec<&str> = if spec.contains([',', ' ']) {
        spec.split([',', ' '])
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect()
    } else {
        if spec.len() != 3 {
            return Err(MoptError::parse(format!(
                "Expected 3 periodic axis flags (e.g. 'TTF'), got '{spec}'"
            )));
        }
        vec![&spec[0..1], &spec[1..2], &spec[2..3]]
    };
    if tokens.len() != 3 {
        return Err(MoptError::parse(format!(
            "Expected 3 periodic axis flags, got {}",
            tokens.len()
        )));
    }
    let mut out = [true; 3];
    for (i, tok) in tokens.iter().enumerate() {
        out[i] = match tok.to_ascii_lowercase().as_str() {
            "t" | "true" | "1" | "p" | "yes" => true,
            "f" | "false" | "0" | "n" | "no" => false,
            _ => {
                return Err(MoptError::parse(format!(
                    "Invalid periodic axis flag '{tok}'"
                )))
            }
        };
    }
    Ok(out)
}

#[inline]
fn mat3_mul_vec(m: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

#[inline]
fn det_3x3(m: &[[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

fn invert_3x3(m: &[[f64; 3]; 3]) -> MoptResult<[[f64; 3]; 3]> {
    let det = det_3x3(m);
    if det.abs() < 1e-14 {
        return Err(MoptError::linalg(
            "Singular 3x3 periodic lattice matrix (zero cell volume)",
        ));
    }
    let inv = 1.0 / det;
    Ok([
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv,
        ],
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::quantity::shorthand::bohr;

    #[test]
    fn partial_skew_periodicity_uses_only_the_active_lattice() {
        for axes in [
            [true, true, true],
            [false, true, true],
            [true, false, true],
            [false, true, false],
        ] {
            let cell = PeriodicCell::from_lattice_vectors(
                Vector3::<Bohr>::new([4.0, 0.2, 0.0]),
                Vector3::new([3.7, 1.3, 0.0]),
                Vector3::new([2.1, 0.4, 3.0]),
                axes,
            )
            .unwrap();
            for fraction in [[0.49, 0.48, 0.31], [-0.7, 1.5, -1.2], [2.1, -1.2, 0.7]] {
                let raw = cell.frac_to_cart(fraction);
                let mut shortest = f64::INFINITY;
                for i in -5..=5 {
                    for j in -5..=5 {
                        for k in -5..=5 {
                            let index = [i, j, k];
                            if (0..3).any(|a| !axes[a] && index[a] != 0) {
                                continue;
                            }
                            let shift = cell.frac_to_cart(index.map(f64::from));
                            let d = std::array::from_fn(|a| raw[a] + shift[a]);
                            shortest = shortest.min(dot3(d, d));
                        }
                    }
                }
                let d = cell.mic_delta(raw, [0.0; 3]);
                assert!((dot3(d, d) - shortest).abs() < 1e-10);
                assert_eq!(cell.mic_delta([0.0; 3], raw), d.map(|v| -v));
            }
        }
    }

    #[test]
    fn skew_cell_compares_neighboring_images() {
        let cell = PeriodicCell::parse_cli_spec("10,10,10,90,90,60").unwrap();
        let delta = cell.frac_to_cart([0.49, 0.49, 0.0]);
        let expected = 10.0 * 0.2503_f64.sqrt();
        let distance = cell.mic_distance(delta, [0.0; 3]) * Units::BOHR_TO_ANGSTROM;
        assert!((distance - expected).abs() < 1e-10);
    }

    #[test]
    fn orthorhombic_and_triclinic_mic_behavior() {
        let cell = PeriodicCell::orthorhombic(bohr(10.0), bohr(10.0), bohr(10.0)).unwrap();
        let r1 = [0.5, 1.0, 9.2];
        let r2 = [9.5, 1.0, 0.2];
        let d = cell.mic_delta(r1, r2);
        assert!((d[0] - 1.0).abs() < 1e-12);
        assert!(d[1].abs() < 1e-12);
        assert!((d[2] - (-1.0)).abs() < 1e-12);

        let slab = PeriodicCell::parse_cli_spec("10.0,10.0,20.0:TTF").unwrap();
        assert_eq!(slab.periodic_axes, [true, true, false]);
        let ext = slab.to_extxyz_header();
        let reparsed = PeriodicCell::parse_extxyz_comment(&ext).unwrap();
        assert_eq!(reparsed.periodic_axes, [true, true, false]);
    }

    #[test]
    fn the_one_enumeration_of_translations_gives_the_former_two() {
        let cell = PeriodicCell::from_lattice_vectors(
            Vector3::<crate::domain::Bohr>::new([7.1, 0.0, 0.0]),
            Vector3::new([2.3, 6.4, 0.0]),
            Vector3::new([-1.2, 0.8, 9.7]),
            [true, true, false],
        )
        .unwrap();
        // The loops of this module and of `pes::native::neighbors` before they were one.
        let former = |extra: i32, minimum: i32, cutoff: f64| {
            let widths = cell.perpendicular_widths_bohr();
            let ranges: [i32; 3] = std::array::from_fn(|k| {
                if cell.periodic_axes[k] && widths[k].is_finite() {
                    let n = libm::ceil(cutoff / widths[k]) as i32;
                    if minimum > 0 {
                        n.max(minimum)
                    } else {
                        n + extra
                    }
                } else {
                    0
                }
            });
            let mut shifts = vec![([0; 3], [0.0; 3])];
            for na in -ranges[0]..=ranges[0] {
                for nb in -ranges[1]..=ranges[1] {
                    for nc in -ranges[2]..=ranges[2] {
                        if (na, nb, nc) != (0, 0, 0) {
                            shifts.push((
                                [na, nb, nc],
                                cell.frac_to_cart([na as f64, nb as f64, nc as f64]),
                            ));
                        }
                    }
                }
            }
            shifts
        };
        for cutoff in [0.5, 6.0, 7.0, 20.0, 41.3] {
            let a = cell.lattice_translations(cell.translation_ranges(cutoff, 1, 0));
            assert_eq!(a, former(1, 0, cutoff));
            let b = cell.lattice_shifts_within_cutoff(cutoff);
            let old: Vec<[f64; 3]> = former(0, 1, cutoff).into_iter().map(|(_, v)| v).collect();
            assert_eq!(b.len(), old.len());
            for (u, v) in b.iter().zip(&old) {
                assert_eq!(u.map(f64::to_bits), v.map(f64::to_bits));
            }
        }
    }

    #[test]
    fn the_lattice_read_back_from_a_file_keeps_its_volume_within_the_guarantee() {
        let cell = PeriodicCell::from_lattice_vectors(
            Vector3::<crate::domain::Bohr>::new([20.123_456_789_012_3, 0.3, -0.7]),
            Vector3::new([1.1, 19.876_543_210_987_6, 0.2]),
            Vector3::new([-0.4, 0.9, 21.000_000_000_123]),
            [true; 3],
        )
        .unwrap();
        let back = PeriodicCell::parse_extxyz_comment(&cell.to_extxyz_header()).unwrap();
        for k in 0..3 {
            let (u, v) = (
                cell.lattice_vector(k).value(),
                back.lattice_vector(k).value(),
            );
            for c in 0..3 {
                assert!((u[c] - v[c]).abs() < 1e-14, "{k} {c}: {} {}", u[c], v[c]);
            }
        }
        // About 8.4e3 Bohr^3, held to 1e-10 from the file.
        let dv = (cell.volume_bohr3() - back.volume_bohr3()).abs();
        assert!(dv < 1e-10, "{dv}");
    }
}
