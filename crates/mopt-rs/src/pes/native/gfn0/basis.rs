//! The GFN0-xTB minimal valence basis (STO-nG) and its overlap integrals with their gradient.
//!
//! As xtb and gfn0-rs (`src/basis.rs`, `src/integrals.rs`): STO-3G/4G/6G contractions by element
//! and shell, the diffuse s shell orthogonalized to the valence s shell of the same atom, real
//! spherical d functions, one-centre overlaps equal to the identity, and Gaussian-product
//! screening of shell pairs.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Mat;

use super::params::ElementParams;
use super::sto::{slater_to_gauss, Primitive};
use crate::error::{MoptError, MoptResult};

/// `x^a y^b z^c` with a coefficient.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Component {
    /// Cartesian powers.
    pub power: [usize; 3],
    /// Coefficient.
    pub coefficient: f64,
}

/// One shell of the basis.
#[derive(Debug, Clone, PartialEq)]
pub struct BasisShell {
    /// Atom index.
    pub atom: usize,
    /// Atomic number.
    pub z: u8,
    /// Angular momentum.
    pub l: usize,
    /// First shell of this angular momentum on the atom (the others are diffuse).
    pub valence: bool,
    /// First AO.
    pub first_ao: usize,
    /// Number of AOs.
    pub nao: usize,
    /// Level (Hartree).
    pub level: f64,
    /// Slater exponent.
    pub exponent: f64,
    /// CN shift.
    pub kcn: f64,
    /// Charge shift.
    pub kq: f64,
    /// Shell polynomial.
    pub poly: f64,
    /// Electronegativity of the atom.
    pub en: f64,
    /// Second-order charge shift of the atom.
    pub kqat2: f64,
    /// Atomic radius of the atom (Bohr).
    pub atomic_radius: f64,
    /// Contracted primitives (shared by the AOs of the shell).
    pub primitives: Vec<Primitive>,
    /// Smallest primitive exponent (screening).
    pub min_exponent: f64,
}

/// The basis of a structure.
#[derive(Debug, Clone, PartialEq)]
pub struct Basis {
    /// Shells.
    pub shells: Vec<BasisShell>,
    /// Cartesian components of every AO.
    pub components: Vec<Vec<Component>>,
}

/// Gaussian product screening: pairs with `alpha beta / (alpha + beta) r^2` above this are
/// below `exp(-40)`.
pub const OVERLAP_SCREEN: f64 = 40.0;

/// Primitive counts of xtb's GFN0 basis (`setGFN0NumberOfPrimitives` of `gfn0.f90`: STO-6G
/// for the 6s and 6p shells of the sixth period).
fn primitive_count(z: u8, n: u8, l: usize, valence: bool) -> MoptResult<usize> {
    if z <= 2 {
        return match l {
            0 => Ok(if valence { 3 } else { 2 }),
            1 => Ok(3),
            _ => Err(MoptError::config("GFN0 H/He basis has only s and p shells")),
        };
    }
    Ok(match l {
        0 | 1 => {
            if n > 5 {
                6
            } else if l == 0 {
                4
            } else {
                3
            }
        }
        _ => 4,
    })
}

fn same_center_s_overlap(a: &[Primitive], b: &[Primitive]) -> f64 {
    let mut value = 0.0;
    for pa in a {
        for pb in b {
            value += pa.coefficient
                * pb.coefficient
                * (std::f64::consts::PI / (pa.exponent + pb.exponent)).powf(1.5);
        }
    }
    value
}

pub(crate) fn orthogonalize_diffuse_s(valence: &[Primitive], diffuse: &mut Vec<Primitive>) {
    let overlap = same_center_s_overlap(valence, diffuse);
    diffuse.extend(valence.iter().map(|p| Primitive {
        exponent: p.exponent,
        coefficient: -overlap * p.coefficient,
    }));
    let norm = same_center_s_overlap(diffuse, diffuse).max(1.0e-30);
    let scale = 1.0 / norm.sqrt();
    for p in diffuse {
        p.coefficient *= scale;
    }
}

pub(crate) fn components(l: usize) -> MoptResult<Vec<Vec<Component>>> {
    let c = |power: [usize; 3], coefficient: f64| Component { power, coefficient };
    let s3 = 3.0_f64.sqrt();
    Ok(match l {
        0 => vec![vec![c([0, 0, 0], 1.0)]],
        1 => vec![
            vec![c([1, 0, 0], 1.0)],
            vec![c([0, 1, 0], 1.0)],
            vec![c([0, 0, 1], 1.0)],
        ],
        2 => vec![
            vec![c([2, 0, 0], 0.5 * s3), c([0, 2, 0], -0.5 * s3)],
            vec![c([2, 0, 0], 0.5), c([0, 2, 0], 0.5), c([0, 0, 2], -1.0)],
            vec![c([1, 1, 0], s3)],
            vec![c([1, 0, 1], s3)],
            vec![c([0, 1, 1], s3)],
        ],
        _ => {
            return Err(MoptError::config(format!(
                "the GFN0/GFN1-xTB basis has s, p and d shells only (no element of either \
                 method carries f functions; the lanthanides keep 4f in the core), got l = {l}"
            )))
        }
    })
}

impl Basis {
    /// The atom of every atomic orbital.
    pub fn atom_of_ao(&self) -> Vec<usize> {
        let mut out = vec![0; self.len()];
        for sh in &self.shells {
            for slot in &mut out[sh.first_ao..sh.first_ao + sh.nao] {
                *slot = sh.atom;
            }
        }
        out
    }

    /// The basis of atoms with parameters `elements`.
    pub fn new(elements: &[ElementParams]) -> MoptResult<Self> {
        let mut shells = Vec::new();
        let mut comps = Vec::new();
        for (atom, el) in elements.iter().enumerate() {
            let mut seen = [false; 5];
            let mut valence_s: Option<Vec<Primitive>> = None;
            for sh in &el.shells {
                let valence = !seen[sh.l];
                seen[sh.l] = true;
                let ng = primitive_count(el.z, sh.n, sh.l, valence)?;
                let mut primitives = slater_to_gauss(ng, sh.n, sh.l, sh.exponent)?;
                if sh.l == 0 {
                    if valence {
                        valence_s = Some(primitives.clone());
                    } else {
                        let reference = valence_s.as_ref().ok_or_else(|| {
                            MoptError::config("GFN0 diffuse s shell without a valence s shell")
                        })?;
                        orthogonalize_diffuse_s(reference, &mut primitives);
                    }
                }
                let ao = components(sh.l)?;
                let first_ao = comps.len();
                let nao = ao.len();
                comps.extend(ao);
                let min_exponent = primitives
                    .iter()
                    .map(|p| p.exponent)
                    .fold(f64::INFINITY, f64::min);
                shells.push(BasisShell {
                    atom,
                    z: el.z,
                    l: sh.l,
                    valence,
                    first_ao,
                    nao,
                    level: sh.level,
                    exponent: sh.exponent,
                    kcn: sh.kcn,
                    kq: sh.kq,
                    poly: sh.poly,
                    en: el.en,
                    kqat2: el.kqat2,
                    atomic_radius: el.atomic_radius,
                    primitives,
                    min_exponent,
                });
            }
        }
        Ok(Self {
            shells,
            components: comps,
        })
    }

    /// Number of AOs.
    pub fn len(&self) -> usize {
        self.components.len()
    }

    /// Whether there are no AOs.
    pub fn is_empty(&self) -> bool {
        self.components.is_empty()
    }

    /// Whether the shell pair is screened out at squared distance `r2`.
    pub fn negligible(&self, a: usize, b: usize, r2: f64) -> bool {
        let (ea, eb) = (self.shells[a].min_exponent, self.shells[b].min_exponent);
        ea * eb / (ea + eb) * r2 > OVERLAP_SCREEN
    }
}

const MAX_POWER: usize = 5;
type Tables = [[[f64; MAX_POWER]; MAX_POWER]; 3];

/// 1D overlap tables of one primitive pair up to powers `(li, lj)`.
fn fill_tables(
    li: usize,
    lj: usize,
    ca: [f64; 3],
    cb: [f64; 3],
    alpha: f64,
    beta: f64,
    t: &mut Tables,
) {
    let p = alpha + beta;
    let mu = alpha * beta / p;
    let inv2p = 0.5 / p;
    let root = (std::f64::consts::PI / p).sqrt();
    for axis in 0..3 {
        let (a, b) = (ca[axis], cb[axis]);
        let center = (alpha * a + beta * b) / p;
        let (pa, pb) = (center - a, center - b);
        let table = &mut t[axis];
        table[0][0] = (-mu * (a - b) * (a - b)).exp() * root;
        for i in 1..=li {
            table[i][0] = pa * table[i - 1][0];
            if i >= 2 {
                table[i][0] += (i - 1) as f64 * inv2p * table[i - 2][0];
            }
        }
        for j in 1..=lj {
            table[0][j] = pb * table[0][j - 1];
            if j >= 2 {
                table[0][j] += (j - 1) as f64 * inv2p * table[0][j - 2];
            }
        }
        for i in 1..=li {
            for j in 1..=lj {
                table[i][j] = pa * table[i - 1][j] + j as f64 * inv2p * table[i - 1][j - 1];
                if i >= 2 {
                    table[i][j] += (i - 1) as f64 * inv2p * table[i - 2][j];
                }
            }
        }
    }
}

fn r2(a: [f64; 3], b: [f64; 3]) -> f64 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)
}

/// Overlaps of a shell pair (row-major `nao_a x nao_b`).
pub fn pair_overlap(basis: &Basis, a: usize, b: usize, ca: [f64; 3], cb: [f64; 3]) -> Vec<f64> {
    let (sa, sb) = (&basis.shells[a], &basis.shells[b]);
    let mut values = vec![0.0; sa.nao * sb.nao];
    let mut t: Tables = [[[0.0; MAX_POWER]; MAX_POWER]; 3];
    let d2 = r2(ca, cb);
    for pa in &sa.primitives {
        for pb in &sb.primitives {
            if pa.exponent * pb.exponent / (pa.exponent + pb.exponent) * d2 > OVERLAP_SCREEN {
                continue;
            }
            fill_tables(sa.l, sb.l, ca, cb, pa.exponent, pb.exponent, &mut t);
            let pre = pa.coefficient * pb.coefficient;
            for row in 0..sa.nao {
                for col in 0..sb.nao {
                    let mut sum = 0.0;
                    for ac in &basis.components[sa.first_ao + row] {
                        for bc in &basis.components[sb.first_ao + col] {
                            sum += ac.coefficient
                                * bc.coefficient
                                * t[0][ac.power[0]][bc.power[0]]
                                * t[1][ac.power[1]][bc.power[1]]
                                * t[2][ac.power[2]][bc.power[2]];
                        }
                    }
                    values[row * sb.nao + col] += pre * sum;
                }
            }
        }
    }
    values
}

/// Gradients of the shell-pair overlaps with respect to centre A (row-major).
pub fn pair_overlap_gradient(
    basis: &Basis,
    a: usize,
    b: usize,
    ca: [f64; 3],
    cb: [f64; 3],
) -> Vec<[f64; 3]> {
    let (sa, sb) = (&basis.shells[a], &basis.shells[b]);
    let mut grads = vec![[0.0; 3]; sa.nao * sb.nao];
    let mut t: Tables = [[[0.0; MAX_POWER]; MAX_POWER]; 3];
    let d2 = r2(ca, cb);
    for pa in &sa.primitives {
        for pb in &sb.primitives {
            if pa.exponent * pb.exponent / (pa.exponent + pb.exponent) * d2 > OVERLAP_SCREEN {
                continue;
            }
            fill_tables(sa.l + 1, sb.l, ca, cb, pa.exponent, pb.exponent, &mut t);
            let two_alpha = 2.0 * pa.exponent;
            let pre = pa.coefficient * pb.coefficient;
            for row in 0..sa.nao {
                for col in 0..sb.nao {
                    let mut g = [0.0; 3];
                    for ac in &basis.components[sa.first_ao + row] {
                        for bc in &basis.components[sb.first_ao + col] {
                            let c = ac.coefficient * bc.coefficient;
                            let s = [0, 1, 2].map(|k| t[k][ac.power[k]][bc.power[k]]);
                            let ds = [0, 1, 2].map(|k| {
                                let mut d = two_alpha * t[k][ac.power[k] + 1][bc.power[k]];
                                if ac.power[k] > 0 {
                                    d -= ac.power[k] as f64 * t[k][ac.power[k] - 1][bc.power[k]];
                                }
                                d
                            });
                            g[0] += c * ds[0] * s[1] * s[2];
                            g[1] += c * s[0] * ds[1] * s[2];
                            g[2] += c * s[0] * s[1] * ds[2];
                        }
                    }
                    let out = &mut grads[row * sb.nao + col];
                    for k in 0..3 {
                        out[k] += g[k] * pre;
                    }
                }
            }
        }
    }
    grads
}

/// Second derivatives of the shell-pair overlaps with respect to centre A (row-major; each
/// a symmetric 3 x 3 block). The overlaps depend on `A - B` only, so `d2/dA dB = -d2/dA2`.
/// With the 1D tables `t[i][j]` of a primitive pair (exponent `alpha` on A):
/// `d/dA_x` of `x^i` is `2 alpha t[i+1] - i t[i-1]`, and
/// `d2/dA_x2` is `4 alpha^2 t[i+2] - 2 alpha (2 i + 1) t[i] + i (i - 1) t[i-2]`.
pub fn pair_overlap_hessian(
    basis: &Basis,
    a: usize,
    b: usize,
    ca: [f64; 3],
    cb: [f64; 3],
) -> Vec<[[f64; 3]; 3]> {
    let (sa, sb) = (&basis.shells[a], &basis.shells[b]);
    let mut out = vec![[[0.0; 3]; 3]; sa.nao * sb.nao];
    let mut t: Tables = [[[0.0; MAX_POWER]; MAX_POWER]; 3];
    let d2 = r2(ca, cb);
    for pa in &sa.primitives {
        for pb in &sb.primitives {
            if pa.exponent * pb.exponent / (pa.exponent + pb.exponent) * d2 > OVERLAP_SCREEN {
                continue;
            }
            fill_tables(sa.l + 2, sb.l, ca, cb, pa.exponent, pb.exponent, &mut t);
            let al = pa.exponent;
            let pre = pa.coefficient * pb.coefficient;
            for row in 0..sa.nao {
                for col in 0..sb.nao {
                    let mut hb = [[0.0; 3]; 3];
                    for ac in &basis.components[sa.first_ao + row] {
                        for bc in &basis.components[sb.first_ao + col] {
                            let c = ac.coefficient * bc.coefficient;
                            let s = [0, 1, 2].map(|k| t[k][ac.power[k]][bc.power[k]]);
                            let ds = [0, 1, 2].map(|k| {
                                let i = ac.power[k];
                                let mut d = 2.0 * al * t[k][i + 1][bc.power[k]];
                                if i > 0 {
                                    d -= i as f64 * t[k][i - 1][bc.power[k]];
                                }
                                d
                            });
                            let dds = [0, 1, 2].map(|k| {
                                let i = ac.power[k];
                                let j = bc.power[k];
                                let mut d = 4.0 * al * al * t[k][i + 2][j]
                                    - 2.0 * al * (2 * i + 1) as f64 * t[k][i][j];
                                if i > 1 {
                                    d += (i * (i - 1)) as f64 * t[k][i - 2][j];
                                }
                                d
                            });
                            for p in 0..3 {
                                for q in 0..3 {
                                    let v = if p == q {
                                        let o = [0, 1, 2].iter().filter(|&&m| m != p);
                                        o.fold(dds[p], |acc, &m| acc * s[m])
                                    } else {
                                        let m = 3 - p - q;
                                        ds[p] * ds[q] * s[m]
                                    };
                                    hb[p][q] += c * v;
                                }
                            }
                        }
                    }
                    let o = &mut out[row * sb.nao + col];
                    for p in 0..3 {
                        for q in 0..3 {
                            o[p][q] += pre * hb[p][q];
                        }
                    }
                }
            }
        }
    }
    out
}

/// Dipole integrals `<mu| r_c - o_c |nu>` of a shell pair (row-major `nao_a x nao_b`, one
/// block per Cartesian component `c`), from the one-dimensional tables of the overlaps:
/// `(x - o) = (x - B) + (B - o)` raises the power on centre B by one,
/// `<a| x - o |b> = t[i][j + 1] + (B_x - o_x) t[i][j]` along the axis of the component.
pub fn pair_dipole(
    basis: &Basis,
    a: usize,
    b: usize,
    ca: [f64; 3],
    cb: [f64; 3],
    origin: [f64; 3],
) -> [Vec<f64>; 3] {
    let (sa, sb) = (&basis.shells[a], &basis.shells[b]);
    let mut values = [
        vec![0.0; sa.nao * sb.nao],
        vec![0.0; sa.nao * sb.nao],
        vec![0.0; sa.nao * sb.nao],
    ];
    let mut t: Tables = [[[0.0; MAX_POWER]; MAX_POWER]; 3];
    let d2 = r2(ca, cb);
    for pa in &sa.primitives {
        for pb in &sb.primitives {
            if pa.exponent * pb.exponent / (pa.exponent + pb.exponent) * d2 > OVERLAP_SCREEN {
                continue;
            }
            fill_tables(sa.l, sb.l + 1, ca, cb, pa.exponent, pb.exponent, &mut t);
            let pre = pa.coefficient * pb.coefficient;
            for row in 0..sa.nao {
                for col in 0..sb.nao {
                    for (c, block) in values.iter_mut().enumerate() {
                        let mut sum = 0.0;
                        for ac in &basis.components[sa.first_ao + row] {
                            for bc in &basis.components[sb.first_ao + col] {
                                let mut product = ac.coefficient * bc.coefficient;
                                for axis in 0..3 {
                                    let (i, j) = (ac.power[axis], bc.power[axis]);
                                    product *= if axis == c {
                                        t[axis][i][j + 1]
                                            + (cb[axis] - origin[axis]) * t[axis][i][j]
                                    } else {
                                        t[axis][i][j]
                                    };
                                }
                                sum += product;
                            }
                        }
                        block[row * sb.nao + col] += pre * sum;
                    }
                }
            }
        }
    }
    values
}

/// The dipole integrals `<mu| r - origin |nu>` (Bohr) of the whole basis at the positions `x`,
/// one matrix per Cartesian component. Unlike the overlaps, the one-centre blocks are computed
/// (the s-p and p-d pairs of an atom have dipoles).
pub fn dipole(basis: &Basis, x: &[[f64; 3]], origin: [f64; 3]) -> [Mat<f64>; 3] {
    let n = basis.len();
    let mut d = [
        Mat::<f64>::zeros(n, n),
        Mat::<f64>::zeros(n, n),
        Mat::<f64>::zeros(n, n),
    ];
    let ns = basis.shells.len();
    for a in 0..ns {
        for b in a..ns {
            let (sa, sb) = (&basis.shells[a], &basis.shells[b]);
            let (ca, cb) = (x[sa.atom], x[sb.atom]);
            if sa.atom != sb.atom && basis.negligible(a, b, r2(ca, cb)) {
                continue;
            }
            let blocks = pair_dipole(basis, a, b, ca, cb, origin);
            for (c, block) in blocks.iter().enumerate() {
                for row in 0..sa.nao {
                    for col in 0..sb.nao {
                        let v = block[row * sb.nao + col];
                        d[c][(sa.first_ao + row, sb.first_ao + col)] = v;
                        d[c][(sb.first_ao + col, sa.first_ao + row)] = v;
                    }
                }
            }
        }
    }
    d
}

/// The overlap matrix (one-centre blocks are the identity).
pub fn overlap(basis: &Basis, x: &[[f64; 3]]) -> Mat<f64> {
    let n = basis.len();
    let mut s = Mat::<f64>::zeros(n, n);
    for mu in 0..n {
        s[(mu, mu)] = 1.0;
    }
    let ns = basis.shells.len();
    for a in 0..ns {
        for b in a + 1..ns {
            let (sa, sb) = (&basis.shells[a], &basis.shells[b]);
            if sa.atom == sb.atom {
                continue;
            }
            let (ca, cb) = (x[sa.atom], x[sb.atom]);
            if basis.negligible(a, b, r2(ca, cb)) {
                continue;
            }
            let block = pair_overlap(basis, a, b, ca, cb);
            for row in 0..sa.nao {
                for col in 0..sb.nao {
                    let v = block[row * sb.nao + col];
                    s[(sa.first_ao + row, sb.first_ao + col)] = v;
                    s[(sb.first_ao + col, sa.first_ao + row)] = v;
                }
            }
        }
    }
    s
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::domain::Element;

    /// `int (x - a)^i (x - b)^j (x - o)^k exp(-p (x - a)^2 - q (x - b)^2) dx` by the
    /// trapezoidal rule (exact to rounding for these Gaussians on this grid).
    #[allow(clippy::too_many_arguments)]
    fn quadrature(i: usize, j: usize, k: usize, a: f64, b: f64, o: f64, p: f64, q: f64) -> f64 {
        let (lo, hi) = (a.min(b) - 16.0, a.max(b) + 16.0);
        let n = ((hi - lo) / 0.01).ceil() as usize;
        let h = (hi - lo) / n as f64;
        (0..=n)
            .map(|s| {
                let x = lo + h * s as f64;
                (x - a).powi(i as i32)
                    * (x - b).powi(j as i32)
                    * (x - o).powi(k as i32)
                    * (-p * (x - a).powi(2) - q * (x - b).powi(2)).exp()
            })
            .sum::<f64>()
            * h
    }

    /// `<mu| (r_c - o_c) |nu>` of the shells `a`, `b` by quadrature (`c = None`: the overlap).
    fn by_quadrature(
        basis: &Basis,
        a: usize,
        b: usize,
        ca: [f64; 3],
        cb: [f64; 3],
        origin: [f64; 3],
        c: Option<usize>,
    ) -> Vec<f64> {
        let (sa, sb) = (&basis.shells[a], &basis.shells[b]);
        let mut out = vec![0.0; sa.nao * sb.nao];
        for row in 0..sa.nao {
            for col in 0..sb.nao {
                let mut v = 0.0;
                for pa in &sa.primitives {
                    for pb in &sb.primitives {
                        for ac in &basis.components[sa.first_ao + row] {
                            for bc in &basis.components[sb.first_ao + col] {
                                let mut product = pa.coefficient
                                    * pb.coefficient
                                    * ac.coefficient
                                    * bc.coefficient;
                                for axis in 0..3 {
                                    product *= quadrature(
                                        ac.power[axis],
                                        bc.power[axis],
                                        usize::from(c == Some(axis)),
                                        ca[axis],
                                        cb[axis],
                                        origin[axis],
                                        pa.exponent,
                                        pb.exponent,
                                    );
                                }
                                v += product;
                            }
                        }
                    }
                }
                out[row * sb.nao + col] = v;
            }
        }
        out
    }

    #[test]
    fn dipole_integrals_match_quadrature_and_shift_with_the_origin() {
        // S with its s, p, and d shells, and H, off the axes.
        let elements: Vec<ElementParams> = [16u8, 1]
            .iter()
            .map(|&z| {
                super::super::params::element(Element::from_atomic_number(z).unwrap()).unwrap()
            })
            .collect();
        let basis = Basis::new(&elements).unwrap();
        let x = [[0.1, -0.2, 0.05], [0.8, 0.6, 2.4]];
        let origin = [0.7, -0.4, 0.3];
        let ns = basis.shells.len();
        let mut worst = 0.0_f64;
        for a in 0..ns {
            for b in 0..ns {
                let (ca, cb) = (x[basis.shells[a].atom], x[basis.shells[b].atom]);
                let d = pair_dipole(&basis, a, b, ca, cb, origin);
                for (c, block) in d.iter().enumerate() {
                    let q = by_quadrature(&basis, a, b, ca, cb, origin, Some(c));
                    for (u, v) in block.iter().zip(&q) {
                        worst = worst.max((u - v).abs());
                    }
                }
                if basis.shells[a].atom != basis.shells[b].atom {
                    let s = pair_overlap(&basis, a, b, ca, cb);
                    let q = by_quadrature(&basis, a, b, ca, cb, origin, None);
                    for (u, v) in s.iter().zip(&q) {
                        worst = worst.max((u - v).abs());
                    }
                }
            }
        }
        assert!(worst < 1e-9, "largest difference {worst:e}");
        // Symmetric, and moving the origin by t subtracts t times the overlap (the one-centre
        // blocks of the integrals by quadrature, which are the identity to the fit of the
        // contractions).
        let d0 = dipole(&basis, &x, [0.0; 3]);
        let d1 = dipole(&basis, &x, origin);
        let n = basis.len();
        let s = overlap(&basis, &x);
        for c in 0..3 {
            for i in 0..n {
                for j in 0..n {
                    assert!((d0[c][(i, j)] - d0[c][(j, i)]).abs() < 1e-14);
                    let shifted = d0[c][(i, j)] - origin[c] * s[(i, j)];
                    assert!((d1[c][(i, j)] - shifted).abs() < 1e-6, "{c} {i} {j}");
                }
            }
        }
    }
}
