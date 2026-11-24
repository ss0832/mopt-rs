//! Anisotropic stochastic cell rescaling (Del Tatto, Raiteri, Bernetti, and Bussi): the
//! first-order stochastic barostat of Bernetti and Bussi ([`super::CellRescale`]) for a cell
//! whose lengths and angles all change, at a constant isotropic pressure `P0`.
//!
//! The cell matrix `h` (the lattice vectors as columns) follows
//!
//! ```text
//! dh = -(beta / (3 tau)) [(P0 I - P_int) - (k_B T / V) I] h dt
//!      + sqrt(2 beta k_B T / (3 V tau)) dW h
//! ```
//!
//! (eq. 2.17 of the thesis; `dW` nine independent Wiener processes, `beta` the assumed
//! isothermal compressibility, `tau` the relaxation time), with the positions taken along
//! (`r -> mu r`) and the scaled momenta kept (`v -> mu^(-T) v`). It samples the isothermal-
//! isobaric ensemble of the fully flexible cell (their eq. 2.2). The step is the
//! time-reversible splitting of the thesis (Algorithm 7, eqs. 2.37-2.40): a half step of the
//! volume, a full step of the shape `mu_s = exp(A_2 dt + b dW_2)` with the traceless parts
//! `A_2` of `(beta / (3 tau)) P_int` and `dW_2` of `dW` and `b` at the volume of the half
//! step, and another half step of the volume; `mu = (V' / V)^(1/3) mu_s`. The volume follows
//! `d ln V = -(beta / tau) (P0 - P) dt + sqrt(2 k_B T beta / (V tau)) dW` as in the isotropic
//! barostat (the `k_B T / V` term cancels in the Ito form for `ln V`). The rotations of the
//! cell that the noise brings are taken out after every scaling by turning the cell back so
//! that `a` lies along `x` and `b` in the `xy` plane (the Gram-Schmidt form of section 2.6),
//! with the positions and velocities turned with it. The internal pressure tensor is
//! symmetrized. Here `exp` of the 3 x 3 matrix is a Taylor series after scaling and
//! squaring, renormalized to determinant one (the thesis uses a Pade form, eq. 2.41).
//!
//! References: V. Del Tatto, P. Raiteri, M. Bernetti, G. Bussi, Appl. Sci. 2022, 12, 1139
//! (not read); V. Del Tatto, A Fully Anisotropic Formulation of Stochastic Cell Rescaling,
//! Master's thesis, University of Trento 2021, arXiv:2111.06402 (sections 2.1, 2.5, 2.6);
//! M. Bernetti, G. Bussi, J. Chem. Phys. 2020, 153, 114107 (the isotropic barostat).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use super::{scale_positions_and_cell, Barostat, BarostatContext};
use crate::domain::PeriodicCell;
use crate::error::{MoptError, MoptResult};
use crate::md::thermostat::positive;

type M3 = [[f64; 3]; 3];

const IDENTITY: M3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

fn product(a: &M3, b: &M3) -> M3 {
    std::array::from_fn(|i| std::array::from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum()))
}

fn determinant(m: &M3) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

/// The inverse of a 3 x 3 matrix (`None` when it is singular).
fn inverse(m: &M3) -> Option<M3> {
    let det = determinant(m);
    if !(det.is_finite() && det.abs() > 1e-300) {
        return None;
    }
    let c = |i: usize, j: usize| {
        let (r0, r1) = ((i + 1) % 3, (i + 2) % 3);
        let (c0, c1) = ((j + 1) % 3, (j + 2) % 3);
        m[r0][c0] * m[r1][c1] - m[r0][c1] * m[r1][c0]
    };
    Some(std::array::from_fn(|i| {
        std::array::from_fn(|j| c(j, i) / det)
    }))
}

/// `exp(m)` by scaling and squaring of a Taylor series of twelve terms.
pub(super) fn exponential(m: &M3) -> M3 {
    let norm = m.iter().flatten().map(|x| x.abs()).sum::<f64>();
    let mut squarings = 0;
    let mut scale = 1.0;
    while norm * scale > 0.25 && squarings < 60 {
        scale *= 0.5;
        squarings += 1;
    }
    let a: M3 = std::array::from_fn(|i| std::array::from_fn(|j| m[i][j] * scale));
    let mut sum = IDENTITY;
    let mut term = IDENTITY;
    for k in 1..=12 {
        term = product(&term, &a);
        for i in 0..3 {
            for j in 0..3 {
                term[i][j] /= k as f64;
                sum[i][j] += term[i][j];
            }
        }
    }
    for _ in 0..squarings {
        sum = product(&sum, &sum);
    }
    sum
}

/// The proper rotation that turns the lattice vectors `rows` (as rows) so that the first
/// lies along `x` and the second in the `xy` plane.
fn upright(rows: &M3) -> Option<M3> {
    let norm = |v: [f64; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    let a = rows[0];
    let la = norm(a);
    if !(la.is_finite() && la > 0.0) {
        return None;
    }
    let e1 = a.map(|x| x / la);
    let b = rows[1];
    let along = b[0] * e1[0] + b[1] * e1[1] + b[2] * e1[2];
    let rest = [
        b[0] - along * e1[0],
        b[1] - along * e1[1],
        b[2] - along * e1[2],
    ];
    let lr = norm(rest);
    if !(lr.is_finite() && lr > 0.0) {
        return None;
    }
    let e2 = rest.map(|x| x / lr);
    let e3 = [
        e1[1] * e2[2] - e1[2] * e2[1],
        e1[2] * e2[0] - e1[0] * e2[2],
        e1[0] * e2[1] - e1[1] * e2[0],
    ];
    Some([e1, e2, e3])
}

/// Anisotropic stochastic cell rescaling.
#[derive(Debug, Clone)]
pub struct CellRescaleAniso {
    tau: f64,
    compressibility: f64,
    /// The scaling of the cell to apply with the next drift.
    pending: M3,
}

impl CellRescaleAniso {
    /// Relaxation time `tau` (atomic units) and compressibility (Bohr^3 / Hartree).
    pub fn new(tau: f64, compressibility: f64) -> MoptResult<Self> {
        Ok(Self {
            tau: positive(tau, "the cell-rescaling coupling time")?,
            compressibility: positive(compressibility, "the compressibility")?,
            pending: IDENTITY,
        })
    }

    /// `d ln V` for `dt` at the volume `volume` and the pressure `pressure`.
    fn volume_step(
        &self,
        dt: f64,
        volume: f64,
        pressure: f64,
        ctx: &mut BarostatContext<'_>,
    ) -> f64 {
        let kt = ctx.kt();
        -self.compressibility / self.tau * (ctx.pressure - pressure) * dt
            + libm::sqrt(2.0 * kt * self.compressibility * dt / (volume * self.tau))
                * ctx.rng.standard_normal()
    }
}

impl Barostat for CellRescaleAniso {
    fn name(&self) -> &str {
        "crescale_aniso"
    }

    fn apply(
        &mut self,
        dt: f64,
        _v: &mut Col<f64>,
        ctx: &mut BarostatContext<'_>,
    ) -> MoptResult<()> {
        let kt = ctx.kt();
        let raw = ctx.pressure_tensor();
        let p: M3 = std::array::from_fn(|a| std::array::from_fn(|b| 0.5 * (raw[a][b] + raw[b][a])));
        let mean = (p[0][0] + p[1][1] + p[2][2]) / 3.0;
        let v0 = ctx.volume;
        // A half step of the volume, the shape, another half step of the volume.
        let first = self.volume_step(0.5 * dt, v0, mean, ctx);
        let v1 = v0 * libm::exp(first);
        let b = libm::sqrt(2.0 * self.compressibility * kt / (3.0 * v1 * self.tau));
        let mut noise: M3 = std::array::from_fn(|_| {
            std::array::from_fn(|_| libm::sqrt(dt) * ctx.rng.standard_normal())
        });
        let trace = (noise[0][0] + noise[1][1] + noise[2][2]) / 3.0;
        for (k, row) in noise.iter_mut().enumerate() {
            row[k] -= trace;
        }
        let a = self.compressibility / (3.0 * self.tau);
        let shape: M3 = std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                let traceless = p[i][j] - if i == j { mean } else { 0.0 };
                a * traceless * dt + b * noise[i][j]
            })
        });
        let mut mu_s = exponential(&shape);
        let det = determinant(&mu_s);
        if !(det.is_finite() && det > 0.0) {
            return Err(MoptError::calc(
                "the anisotropic cell rescaling gave a singular change of shape",
            ));
        }
        let fix = libm::cbrt(det);
        for x in mu_s.iter_mut().flatten() {
            *x /= fix;
        }
        let second = self.volume_step(0.5 * dt, v1, mean, ctx);
        let iso = libm::exp((first + second) / 3.0);
        let mu: M3 = std::array::from_fn(|i| std::array::from_fn(|j| iso * mu_s[i][j]));
        self.pending = product(&mu, &self.pending);
        Ok(())
    }

    fn drift(
        &mut self,
        h: f64,
        x: &mut Col<f64>,
        v: &mut Col<f64>,
        cell: &mut PeriodicCell,
    ) -> MoptResult<()> {
        if self.pending != IDENTITY {
            // The scaled lattice vectors, then the rotation that takes them back upright.
            let rows: M3 = std::array::from_fn(|k| {
                let r = cell.lattice_vector(k).value();
                std::array::from_fn(|a| (0..3).map(|c| self.pending[a][c] * r[c]).sum())
            });
            let turn =
                upright(&rows).ok_or_else(|| MoptError::calc("the rescaled cell has collapsed"))?;
            let mu = product(&turn, &self.pending);
            let inverse = inverse(&mu)
                .ok_or_else(|| MoptError::calc("the rescaling of the cell is singular"))?;
            scale_positions_and_cell(&mu, x, cell)?;
            // v -> mu^(-T) v.
            for i in 0..v.nrows() / 3 {
                let w = [v[3 * i], v[3 * i + 1], v[3 * i + 2]];
                for a in 0..3 {
                    v[3 * i + a] = (0..3).map(|c| inverse[c][a] * w[c]).sum();
                }
            }
            self.pending = IDENTITY;
        }
        for (p, w) in x.iter_mut().zip(v.iter()) {
            *p += h * w;
        }
        Ok(())
    }

    fn reference(&self) -> Option<&'static str> {
        Some(
            "V. Del Tatto, P. Raiteri, M. Bernetti, G. Bussi, Appl. Sci. 2022, 12, 1139; M. \
             Bernetti, G. Bussi, J. Chem. Phys. 2020, 153, 114107",
        )
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::core_infra::rng::Rng;
    use crate::domain::{Bohr, Quantity};
    use crate::md::space::MdSpace;

    #[test]
    fn the_matrix_functions() {
        let m = [[0.1, 0.02, -0.03], [0.04, -0.05, 0.01], [0.0, 0.02, -0.05]];
        let e = exponential(&m);
        // det exp(m) = exp(tr m) = 1 for this traceless m; exp(m) exp(-m) = I.
        assert!((determinant(&e) - 1.0).abs() < 1e-13);
        let minus = m.map(|r| r.map(|x| -x));
        let back = product(&e, &exponential(&minus));
        let inv = inverse(&e).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((back[i][j] - want).abs() < 1e-13);
                assert!((inv[i][j] - exponential(&minus)[i][j]).abs() < 1e-12);
            }
        }
        let large = m.map(|r| r.map(|x| 30.0 * x));
        assert!((determinant(&exponential(&large)) - 1.0).abs() < 1e-10);
        let turn = upright(&[[1.0, 1.0, 0.0], [0.0, 2.0, 0.5], [0.3, 0.2, 3.0]]).unwrap();
        assert!((determinant(&turn) - 1.0).abs() < 1e-14);
        let a = [1.0, 1.0, 0.0];
        let ra: Vec<f64> = (0..3)
            .map(|i| (0..3).map(|k| turn[i][k] * a[k]).sum())
            .collect();
        assert!(ra[1].abs() < 1e-14 && ra[2].abs() < 1e-14 && ra[0] > 0.0);
    }

    /// An ideal gas of `dof` degrees of freedom with a traceless elastic stress against
    /// changes of the shape of the cell (an ideal gas alone has no shear stiffness, and its
    /// cell wanders to any shape): the volume samples `V^(dof/3) exp(-P0 V / kT)` as with the
    /// isotropic barostat (the traceless stress does not enter the volume), and the cell
    /// changes its shape, staying upright.
    #[test]
    fn the_anisotropic_cell_rescaling_of_an_ideal_gas() {
        let space = MdSpace::variables(3).unwrap();
        let kt = 1.0e-3;
        let temperature = kt / crate::domain::Units::BOLTZMANN_HARTREE_PER_K;
        let (dof, p0) = (30.0, 1.0e-5);
        let expected = (dof / 3.0 + 1.0) * kt / p0;
        let mut barostat = CellRescaleAniso::new(50.0, 1.0 / (p0 * 10.0)).unwrap();
        let edge = libm::cbrt(expected);
        let mut cell = PeriodicCell::orthorhombic(
            Quantity::<Bohr>::new(edge),
            Quantity::new(edge),
            Quantity::new(edge),
        )
        .unwrap();
        let mut rng = Rng::from_seed(4);
        let (mut x, mut v) = (Col::zeros(3), Col::zeros(3));
        let mut sum = 0.0;
        let mut ratio: f64 = 1.0;
        let shear = 10.0 * p0;
        let steps = 400_000;
        for step in 0..steps {
            let mut ctx = BarostatContext {
                space: &space,
                dof,
                temperature,
                pressure: p0,
                volume: cell.volume_bohr3(),
                kinetic: std::array::from_fn(|a| {
                    std::array::from_fn(|b| if a == b { dof * kt / 3.0 } else { 0.0 })
                }),
                stress: shape_stress(&cell, shear),
                rng: &mut rng,
            };
            barostat.apply(1.0, &mut v, &mut ctx).unwrap();
            barostat.drift(1.0, &mut x, &mut v, &mut cell).unwrap();
            if step >= steps / 10 {
                sum += cell.volume_bohr3();
                let a = cell.lattice_vector(0).value();
                let b = cell.lattice_vector(1).value();
                // Upright to round-off of the size of the cell (with no shear stiffness the
                // shape of an ideal gas wanders far: lengths of 1e6 and 1e-3 Bohr).
                let size = a[0].abs().max(b[0].abs()).max(b[1].abs());
                assert!(
                    a[1].abs() < 1e-8 * size
                        && a[2].abs() < 1e-8 * size
                        && b[2].abs() < 1e-8 * size,
                    "step {step}: a {a:?} b {b:?}"
                );
                let la = (a[0] * a[0]).sqrt();
                let lb = (b[0] * b[0] + b[1] * b[1]).sqrt();
                ratio = ratio.max(la / lb).max(lb / la);
            }
        }
        let mean = sum / (steps - steps / 10) as f64;
        assert!(
            (mean / expected - 1.0).abs() < 0.03,
            "{mean} against {expected}"
        );
        assert!(ratio > 1.05 && ratio < 3.0, "the shape moves: {ratio}");
    }

    /// `G dev(eps)` for the strain `eps = sym(F) - I` of the shape `F = h / V^(1/3)` against
    /// a cube.
    fn shape_stress(cell: &PeriodicCell, shear: f64) -> [[f64; 3]; 3] {
        let scale = libm::cbrt(cell.volume_bohr3());
        let f: [[f64; 3]; 3] = std::array::from_fn(|i| {
            std::array::from_fn(|k| cell.lattice_vector(k).value()[i] / scale)
        });
        let eps: [[f64; 3]; 3] = std::array::from_fn(|i| {
            std::array::from_fn(|j| 0.5 * (f[i][j] + f[j][i]) - if i == j { 1.0 } else { 0.0 })
        });
        let mean = (eps[0][0] + eps[1][1] + eps[2][2]) / 3.0;
        std::array::from_fn(|i| {
            std::array::from_fn(|j| shear * (eps[i][j] - if i == j { mean } else { 0.0 }))
        })
    }
}
