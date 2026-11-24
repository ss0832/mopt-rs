//! Electronegativity equilibration (EEQ) charges as in xtb and gfn0-rs (`src/eeq.rs`):
//! `A q = x` with the total-charge constraint, `A_ij = s erf(r / sqrt(a_i^2 + a_j^2)) / r`,
//! `A_ii = eta_i + s sqrt(2/pi) / a_i`, `x_i = -chi_i + kappa_i CN_i / (sqrt(CN_i + 1e-12) +
//! 1e-14)` (`CN` clamped at 0). GFN0-xTB uses it twice: with its own parameters (charges of H0
//! and the electrostatic energy `q (A q / 2 - x)`) and with EEQ-2019 (charges of D4).
//!
//! Gradients: the electrostatic energy is stationary in `q` under the constraint, so only its
//! explicit dependence enters; other uses of the charges go through one adjoint solve
//! `K [lambda; mu] = [w; 0]` of the bordered matrix.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use crate::error::{MoptError, MoptResult};
use crate::math::linalg::DenseLu;

const SQRT_PI: f64 = 1.772_453_850_905_516;

/// Per-atom EEQ parameters `(chi, eta, kappa, width)`.
pub type EeqParams = [f64; 4];

/// A solved EEQ system.
#[derive(Debug, Clone)]
pub struct Eeq {
    /// Charges.
    pub q: Vec<f64>,
    /// Electrostatic energy `q (A q / 2 - x)`.
    pub energy: f64,
    kkt: Mat<f64>,
    /// The factorised `kkt` (once per structure: the charges and every response solve with
    /// it).
    lu: DenseLu,
    /// `d x_i / d CN_i`.
    dx_dcn: Vec<f64>,
    widths: Vec<f64>,
    scale: f64,
}

fn rhs_slope(cn: f64, kappa: f64) -> f64 {
    if cn <= 0.0 {
        return 0.0;
    }
    let root = (cn + 1.0e-12).sqrt();
    let den = root + 1.0e-14;
    kappa * (den - cn / (2.0 * root)) / (den * den)
}

fn dist(a: [f64; 3], b: [f64; 3]) -> ([f64; 3], f64) {
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    (d, r)
}

/// `d A_ij / d r`.
fn kernel_slope(r: f64, gamma: f64, scale: f64) -> f64 {
    let s = r / gamma;
    scale * (2.0 / (SQRT_PI * gamma) * (-(s * s)).exp() / r - libm::erf(s) / (r * r))
}

impl Eeq {
    /// Charges for parameters `p`, coordination numbers `cn`, total charge `charge`, and the
    /// Coulomb scale `scale`.
    pub fn solve(
        p: &[EeqParams],
        x: &[[f64; 3]],
        cn: &[f64],
        charge: f64,
        scale: f64,
    ) -> MoptResult<Self> {
        let n = x.len();
        let rhs_x: Vec<f64> = (0..n)
            .map(|i| {
                let c = cn[i].max(0.0);
                -p[i][0] + p[i][2] * c / ((c + 1.0e-12).sqrt() + 1.0e-14)
            })
            .collect();
        let widths: Vec<f64> = p.iter().map(|v| v[3].max(1.0e-12)).collect();
        let cself = (2.0 / SQRT_PI) * std::f64::consts::FRAC_1_SQRT_2;
        let mut kkt = Mat::<f64>::zeros(n + 1, n + 1);
        for i in 0..n {
            kkt[(i, i)] = p[i][1] + scale * cself / widths[i];
            kkt[(i, n)] = 1.0;
            kkt[(n, i)] = 1.0;
            for j in 0..i {
                let (_, r) = dist(x[i], x[j]);
                if r <= 1.0e-12 {
                    return Err(MoptError::calc("coincident atoms in the EEQ model"));
                }
                let gamma = (widths[i] * widths[i] + widths[j] * widths[j]).sqrt();
                let v = scale * libm::erf(r / gamma) / r;
                kkt[(i, j)] = v;
                kkt[(j, i)] = v;
            }
        }
        let mut b = Col::<f64>::zeros(n + 1);
        for i in 0..n {
            b[i] = rhs_x[i];
        }
        b[n] = charge;
        let lu = DenseLu::new(kkt.as_ref())?;
        let sol = solve(&lu, &b)?;
        let q: Vec<f64> = (0..n).map(|i| sol[i]).collect();
        let mut energy = 0.0;
        for i in 0..n {
            let aq: f64 = (0..n).map(|j| kkt[(i, j)] * q[j]).sum();
            energy += q[i] * (0.5 * aq - rhs_x[i]);
        }
        let dx_dcn = (0..n).map(|i| rhs_slope(cn[i], p[i][2])).collect();
        Ok(Self {
            q,
            energy,
            kkt,
            lu,
            dx_dcn,
            widths,
            scale,
        })
    }

    /// Adds `sum_ij v_i u_j dA_ij/dx` (symmetrized over the pair) to `grad`.
    fn add_kernel_gradient(&self, x: &[[f64; 3]], v: &[f64], u: &[f64], grad: &mut [f64]) {
        let n = x.len();
        for i in 0..n {
            for j in 0..i {
                let (d, r) = dist(x[i], x[j]);
                let gamma =
                    (self.widths[i] * self.widths[i] + self.widths[j] * self.widths[j]).sqrt();
                let s = (v[i] * u[j] + v[j] * u[i]) * kernel_slope(r, gamma, self.scale) / r;
                for k in 0..3 {
                    grad[3 * i + k] += s * d[k];
                    grad[3 * j + k] -= s * d[k];
                }
            }
        }
    }

    /// Gradient of the electrostatic energy at fixed charges: the Cartesian part is added to
    /// `grad`, the part through the coordination numbers to `w_cn` (`dE/dCN_i`).
    pub fn energy_gradient(&self, x: &[[f64; 3]], grad: &mut [f64], w_cn: &mut [f64]) {
        let half: Vec<f64> = self.q.iter().map(|v| 0.5 * v).collect();
        self.add_kernel_gradient(x, &half, &self.q, grad);
        for (i, w) in w_cn.iter_mut().enumerate() {
            *w -= self.q[i] * self.dx_dcn[i];
        }
    }

    /// The response of `sum_i w_i q_i` (`w = dE/dq`): Cartesian part added to `grad`, the part
    /// through the coordination numbers to `w_cn`.
    pub fn response(
        &self,
        x: &[[f64; 3]],
        w: &[f64],
        grad: &mut [f64],
        w_cn: &mut [f64],
    ) -> MoptResult<()> {
        let n = x.len();
        let mut b = Col::<f64>::zeros(n + 1);
        for i in 0..n {
            b[i] = w[i];
        }
        let lambda = solve(&self.lu, &b)?;
        let lam: Vec<f64> = (0..n).map(|i| lambda[i]).collect();
        // d(sum w q) = lambda^T (dx - dA q).
        let neg: Vec<f64> = lam.iter().map(|v| -v).collect();
        self.add_kernel_gradient(x, &neg, &self.q, grad);
        for i in 0..n {
            w_cn[i] += lam[i] * self.dx_dcn[i];
        }
        Ok(())
    }
}

fn solve(lu: &DenseLu, b: &Col<f64>) -> MoptResult<Col<f64>> {
    let out = lu.solve_col(b);
    if out.iter().all(|v| v.is_finite()) {
        Ok(out)
    } else {
        Err(MoptError::linalg("EEQ linear system is singular"))
    }
}
