//! The anisotropic (fully flexible) barostat of Martyna, Tobias, and Klein: every length and
//! angle of the cell follows its component of the pressure tensor, with the
//! measure-preserving integration of Tuckerman, Alejandre, Lopez-Rendon, Jochim, and
//! Martyna.
//!
//! Equations of motion with the symmetric velocity of the cell `v_g = p_g / W_g` (3 x 3),
//! `N_f` degrees of freedom, and `K = sum m v^2 / 2`:
//!
//! ```text
//! dr/dt   = v + v_g r,
//! dv/dt   = F / m - (v_g + Tr(v_g) / N_f) v,
//! dh/dt   = v_g h,
//! dp_g/dt = V (P_int - P0 I) + (2 K / N_f) I,
//! ```
//!
//! with a Nose-Hoover chain of its own on `p_g`. The internal pressure tensor is symmetrized,
//! so that `v_g` stays symmetric and the cell does not rotate (Martyna et al.). The mass of
//! every component (a choice made here; the papers leave it to the user) is
//! `W_g = (N_f + 3) k_B T tau^2 / 3`, a third of the mass of the isotropic
//! barostat ([`super::Mtk`]), so that the volume moves with the same period: for
//! `v_g = v_e I` the kinetic energy `W_g Tr(v_g^2) / 2` is `3 W_g v_e^2 / 2`. The conserved
//! quantity is `K + U + W_g Tr(v_g^2) / 2 + P0 V` and the energies of the chains.
//!
//! The kicks and drifts act in the eigenvectors of `v_g`, where they are those of the
//! isotropic barostat along each eigenvector: `v' = v' e^(-z) + a' h e^(-z/2) sinhc(z/2)`
//! with `z = (lambda + Tr(v_g) / N_f) h`, and `r' = r' e^(lambda h) + v' h e^(lambda h / 2)
//! sinhc(lambda h / 2)`; the cell is multiplied by `O e^(Lambda h) O^T`.
//!
//! References: G. J. Martyna, D. J. Tobias, M. L. Klein, J. Chem. Phys. 1994, 101, 4177 (the
//! equations; not read); M. E. Tuckerman, J. Alejandre, R. Lopez-Rendon, A. L. Jochim, G. J.
//! Martyna, J. Phys. A 2006, 39, 5629 (the integration; not read); the equations as
//! summarized by V. Del Tatto, Master's thesis, University of Trento 2021 (arXiv:2111.06402,
//! eqs. 1.104-1.105 and 2.46).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use super::{scale_positions_and_cell, sinhc, Barostat, BarostatContext};
use crate::domain::{PeriodicCell, Units};
use crate::error::MoptResult;
use crate::math::linalg::symmetric_eigen;
use crate::md::thermostat::{positive, ThermostatChain};

/// Degrees of freedom of the symmetric velocity of the cell (for its chain).
const CELL_DOF: f64 = 6.0;

/// The eigenvalues and eigenvectors (columns) of a symmetric 3 x 3 matrix.
pub(super) fn eigen3(m: &[[f64; 3]; 3]) -> MoptResult<([f64; 3], [[f64; 3]; 3])> {
    let a = Mat::<f64>::from_fn(3, 3, |i, j| 0.5 * (m[i][j] + m[j][i]));
    let (values, vectors) = symmetric_eigen(&a)?;
    Ok((
        std::array::from_fn(|k| values[k]),
        std::array::from_fn(|i| std::array::from_fn(|k| vectors[(i, k)])),
    ))
}

/// `O^T r` for the eigenvectors `o` (columns).
fn into_eigen(o: &[[f64; 3]; 3], r: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|k| (0..3).map(|i| o[i][k] * r[i]).sum())
}

/// `O r'`.
fn from_eigen(o: &[[f64; 3]; 3], r: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| (0..3).map(|k| o[i][k] * r[k]).sum())
}

/// Fully flexible Martyna-Tobias-Klein barostat.
#[derive(Debug, Clone)]
pub struct MtkAniso {
    tau: f64,
    /// Velocity of the cell `p_g / W_g` (symmetric, inverse atomic time).
    velocity: [[f64; 3]; 3],
    /// `W_g` at the last action.
    mass: f64,
    /// `N_f` at the last action.
    dof: f64,
    /// Eigenvalues and eigenvectors of `velocity` (for the kicks and drifts).
    eigen: ([f64; 3], [[f64; 3]; 3]),
    chain: ThermostatChain,
}

impl MtkAniso {
    /// Period `tau` (atomic units) of the motion of the cell; `links` of its chain.
    pub fn new(tau: f64, links: usize) -> MoptResult<Self> {
        let tau = positive(tau, "the Martyna-Tobias-Klein period")?;
        Ok(Self {
            tau,
            velocity: [[0.0; 3]; 3],
            mass: 0.0,
            dof: 1.0,
            eigen: (
                [0.0; 3],
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            ),
            chain: ThermostatChain::new(links, tau, 3, 1)?,
        })
    }

    fn twice_kinetic(&self) -> f64 {
        self.mass * self.velocity.iter().flatten().map(|v| v * v).sum::<f64>()
    }

    fn scale(&mut self, s: f64) {
        for v in self.velocity.iter_mut().flatten() {
            *v *= s;
        }
    }

    fn trace(&self) -> f64 {
        self.velocity[0][0] + self.velocity[1][1] + self.velocity[2][2]
    }
}

impl Barostat for MtkAniso {
    fn name(&self) -> &str {
        "mtk_aniso"
    }

    fn apply(
        &mut self,
        dt: f64,
        _v: &mut Col<f64>,
        ctx: &mut BarostatContext<'_>,
    ) -> MoptResult<()> {
        let kt = ctx.kt();
        self.dof = ctx.dof.max(1.0);
        self.mass = (self.dof + 3.0) * kt * self.tau * self.tau / 3.0;
        let p = ctx.pressure_tensor();
        let k2 = ctx.kinetic[0][0] + ctx.kinetic[1][1] + ctx.kinetic[2][2];
        let force: [[f64; 3]; 3] = std::array::from_fn(|a| {
            std::array::from_fn(|b| {
                let symmetric = 0.5 * (p[a][b] + p[b][a]);
                let diagonal = if a == b {
                    k2 / self.dof - ctx.volume * ctx.pressure
                } else {
                    0.0
                };
                ctx.volume * symmetric + diagonal
            })
        });
        let w = self.mass;
        let s = self
            .chain
            .propagate(0.5 * dt, self.twice_kinetic(), CELL_DOF, kt);
        self.scale(s);
        for (row, f) in self.velocity.iter_mut().zip(&force) {
            for (v, x) in row.iter_mut().zip(f) {
                *v += dt * x / w;
            }
        }
        let s = self
            .chain
            .propagate(0.5 * dt, self.twice_kinetic(), CELL_DOF, kt);
        self.scale(s);
        self.eigen = eigen3(&self.velocity)?;
        Ok(())
    }

    fn kick(&self, h: f64, v: &mut Col<f64>, a: &Col<f64>) {
        let (lambda, o) = &self.eigen;
        let extra = self.trace() / self.dof;
        let factors: [(f64, f64); 3] = std::array::from_fn(|k| {
            let z = (lambda[k] + extra) * h;
            (libm::exp(-z), h * libm::exp(-0.5 * z) * sinhc(0.5 * z))
        });
        for i in 0..v.nrows() / 3 {
            let vi = into_eigen(o, [v[3 * i], v[3 * i + 1], v[3 * i + 2]]);
            let ai = into_eigen(o, [a[3 * i], a[3 * i + 1], a[3 * i + 2]]);
            let moved: [f64; 3] =
                std::array::from_fn(|k| vi[k] * factors[k].0 + ai[k] * factors[k].1);
            let back = from_eigen(o, moved);
            for c in 0..3 {
                v[3 * i + c] = back[c];
            }
        }
    }

    fn drift(
        &mut self,
        h: f64,
        x: &mut Col<f64>,
        v: &mut Col<f64>,
        cell: &mut PeriodicCell,
    ) -> MoptResult<()> {
        let (lambda, o) = self.eigen;
        let factors: [(f64, f64); 3] = std::array::from_fn(|k| {
            let z = lambda[k] * h;
            (libm::exp(z), h * libm::exp(0.5 * z) * sinhc(0.5 * z))
        });
        for i in 0..x.nrows() / 3 {
            let ri = into_eigen(&o, [x[3 * i], x[3 * i + 1], x[3 * i + 2]]);
            let vi = into_eigen(&o, [v[3 * i], v[3 * i + 1], v[3 * i + 2]]);
            let moved: [f64; 3] =
                std::array::from_fn(|k| ri[k] * factors[k].0 + vi[k] * factors[k].1);
            let back = from_eigen(&o, moved);
            for c in 0..3 {
                x[3 * i + c] = back[c];
            }
        }
        // The cell: O e^(Lambda h) O^T.
        let mu: [[f64; 3]; 3] = std::array::from_fn(|a| {
            std::array::from_fn(|b| (0..3).map(|k| o[a][k] * factors[k].0 * o[b][k]).sum())
        });
        let mut scratch = Col::zeros(0);
        scale_positions_and_cell(&mu, &mut scratch, cell)
    }

    fn energy(&self, volume: f64, pressure: f64, temperature: f64) -> f64 {
        let kt = Units::BOLTZMANN_HARTREE_PER_K * temperature;
        0.5 * self.twice_kinetic() + pressure * volume + self.chain.energy(CELL_DOF, kt)
    }

    fn has_conserved_quantity(&self) -> bool {
        true
    }

    fn notes(&self) -> Vec<String> {
        let (lambda, _) = self.eigen;
        vec![format!(
            "Martyna-Tobias-Klein (flexible cell): eigenvalues of the velocity of the cell \
             {:.6e} {:.6e} {:.6e} 1/a.u.",
            lambda[0], lambda[1], lambda[2]
        )]
    }

    fn reference(&self) -> Option<&'static str> {
        Some(
            "G. J. Martyna, D. J. Tobias, M. L. Klein, J. Chem. Phys. 1994, 101, 4177; M. E. \
             Tuckerman, J. Alejandre, R. Lopez-Rendon, A. L. Jochim, G. J. Martyna, J. Phys. A \
             2006, 39, 5629",
        )
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn the_eigenvectors_rebuild_the_matrix() {
        let m = [[0.3, 0.1, -0.2], [0.1, -0.4, 0.05], [-0.2, 0.05, 0.1]];
        let (l, o) = eigen3(&m).unwrap();
        for a in 0..3 {
            for b in 0..3 {
                let back: f64 = (0..3).map(|k| o[a][k] * l[k] * o[b][k]).sum();
                assert!((back - m[a][b]).abs() < 1e-12);
            }
        }
        let r = [1.0, -2.0, 0.5];
        let again = from_eigen(&o, into_eigen(&o, r));
        assert!((0..3).all(|c| (again[c] - r[c]).abs() < 1e-14));
    }
}
