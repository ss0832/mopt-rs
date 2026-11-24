//! Metadynamics with the RMSD to a set of structures as its collective variables.
//!
//! `V(x) = sum_i k exp(-alpha Delta_i(x)^2)`, where `Delta_i` is the RMSD of the biased atoms
//! of `x` to reference structure `i` after the best superposition (Kabsch; rotations only).
//! It is the bias of the RMSD metadynamics of Grimme (J. Chem. Theory Comput. 2019, 15, 2847)
//! and of the iMTD-GC conformer search (Pracht, Bohle, Grimme, Phys. Chem. Chem. Phys. 2020,
//! 22, 7169), implemented from those descriptions: each Gaussian sits at a structure instead
//! of a value of a collective variable, so that the dynamics is pushed away from every
//! structure in the list. The references are given (the conformers known so far: a static
//! bias) and, with [`RmsdMetadynamics::depositing`], taken from the run itself at a fixed
//! interval of simulated time, as the Gaussians of [`super::metadynamics::Metadynamics`] are
//! deposited along their variables.
//!
//! The gradient is exact: at the optimal superposition the derivative of the rotation does
//! not enter, so `d Delta^2 / d x_a = (2 / n) (p_a - R^T y_a)` with the centred positions
//! `p_a` and `y_a` and the rotation `R` of `p` onto `y`. The Hessian is exact as well: the
//! optimal superposition is `max_R sum_a y_a . R p_a = lambda_max(N(S))`, the largest
//! eigenvalue of Horn's symmetric 4 x 4 quaternion matrix (J. Opt. Soc. Am. A 1987, 4, 629) of
//! the covariance `S = sum_a p_a y_a^T`, so `n Delta^2 = sum |p|^2 + sum |y|^2 - 2 lambda_max`.
//! `N` is linear in the positions, so second-order perturbation theory gives
//! `d2 lambda = 2 sum_{j > 0} (v_j^T dN v_0)^2 / (lambda_0 - lambda_j)` exactly. Where the
//! largest eigenvalue is degenerate the superposition is not unique and the RMSD has no
//! second derivative; the Hessian is then refused.
//!
//! MultiOptPy: no counterpart (new in mopt-rs); MultiOptPy's metadynamics
//!  (`Potential/gaussian_potential.py:GaussianPotential`) biases geometric variables only.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::sync::{Arc, Mutex};

use faer::{Col, Mat};

use crate::coord::alignment::proper_rotation_from_covariance;
use crate::domain::{AtomId, EvalOutput, MolecularSystem, Quantity};
use crate::error::{MoptError, MoptResult};
use crate::math::linalg::symmetric_eigen;
use crate::pes::term::{Frame, History, PesTerm, TermContext};

/// Relative gap below which the largest eigenvalue of Horn's matrix counts as degenerate
/// (the superposition is then not unique and the Hessian of the RMSD does not exist).
const DEGENERATE_GAP: f64 = 1e-12;

/// Horn's symmetric 4 x 4 matrix of the covariance `s = sum_a p_a y_a^T`: its largest
/// eigenvalue is `max_R sum_a y_a . R p_a` over proper rotations.
fn horn(s: &[[f64; 3]; 3]) -> Mat<f64> {
    let rows = crate::coord::alignment::horn_matrix(s);
    Mat::from_fn(4, 4, |i, j| rows[i][j])
}

#[derive(Debug, Default, Clone)]
struct References {
    /// Centred positions of the biased atoms of every reference: the given ones first.
    structures: Vec<Vec<[f64; 3]>>,
    /// How many of them were given (never dropped).
    given: usize,
    /// Simulated time of the next deposit.
    next: f64,
}

/// RMSD metadynamics bias.
#[derive(Debug)]
pub struct RmsdMetadynamics {
    /// Biased atoms (from 0); empty: all.
    atoms: Vec<usize>,
    /// Height `k` of every Gaussian (Hartree).
    height: f64,
    /// Exponent `alpha` (Bohr^-2).
    alpha: f64,
    /// Superpose by rotations (otherwise by the centroids only).
    rotation: bool,
    /// Interval of simulated time between deposits (atomic units); `None`: a static bias.
    every: Option<f64>,
    /// Largest number of deposited references (the oldest is dropped beyond it).
    max_deposits: usize,
    references: Mutex<References>,
}

fn centred(sys: &MolecularSystem, atoms: &[usize]) -> Vec<[f64; 3]> {
    let pts: Vec<[f64; 3]> = atoms
        .iter()
        .map(|&a| sys.position(AtomId::new(a)).value())
        .collect();
    let n = pts.len().max(1) as f64;
    let mut c = [0.0; 3];
    for p in &pts {
        for k in 0..3 {
            c[k] += p[k] / n;
        }
    }
    pts.iter()
        .map(|p| [p[0] - c[0], p[1] - c[1], p[2] - c[2]])
        .collect()
}

impl RmsdMetadynamics {
    /// A bias of height `height` (Hartree) and exponent `alpha` (Bohr^-2) on all atoms, with no
    /// references yet.
    pub fn new(height: f64, alpha: f64) -> MoptResult<Self> {
        if !(height.is_finite() && alpha.is_finite() && alpha > 0.0) {
            return Err(MoptError::config(format!(
                "RMSD metadynamics needs a finite height and a positive exponent, got k = \
                 {height}, alpha = {alpha}"
            )));
        }
        Ok(Self {
            atoms: Vec::new(),
            height,
            alpha,
            rotation: true,
            every: None,
            max_deposits: usize::MAX,
            references: Mutex::new(References::default()),
        })
    }

    /// Bias only these atoms (the RMSD is taken over them).
    #[must_use]
    pub fn on_atoms(mut self, atoms: &[AtomId]) -> Self {
        self.atoms = atoms.iter().map(|a| a.index()).collect();
        self.atoms.sort_unstable();
        self.atoms.dedup();
        self
    }

    /// Superpose the centroids only (structures held in space, such as by `-fix`).
    #[must_use]
    pub fn without_rotation(mut self) -> Self {
        self.rotation = false;
        self
    }

    /// Deposit the current structure as a new reference every `every` atomic time units of a
    /// dynamics (from its start on), keeping at most `max` deposits.
    pub fn depositing(mut self, every: f64, max: usize) -> MoptResult<Self> {
        if !(every.is_finite() && every > 0.0) || max == 0 {
            return Err(MoptError::config(
                "the deposits need a positive interval and room for at least one",
            ));
        }
        self.every = Some(every);
        self.max_deposits = max;
        Ok(self)
    }

    fn selected(&self, n: usize) -> Vec<usize> {
        if self.atoms.is_empty() {
            (0..n).collect()
        } else {
            self.atoms.iter().copied().filter(|&a| a < n).collect()
        }
    }

    /// Add `sys` as a given reference (kept for the whole run).
    pub fn add_reference(&self, sys: &MolecularSystem) -> MoptResult<()> {
        let atoms = self.selected(sys.n_atoms());
        let mut r = self
            .references
            .lock()
            .map_err(|_| MoptError::calc("the references of the bias are poisoned"))?;
        let at = r.given;
        r.structures.insert(at, centred(sys, &atoms));
        r.given += 1;
        Ok(())
    }

    /// Number of references (given and deposited).
    pub fn references(&self) -> usize {
        self.references.lock().map_or(0, |r| r.structures.len())
    }

    /// Height (Hartree) and exponent (Bohr^-2).
    pub fn parameters(&self) -> (f64, f64) {
        (self.height, self.alpha)
    }

    /// Energy and gradient over the biased atoms (`3 n` components).
    fn value(&self, p: &[[f64; 3]], refs: &[Vec<[f64; 3]>]) -> (f64, Vec<f64>) {
        let n = p.len();
        let mut energy = 0.0;
        let mut grad = vec![0.0; 3 * n];
        for y in refs.iter().filter(|y| y.len() == n) {
            let rot = if self.rotation {
                let mut cov = Mat::<f64>::zeros(3, 3);
                for (a, b) in p.iter().zip(y) {
                    for r in 0..3 {
                        for c in 0..3 {
                            cov[(r, c)] += a[r] * b[c];
                        }
                    }
                }
                proper_rotation_from_covariance(&cov)
            } else {
                Mat::identity(3, 3)
            };
            // R^T y_a and Delta^2.
            let mut back = vec![[0.0; 3]; n];
            let mut d2 = 0.0;
            for (k, b) in y.iter().enumerate() {
                for c in 0..3 {
                    back[k][c] = (0..3).map(|r| rot[(r, c)] * b[r]).sum::<f64>();
                    d2 += (p[k][c] - back[k][c]).powi(2);
                }
            }
            d2 /= n.max(1) as f64;
            let e = self.height * (-self.alpha * d2).exp();
            energy += e;
            let factor = -self.alpha * e * 2.0 / n.max(1) as f64;
            for k in 0..n {
                for c in 0..3 {
                    grad[3 * k + c] += factor * (p[k][c] - back[k][c]);
                }
            }
        }
        (energy, grad)
    }

    /// Hessian over the biased atoms (`3 n x 3 n`) at the centred positions `p`.
    fn hessian(&self, p: &[[f64; 3]], refs: &[Vec<[f64; 3]>]) -> MoptResult<Mat<f64>> {
        let n = p.len();
        let m = 3 * n;
        let nf = n.max(1) as f64;
        let mut h = Mat::<f64>::zeros(m, m);
        for y in refs.iter().filter(|y| y.len() == n) {
            // (n / 2) d2 Delta^2 = (delta_bc - 1/n) delta_de - 2 sum_j a_j a_j^T / gap_j.
            let mut hd = Mat::<f64>::from_fn(m, m, |i, j| {
                let same_axis = i % 3 == j % 3;
                if !same_axis {
                    0.0
                } else if i / 3 == j / 3 {
                    1.0 - 1.0 / nf
                } else {
                    -1.0 / nf
                }
            });
            let mut back = vec![[0.0; 3]; n];
            if self.rotation {
                let mut s = [[0.0; 3]; 3];
                for (a, b) in p.iter().zip(y) {
                    for (r, row) in s.iter_mut().enumerate() {
                        for (c, v) in row.iter_mut().enumerate() {
                            *v += a[r] * b[c];
                        }
                    }
                }
                let (w, v) = symmetric_eigen(&horn(&s))?;
                // Eigenvalues ascending: the last one is the largest.
                let top = (0..4).max_by(|&i, &j| w[i].total_cmp(&w[j])).unwrap_or(3);
                let v0: [f64; 4] = std::array::from_fn(|k| v[(k, top)]);
                // a_j at (atom b, axis d): v_j^T N(e_d y_b^T) v_0.
                let dn = |b: usize, d: usize| {
                    let mut e = [[0.0; 3]; 3];
                    e[d] = y[b];
                    horn(&e)
                };
                let derivs: Vec<Mat<f64>> = (0..m).map(|i| dn(i / 3, i % 3)).collect();
                let scale = w.iter().fold(1.0_f64, |a, x| a.max(x.abs()));
                for j in (0..4).filter(|&j| j != top) {
                    let gap = w[top] - w[j];
                    if gap <= DEGENERATE_GAP * scale {
                        return Err(MoptError::calc(
                            "RMSD metadynamics: the superposition onto a reference is not \
                             unique here (degenerate rotation), so the bias has no Hessian",
                        ));
                    }
                    let vj: [f64; 4] = std::array::from_fn(|k| v[(k, j)]);
                    let a: Vec<f64> = derivs
                        .iter()
                        .map(|d| {
                            (0..4)
                                .map(|r| vj[r] * (0..4).map(|c| d[(r, c)] * v0[c]).sum::<f64>())
                                .sum()
                        })
                        .collect();
                    for (i1, x) in a.iter().enumerate() {
                        for (i2, z) in a.iter().enumerate() {
                            hd[(i1, i2)] -= 2.0 * x * z / gap;
                        }
                    }
                }
                let rot = proper_rotation_from_covariance(&Mat::from_fn(3, 3, |r, c| s[r][c]));
                for (k, b) in y.iter().enumerate() {
                    for c in 0..3 {
                        back[k][c] = (0..3).map(|r| rot[(r, c)] * b[r]).sum::<f64>();
                    }
                }
            } else {
                back.clone_from_slice(y);
            }
            let mut d2 = 0.0;
            let mut gd = vec![0.0; m];
            for k in 0..n {
                for c in 0..3 {
                    let diff = p[k][c] - back[k][c];
                    d2 += diff * diff;
                    gd[3 * k + c] = 2.0 * diff / nf;
                }
            }
            d2 /= nf;
            let e = self.height * (-self.alpha * d2).exp();
            // d2 e = e (alpha^2 dD dD^T - alpha d2D), d2D = (2 / n) hd.
            for i in 0..m {
                for j in 0..m {
                    h[(i, j)] += e
                        * (self.alpha * self.alpha * gd[i] * gd[j]
                            - self.alpha * 2.0 / nf * hd[(i, j)]);
                }
            }
        }
        Ok(h)
    }

    fn scatter(dof: usize, atoms: &[usize], local: &[f64]) -> Col<f64> {
        let mut g = Col::zeros(dof);
        for (k, &a) in atoms.iter().enumerate() {
            for c in 0..3 {
                g[3 * a + c] = local[3 * k + c];
            }
        }
        g
    }
}

impl PesTerm for RmsdMetadynamics {
    fn name(&self) -> &str {
        "rmsd_metadynamics"
    }

    /// Invariant with the superposition by rotations; superposing the centroids only keeps
    /// the orientation, so a rotation changes the RMSD.
    fn frame(&self) -> Frame {
        if self.rotation {
            Frame::Invariant
        } else {
            Frame::TranslationOnly
        }
    }

    /// Stateful when it deposits references during a run; with given references only, a
    /// function of the structure (references added with [`RmsdMetadynamics::add_reference`]
    /// change it for every holder of the term).
    fn history(&self) -> History {
        if self.every.is_some() {
            History::Stateful
        } else {
            History::Static
        }
    }

    fn analytic_hessian(&self) -> bool {
        true
    }

    /// The same bias with a copy of the references so far.
    fn fork(&self) -> Option<Arc<dyn PesTerm>> {
        let references = self.references.lock().ok()?.clone();
        Some(Arc::new(Self {
            atoms: self.atoms.clone(),
            height: self.height,
            alpha: self.alpha,
            rotation: self.rotation,
            every: self.every,
            max_deposits: self.max_deposits,
            references: Mutex::new(references),
        }))
    }

    fn evaluate(
        &self,
        sys: &MolecularSystem,
        _ctx: &TermContext<'_>,
        need_hessian: bool,
    ) -> MoptResult<EvalOutput> {
        let atoms = self.selected(sys.n_atoms());
        let refs = self
            .references
            .lock()
            .map_err(|_| MoptError::calc("the references of the bias are poisoned"))?
            .structures
            .clone();
        let p = centred(sys, &atoms);
        let (energy, local) = self.value(&p, &refs);
        if !energy.is_finite() || local.iter().any(|g| !g.is_finite()) {
            return Err(MoptError::calc("the RMSD metadynamics bias is not finite"));
        }
        let dof = sys.dof();
        let hessian = if need_hessian {
            let local_h = self.hessian(&p, &refs)?;
            let mut h = Mat::<f64>::zeros(dof, dof);
            for (k, &a) in atoms.iter().enumerate() {
                for (l, &b) in atoms.iter().enumerate() {
                    for c in 0..3 {
                        for d in 0..3 {
                            h[(3 * a + c, 3 * b + d)] = local_h[(3 * k + c, 3 * l + d)];
                        }
                    }
                }
            }
            if (0..dof).any(|i| (0..dof).any(|j| !h[(i, j)].is_finite())) {
                return Err(MoptError::calc(
                    "the Hessian of the RMSD metadynamics bias is not finite",
                ));
            }
            Some(h)
        } else {
            None
        };
        Ok(EvalOutput {
            energy: Quantity::new(energy),
            gradient_bohr: Self::scatter(dof, &atoms, &local),
            hessian_bohr: hessian,
            orbitals: None,
            stress: None,
            strain_hessian: None,
        })
    }

    fn commit(&self, sys: &MolecularSystem, ctx: &TermContext<'_>) {
        let (Some(every), Some(time)) = (self.every, ctx.time_au) else {
            return;
        };
        let atoms = self.selected(sys.n_atoms());
        if let Ok(mut r) = self.references.lock() {
            if time + 1e-9 < r.next {
                return;
            }
            r.structures.push(centred(sys, &atoms));
            let deposits = r.structures.len() - r.given;
            if deposits > self.max_deposits {
                let first = r.given;
                r.structures.remove(first);
            }
            // The next deposit is one interval after this one.
            r.next = (time / every).floor() * every + every;
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::domain::{Position, SpinCharge};
    use crate::io::xyz::parse_xyz_str;

    fn water(shift: f64) -> MolecularSystem {
        parse_xyz_str(
            &format!("3\n\nO 0.0 0.0 0.0\nH 0.96 {shift} 0.0\nH -0.24 0.93 0.1\n"),
            SpinCharge::default(),
            None,
        )
        .unwrap()
    }

    #[test]
    fn the_bias_is_highest_at_a_reference_and_its_gradient_is_exact() {
        let bias = RmsdMetadynamics::new(0.01, 1.3).unwrap();
        bias.add_reference(&water(0.0)).unwrap();
        let ctx_sys = water(0.0);
        let ctx = TermContext::at_iteration(0, &ctx_sys);
        let at = bias.evaluate(&water(0.0), &ctx, false).unwrap();
        assert!((at.energy.value() - 0.01).abs() < 1e-12);
        let sys = water(0.3);
        let out = bias.evaluate(&sys, &ctx, true).unwrap();
        assert!(out.energy.value() < 0.01 && out.energy.value() > 0.0);
        let h = 1e-5;
        for k in 0..9 {
            let mut x = sys.coords_flat_bohr();
            x[k] += h;
            let ep = bias
                .evaluate(&sys.with_flat_coords_bohr(&x).unwrap(), &ctx, false)
                .unwrap();
            x[k] -= 2.0 * h;
            let em = bias
                .evaluate(&sys.with_flat_coords_bohr(&x).unwrap(), &ctx, false)
                .unwrap();
            let fd = (ep.energy.value() - em.energy.value()) / (2.0 * h);
            assert!(
                (fd - out.gradient_bohr[k]).abs() < 1e-8,
                "{k}: {fd} {}",
                out.gradient_bohr[k]
            );
        }
        let hess = out.hessian_bohr.unwrap();
        assert!((hess[(3, 4)] - hess[(4, 3)]).abs() < 1e-12);
        // A rigid motion changes nothing.
        let mut moved = sys.clone();
        for a in sys.atoms() {
            let p = sys.position(a).value();
            moved.set_position(a, Position::new([-p[1] + 2.0, p[0], p[2] - 1.0]));
        }
        let m = bias.evaluate(&moved, &ctx, false).unwrap();
        assert!((m.energy.value() - out.energy.value()).abs() < 1e-12);
    }

    /// Four atoms out of a plane.
    fn ammonia(shift: f64) -> MolecularSystem {
        parse_xyz_str(
            &format!(
                "4\n\nN 0.0 0.0 0.1\nH 0.94 {shift} -0.3\nH -0.47 0.81 -0.28\nH -0.45 -0.83 -0.33\n"
            ),
            SpinCharge::default(),
            None,
        )
        .unwrap()
    }

    #[test]
    fn the_hessian_is_the_derivative_of_the_exact_gradient() {
        for rotation in [true, false] {
            let mut bias = RmsdMetadynamics::new(0.01, 1.3).unwrap();
            if !rotation {
                bias = bias.without_rotation();
            }
            bias.add_reference(&ammonia(0.0)).unwrap();
            bias.add_reference(&water(0.2)).unwrap(); // other size: ignored
            let sys = ammonia(0.25);
            let ctx = TermContext::at_iteration(0, &sys);
            let hess = bias
                .evaluate(&sys, &ctx, true)
                .unwrap()
                .hessian_bohr
                .unwrap();
            let step = 1e-5;
            for k in 0..sys.dof() {
                let mut x = sys.coords_flat_bohr();
                x[k] += step;
                let gp = bias
                    .evaluate(&sys.with_flat_coords_bohr(&x).unwrap(), &ctx, false)
                    .unwrap()
                    .gradient_bohr;
                x[k] -= 2.0 * step;
                let gm = bias
                    .evaluate(&sys.with_flat_coords_bohr(&x).unwrap(), &ctx, false)
                    .unwrap()
                    .gradient_bohr;
                for l in 0..sys.dof() {
                    let fd = (gp[l] - gm[l]) / (2.0 * step);
                    assert!(
                        (fd - hess[(l, k)]).abs() < 1e-8,
                        "rotation {rotation}, ({l}, {k}): {fd} vs {}",
                        hess[(l, k)]
                    );
                    assert!((hess[(l, k)] - hess[(k, l)]).abs() < 1e-14);
                }
            }
        }
    }

    #[test]
    fn deposits_follow_the_simulated_time_and_keep_the_given_references() {
        let bias = RmsdMetadynamics::new(0.01, 1.0)
            .unwrap()
            .depositing(10.0, 2)
            .unwrap();
        bias.add_reference(&water(0.5)).unwrap();
        let sys = water(0.0);
        for (t, expected) in [
            (0.0, 2),
            (5.0, 2),
            (10.0, 3),
            (19.0, 3),
            (20.0, 3),
            (31.0, 3),
        ] {
            let ctx = TermContext::new(0, &sys, Some(t));
            bias.commit(&sys, &ctx);
            assert_eq!(bias.references(), expected, "t = {t}");
        }
        // Optimizations (no simulated time) deposit nothing.
        bias.commit(&sys, &TermContext::at_iteration(3, &sys));
        assert_eq!(bias.references(), 3);
        assert!(RmsdMetadynamics::new(0.01, 0.0).is_err());
    }
}
