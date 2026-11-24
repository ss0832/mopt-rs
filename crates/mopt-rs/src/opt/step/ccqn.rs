//! Cone-shaped constrained quasi-Newton (CCQN) transition-state search.
//!
//! Wu and Wang, "Cone-Shaped Constrained Quasi-Newton Method: Efficient and Robust Single-Ended
//! Transition State Optimization Algorithm", J. Chem. Theory Comput. 2025. Equation and
//! algorithm numbers refer to that paper.
//!
//! Each iteration looks at the (TS-BFGS-updated) Hessian `B`:
//! - `B` positive definite (potential well): the uphill step solves eqn (9),
//!   `min g^T s + s^T B s / 2` subject to `||s|| = Delta` and `||s|| cos(phi) <= e^T s`, i.e. the
//!   lowest point of the quadratic model on the spherical cap around the axis `e`, with the
//!   augmented Lagrangian method of Algorithm 1;
//! - `B` indefinite (saddle region): a saddle-point step, either the restricted-step P-RFO of the
//!   paper (section 2.3.2, eqn (19)) or RS-I-RFO.
//!
//! The axis `e` estimates the reaction coordinate: from reactive bonds (Algorithm 2), from the
//! midpoint of a path to the product (eqn (18)), or given directly. The paper works in Cartesian
//! coordinates; in other coordinate systems (DIC, RIC, ...) the Cartesian axis is mapped to the
//! optimizer's coordinates with `dq = B dx` and the subproblem is solved there.
//!
//! MultiOptPy: no counterpart (new in mopt-rs) (CCQN, Wu and Wang 2025, is not in MultiOptPy).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use super::cone::exact_cone_step;
use super::prfo::{PrfoTrust, RestrictedPrfo};
use super::rsirfo::{Rsirfo, RsirfoTrust};
use super::{StepGenerator, StepInput};
use crate::domain::{AtomId, MolecularSystem, Units};
use crate::error::{MoptError, MoptResult};
use crate::math::linalg::{col_dot, col_norm, symmetric_eigen, symmetrize};
use crate::math::projection::{project_out_basis_mat, project_out_basis_vec};

/// Modes with `|lambda|` below this are null modes (rigid-body motion) for the curvature test.
const NULL_MODE: f64 = 1e-6;

/// Estimate of the reaction coordinate.
#[derive(Debug, Clone)]
pub enum CcqnAxis {
    /// Stretch or compress the listed reactive bonds depending on whether the atoms attract or
    /// repel (Algorithm 2).
    ReactiveBonds(Vec<(AtomId, AtomId)>),
    /// Point toward the midpoint `x_mid` of a path from the current structure to the product
    /// (eqn (18)), recomputed from the current position at every step.
    Midpoint(Box<MolecularSystem>),
    /// A fixed Cartesian direction (`3N`).
    Direction(Col<f64>),
}

impl CcqnAxis {
    /// Midpoint of the straight line from `initial` to `product` (minimum-image for periodic
    /// cells). The paper takes the midpoint of an IDPP path; any interpolated midpoint can be
    /// passed with [`CcqnAxis::Midpoint`].
    pub fn linear_midpoint(
        initial: &MolecularSystem,
        product: &MolecularSystem,
    ) -> MoptResult<Self> {
        let d = product.displacement_from(initial)?;
        let mid = initial.with_flat_coords_bohr(&(initial.coords_flat_bohr() + d * 0.5))?;
        Ok(Self::Midpoint(Box::new(mid)))
    }

    /// Unit axis `e` at `sys`, where the forces are `-gradient` (Cartesian, Hartree/Bohr).
    pub fn direction(&self, sys: &MolecularSystem, gradient: &Col<f64>) -> MoptResult<Col<f64>> {
        self.direction_at(Some(sys), gradient)
    }

    /// Unit axis at `sys`; without a structure (a plain vector of variables) only a fixed
    /// [`CcqnAxis::Direction`] of the gradient's length is possible.
    pub fn direction_at(
        &self,
        sys: Option<&MolecularSystem>,
        gradient: &Col<f64>,
    ) -> MoptResult<Col<f64>> {
        let need = |what: &str| {
            MoptError::config(format!(
                "the CCQN axis '{what}' needs atoms; give a direction vector for plain variables"
            ))
        };
        let e = match self {
            Self::ReactiveBonds(bonds) => {
                reactive_bond_axis(sys.ok_or_else(|| need("reactive bonds"))?, gradient, bonds)?
            }
            Self::Midpoint(mid) => mid.displacement_from(sys.ok_or_else(|| need("midpoint"))?)?,
            Self::Direction(d) => {
                let n = sys.map_or(gradient.nrows(), MolecularSystem::dof);
                if d.nrows() != n {
                    return Err(MoptError::config(format!(
                        "CCQN axis has length {}, expected {n}",
                        d.nrows()
                    )));
                }
                d.clone()
            }
        };
        let n = col_norm(&e);
        if !(n.is_finite() && n > 0.0) {
            return Err(MoptError::config(
                "the CCQN axis vanishes at this structure",
            ));
        }
        Ok(e * (1.0 / n))
    }
}

/// Algorithm 2: for each reactive bond `(i, j)` with `v = R_j - R_i`,
/// `p = v (v . (F_j - F_i)) / |v|^2`, normalized; `E_i += p`, `E_j -= p`. In the attractive range
/// this stretches the bond and in the repulsive range it compresses it (Figure 2). A bond with
/// no force along it (`p = 0`) contributes nothing (not covered by the paper); if every bond
/// vanishes the bonds are stretched.
pub fn reactive_bond_axis(
    sys: &MolecularSystem,
    gradient: &Col<f64>,
    bonds: &[(AtomId, AtomId)],
) -> MoptResult<Col<f64>> {
    if bonds.is_empty() {
        return Err(MoptError::config("CCQN needs at least one reactive bond"));
    }
    let n = sys.n_atoms();
    let force = |a: AtomId| {
        [
            -gradient[a.dof(0)],
            -gradient[a.dof(1)],
            -gradient[a.dof(2)],
        ]
    };
    let accumulate = |stretch_only: bool| -> MoptResult<Col<f64>> {
        let mut e = Col::zeros(sys.dof());
        for &(i, j) in bonds {
            let (i, j) = (i.check(n)?, j.check(n)?);
            let v = sys.delta(j, i).value();
            let v2 = v.iter().map(|x| x * x).sum::<f64>();
            if v2 == 0.0 {
                return Err(MoptError::config("reactive bond between coincident atoms"));
            }
            let (fi, fj) = (force(i), force(j));
            let proj = if stretch_only {
                -1.0
            } else {
                (0..3).map(|c| v[c] * (fj[c] - fi[c])).sum::<f64>() / v2
            };
            let p: [f64; 3] = std::array::from_fn(|c| v[c] * proj);
            let pn = p.iter().map(|x| x * x).sum::<f64>().sqrt();
            if pn == 0.0 {
                continue;
            }
            for c in 0..3 {
                e[i.dof(c)] += p[c] / pn;
                e[j.dof(c)] -= p[c] / pn;
            }
        }
        Ok(e)
    };
    let e = accumulate(false)?;
    if col_norm(&e) > 0.0 {
        Ok(e)
    } else {
        accumulate(true)
    }
}

/// Parameters of Algorithm 1 (augmented Lagrangian method).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AlmSettings {
    /// Initial penalty `sigma_0 > 0`.
    pub sigma0: f64,
    /// Penalty growth `rho > 1`.
    pub rho: f64,
    /// Exponent `alpha` (`0 < alpha <= beta <= 1`).
    pub alpha: f64,
    /// Exponent `beta`.
    pub beta: f64,
    /// Final constraint tolerance `epsilon`, relative to `Delta`.
    pub epsilon: f64,
    /// Final stationarity tolerance `eta`, relative to `||g|| + Delta ||B||`.
    pub eta: f64,
    /// Outer iterations.
    pub max_outer: usize,
    /// Inner (L-BFGS) iterations per subproblem.
    pub max_inner: usize,
}

impl Default for AlmSettings {
    fn default() -> Self {
        Self {
            sigma0: 10.0,
            rho: 10.0,
            alpha: 0.1,
            beta: 0.9,
            epsilon: 1e-8,
            eta: 1e-8,
            max_outer: 100,
            max_inner: 500,
        }
    }
}

impl AlmSettings {
    fn validate(&self) -> MoptResult<()> {
        let ok = self.sigma0 > 0.0
            && self.rho > 1.0
            && 0.0 < self.alpha
            && self.alpha <= self.beta
            && self.beta <= 1.0
            && self.epsilon > 0.0
            && self.eta > 0.0
            && self.max_outer > 0
            && self.max_inner > 0;
        if ok {
            Ok(())
        } else {
            Err(MoptError::config(format!("invalid ALM settings: {self:?}")))
        }
    }
}

/// The cone-constrained subproblem of eqn (9) at one step.
struct ConeProblem<'a> {
    g: &'a Col<f64>,
    b: &'a Mat<f64>,
    e: &'a Col<f64>,
    radius: f64,
    cos_phi: f64,
}

impl ConeProblem<'_> {
    /// `(||s|| - Delta, ||s|| cos(phi) - e^T s)`.
    fn constraints(&self, s: &Col<f64>) -> (f64, f64) {
        let n = col_norm(s);
        (n - self.radius, n * self.cos_phi - col_dot(self.e, s))
    }

    /// Augmented Lagrangian `L(s; lambda, mu, sigma)` and its gradient.
    fn lagrangian(&self, s: &Col<f64>, lambda: f64, mu: f64, sigma: f64) -> (f64, Col<f64>) {
        let bs = self.b * s;
        let f = col_dot(self.g, s) + 0.5 * col_dot(s, &bs);
        let mut grad = self.g + &bs;
        let n = col_norm(s);
        let unit = if n > 0.0 {
            s * (1.0 / n)
        } else {
            self.e.clone()
        };
        let (h, c) = self.constraints(s);
        let shifted = (mu / sigma + c).max(0.0);
        let value = f
            + lambda * h
            + 0.5 * sigma * h * h
            + 0.5 * sigma * (shifted * shifted - (mu / sigma).powi(2));
        grad += &unit * (lambda + sigma * h);
        grad += (&unit * self.cos_phi - self.e) * (sigma * shifted);
        (value, grad)
    }

    /// Approximate minimizer of `L` from `s0` by L-BFGS with Armijo backtracking.
    fn minimize(
        &self,
        s0: Col<f64>,
        lambda: f64,
        mu: f64,
        sigma: f64,
        tol: f64,
        max_iter: usize,
    ) -> Col<f64> {
        const MEMORY: usize = 10;
        let mut s = s0;
        let (mut val, mut grad) = self.lagrangian(&s, lambda, mu, sigma);
        let mut pairs: Vec<(Col<f64>, Col<f64>)> = Vec::new();
        for _ in 0..max_iter {
            if col_norm(&grad) <= tol {
                break;
            }
            // Two-loop recursion.
            let mut q = grad.clone();
            let mut alphas = Vec::with_capacity(pairs.len());
            for (ds, dy) in pairs.iter().rev() {
                let a = col_dot(ds, &q) / col_dot(dy, ds);
                q -= dy * a;
                alphas.push(a);
            }
            if let Some((ds, dy)) = pairs.last() {
                q *= col_dot(ds, dy) / col_dot(dy, dy);
            } else {
                q *= self.radius / col_norm(&grad).max(1e-300);
            }
            for ((ds, dy), a) in pairs.iter().zip(alphas.into_iter().rev()) {
                let b = col_dot(dy, &q) / col_dot(dy, ds);
                q += ds * (a - b);
            }
            let mut dir = -q;
            let mut slope = col_dot(&grad, &dir);
            if slope >= 0.0 {
                dir = -&grad;
                slope = -col_dot(&grad, &grad);
                pairs.clear();
            }
            let mut t = 1.0;
            let mut accepted = None;
            for _ in 0..60 {
                let trial = &s + &dir * t;
                let (v, g) = self.lagrangian(&trial, lambda, mu, sigma);
                if v.is_finite() && v <= val + 1e-4 * t * slope {
                    accepted = Some((trial, v, g));
                    break;
                }
                t *= 0.5;
            }
            let Some((s_new, v_new, g_new)) = accepted else {
                break;
            };
            let ds = &s_new - &s;
            let dy = &g_new - &grad;
            if col_dot(&ds, &dy) > 1e-16 * col_norm(&ds) * col_norm(&dy) {
                if pairs.len() == MEMORY {
                    pairs.remove(0);
                }
                pairs.push((ds, dy));
            }
            s = s_new;
            val = v_new;
            grad = g_new;
        }
        s
    }
}

/// Uphill step of eqn (9) solved by Algorithm 1: the minimizer of `g^T s + s^T B s / 2` on the
/// sphere `||s|| = radius` inside the cone of half-apex angle `phi` around the unit axis `e`.
pub fn cone_step(
    g: &Col<f64>,
    b: &Mat<f64>,
    e: &Col<f64>,
    radius: f64,
    cos_phi: f64,
    alm: &AlmSettings,
) -> MoptResult<Col<f64>> {
    alm.validate()?;
    if !(radius > 0.0 && cos_phi > 0.0 && cos_phi < 1.0) {
        return Err(MoptError::config(format!(
            "CCQN needs a positive radius and 0 < cos(phi) < 1, got {radius} and {cos_phi}"
        )));
    }
    let p = ConeProblem {
        g,
        b,
        e,
        radius,
        cos_phi,
    };
    // Algorithm 1 in the units of the problem: constraint violations relative to Delta,
    // stationarity relative to the size of the model gradient on the sphere, and the penalty
    // `sigma` relative to their ratio.
    let b_norm = (0..b.nrows())
        .map(|i| (0..b.ncols()).map(|j| b[(i, j)].abs()).sum::<f64>())
        .fold(0.0_f64, f64::max);
    let eps_unit = radius;
    let eta_unit = col_norm(g) + radius * b_norm + f64::MIN_POSITIVE;
    let sigma_unit = eta_unit / eps_unit;
    let (eps_final, eta_final) = (alm.epsilon * eps_unit, alm.eta * eta_unit);
    let (mut lambda, mut mu) = (0.0, 0.0);
    // Dimensionless penalty factor of the paper.
    let mut sig = alm.sigma0;
    let mut eta_t = eta_unit / sig;
    let mut eps_t = eps_unit * sig.powf(-alm.alpha);
    // The axis itself is feasible.
    let mut s = e * radius;
    for _ in 0..alm.max_outer {
        let sigma = sig * sigma_unit;
        s = p.minimize(s, lambda, mu, sigma, eta_t, alm.max_inner);
        let (h, c) = p.constraints(&s);
        let violation = (h * h + c.max(-mu / sigma).powi(2)).sqrt();
        if violation <= eps_t {
            let stationarity = col_norm(&p.lagrangian(&s, lambda, mu, sigma).1);
            if violation <= eps_final && stationarity <= eta_final {
                break;
            }
            lambda += sigma * h;
            mu = (mu + sigma * c).max(0.0);
            eta_t = (eta_t / sig).max(eta_final);
            eps_t = (eps_t / sig.powf(alm.beta)).max(eps_final);
        } else {
            if sig >= 1e12 {
                break;
            }
            sig *= alm.rho;
            eta_t = (eta_t / sig).max(eta_final);
            eps_t = (eps_t / sig.powf(alm.alpha)).max(eps_final);
        }
    }
    // A local solution suffices (section 2.3.1). Remove the remaining constraint violation:
    // project onto the cap (rotate onto the cone surface if outside) and onto the sphere.
    let n = col_norm(&s);
    if !(n.is_finite() && n > 0.0) {
        return Ok(e * radius);
    }
    let along = col_dot(e, &s);
    if along < n * cos_phi {
        let perp = &s - e * along;
        let pn = col_norm(&perp);
        if pn == 0.0 {
            return Ok(e * radius);
        }
        let sin_phi = (1.0 - cos_phi * cos_phi).sqrt();
        return Ok((e * cos_phi + perp * (sin_phi / pn)) * radius);
    }
    Ok(s * (radius / n))
}

/// Solver of the uphill subproblem (eqn (9)).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConeSolver {
    /// Global solution through the secular equation ([`super::cone::exact_cone_step`]).
    #[default]
    Exact,
    /// The paper's augmented Lagrangian method (Algorithm 1; a local solution).
    Alm,
}

/// Saddle-region step method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SaddleStep {
    /// Restricted-step P-RFO with the trust policy of eqn (19) (the paper's choice).
    #[default]
    RsPrfo,
    /// RS-I-RFO (MultiOptPy's saddle optimizer).
    RsIrfo,
}

/// CCQN settings.
#[derive(Debug, Clone)]
pub struct CcqnSettings {
    /// Reaction-coordinate estimate.
    pub axis: CcqnAxis,
    /// `cos(phi)` of the cone half-apex angle (small for reactive bonds, large for a midpoint).
    pub cos_phi: f64,
    /// Step length `Delta` of the uphill phase (Bohr, fixed).
    pub uphill_radius: f64,
    /// Saddle-region step method.
    pub saddle_step: SaddleStep,
    /// Trust region of the saddle phase: radii for both step methods, and the eqn (19) policy
    /// for RS-P-RFO (RS-I-RFO keeps its own ratio policy within these radii).
    pub trust: PrfoTrust,
    /// Solver of the uphill subproblem.
    pub cone_solver: ConeSolver,
    /// Algorithm 1 parameters (for [`ConeSolver::Alm`]).
    pub alm: AlmSettings,
    /// Initial Hessian `c I` (Hartree/Bohr^2).
    pub initial_hessian: f64,
}

impl CcqnSettings {
    /// Settings of the paper's benchmarks (Supporting Information, section 1): `cos(phi) = 0.1`
    /// for reactive bonds and fixed directions, `0.5` for a midpoint; step sizes
    /// `0.1 <= Delta <= 0.2` Angstrom (uphill steps of 0.1 Angstrom; the saddle-phase radius
    /// starts at 0.1 Angstrom); initial Hessian `70 I` eV/Angstrom^2. The eqn (19) ratio
    /// thresholds and factors are not given there; Sella's values are used.
    pub fn new(axis: CcqnAxis) -> Self {
        let cos_phi = if matches!(axis, CcqnAxis::Midpoint(_)) {
            0.5
        } else {
            0.1
        };
        let a = Units::ANGSTROM_TO_BOHR;
        Self {
            axis,
            cos_phi,
            uphill_radius: 0.1 * a,
            saddle_step: SaddleStep::default(),
            trust: PrfoTrust {
                initial: 0.1 * a,
                min: 0.1 * a,
                max: 0.2 * a,
                ..PrfoTrust::default()
            },
            cone_solver: ConeSolver::default(),
            alm: AlmSettings::default(),
            initial_hessian: 70.0 * Units::EV_TO_HARTREE * Units::BOHR_TO_ANGSTROM.powi(2),
        }
    }
}

/// Phase of the last CCQN step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CcqnPhase {
    /// Cone-constrained uphill step (potential well).
    Uphill,
    /// Saddle-point step (saddle region).
    Saddle,
}

/// CCQN step generator.
#[derive(Debug, Clone)]
pub struct Ccqn {
    settings: CcqnSettings,
    prfo: RestrictedPrfo,
    rsirfo: Rsirfo,
    phase: Option<CcqnPhase>,
}

impl Ccqn {
    /// Generator for a first-order saddle search; `rsirfo_trust` configures the RS-I-RFO phase
    /// (its radii are replaced by those of `settings.trust`).
    pub fn new(settings: CcqnSettings, rsirfo_trust: RsirfoTrust) -> MoptResult<Self> {
        let rsirfo_trust = RsirfoTrust {
            initial: settings.trust.initial,
            min: settings.trust.min,
            max: settings.trust.max,
            ..rsirfo_trust
        };
        if !(settings.initial_hessian.is_finite() && settings.initial_hessian > 0.0) {
            return Err(MoptError::config(
                "CCQN needs a positive initial Hessian scale",
            ));
        }
        settings.alm.validate()?;
        if !(settings.uphill_radius > 0.0 && settings.cos_phi > 0.0 && settings.cos_phi < 1.0) {
            return Err(MoptError::config(
                "CCQN needs a positive uphill radius and 0 < cos(phi) < 1",
            ));
        }
        Ok(Self {
            prfo: RestrictedPrfo::new(settings.trust)?,
            rsirfo: Rsirfo::new(rsirfo_trust),
            settings,
            phase: None,
        })
    }

    /// Phase of the last step.
    pub fn phase(&self) -> Option<CcqnPhase> {
        self.phase
    }

    /// Settings.
    pub fn settings(&self) -> &CcqnSettings {
        &self.settings
    }
}

impl StepGenerator for Ccqn {
    fn name(&self) -> &str {
        match self.settings.saddle_step {
            SaddleStep::RsPrfo => "ccqn",
            SaddleStep::RsIrfo => "ccqn_rsirfo",
        }
    }

    fn uses_hessian(&self) -> bool {
        true
    }

    fn learns_negative_curvature(&self) -> bool {
        true
    }

    fn initial_hessian_scale(&self) -> Option<f64> {
        Some(self.settings.initial_hessian)
    }

    fn controls_step_length(&self) -> bool {
        true
    }

    fn propose(&mut self, input: &StepInput<'_>) -> MoptResult<Col<f64>> {
        if input.saddle_order == 0 {
            return Err(MoptError::config(
                "CCQN searches saddle points (saddle order >= 1)",
            ));
        }
        let sys = input.system;
        let gradient = match input.rigid_body_basis {
            Some(d) => project_out_basis_vec(d, input.gradient),
            None => input.gradient.clone(),
        };
        let mut h = match input.bias_hessian {
            Some(b) => input.hessian + b,
            None => input.hessian.clone(),
        };
        if let Some(d) = input.rigid_body_basis {
            h = project_out_basis_mat(d, &h);
        }
        symmetrize(&mut h);
        let (vals, _) = symmetric_eigen(&h)?;
        let indefinite = vals.iter().any(|&v| v <= -NULL_MODE);

        if !indefinite {
            let mut e = if input.cartesian_coordinates {
                self.settings.axis.direction_at(sys, input.gradient)?
            } else {
                let (Some(frame), Some(g_x)) = (input.frame, input.cartesian_gradient) else {
                    return Err(MoptError::config(
                        "CCQN in internal coordinates needs the coordinate frame",
                    ));
                };
                let e_q = frame.displacement_to_q(&self.settings.axis.direction_at(sys, g_x)?);
                let n = col_norm(&e_q);
                if !(n.is_finite() && n > 0.0) {
                    return Err(MoptError::config(
                        "the CCQN axis has no component in the optimizer's coordinates",
                    ));
                }
                e_q * (1.0 / n)
            };
            if let Some(d) = input.rigid_body_basis {
                e = project_out_basis_vec(d, &e);
                let n = col_norm(&e);
                if n == 0.0 {
                    return Err(MoptError::config("the CCQN axis is a rigid-body motion"));
                }
                e *= 1.0 / n;
            }
            self.phase = Some(CcqnPhase::Uphill);
            self.prfo.forget_last_step();
            self.rsirfo.forget_last_step();
            let (r, cphi) = (self.settings.uphill_radius, self.settings.cos_phi);
            return match self.settings.cone_solver {
                ConeSolver::Exact => {
                    exact_cone_step(&gradient, &h, &e, r, cphi, input.rigid_body_basis)
                }
                ConeSolver::Alm => cone_step(&gradient, &h, &e, r, cphi, &self.settings.alm),
            };
        }

        let entering = self.phase != Some(CcqnPhase::Saddle);
        self.phase = Some(CcqnPhase::Saddle);
        match self.settings.saddle_step {
            SaddleStep::RsPrfo => {
                if entering {
                    self.prfo.forget_last_step();
                }
                self.prfo.propose(input)
            }
            SaddleStep::RsIrfo => {
                if entering {
                    self.rsirfo.forget_last_step();
                }
                self.rsirfo.propose(input)
            }
        }
    }

    fn reset(&mut self) {
        self.prfo.reset();
        self.rsirfo.reset();
        self.phase = None;
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::domain::{positions, Bohr, Element, SpinCharge};

    #[test]
    fn cone_step_satisfies_the_constraints() {
        let g = Col::from_fn(3, |i| [0.05, -0.02, 0.01][i]);
        let b = Mat::from_fn(3, 3, |i, j| if i == j { [0.3, 1.0, 2.0][i] } else { 0.0 });
        // Uphill along +x against the gradient.
        let e = Col::from_fn(3, |i| [1.0, 0.0, 0.0][i]);
        let s = cone_step(&g, &b, &e, 0.1, 0.7, &AlmSettings::default()).unwrap();
        assert!((col_norm(&s) - 0.1).abs() < 1e-10);
        assert!(col_dot(&e, &s) >= 0.1 * 0.7 - 1e-7);
        // The model favors leaving the axis, so the solution sits on the cone surface.
        assert!((col_dot(&e, &s) - 0.07).abs() < 1e-5, "{s:?}");
    }

    #[test]
    fn cone_step_is_the_constrained_minimum() {
        // Compare with a dense scan of the cap in 2D: once with the optimum on the cone surface,
        // once inside the cap.
        let b = Mat::from_fn(2, 2, |i, j| [[0.5, 0.1], [0.1, 1.5]][i][j]);
        let e = Col::from_fn(2, |i| [1.0, 0.0][i]);
        let (r, cphi) = (0.2, 0.6);
        for gv in [[0.03, 0.04], [-0.01, 0.002]] {
            let g = Col::from_fn(2, |i| gv[i]);
            let s = cone_step(&g, &b, &e, r, cphi, &AlmSettings::default()).unwrap();
            let x = exact_cone_step(&g, &b, &e, r, cphi, None).unwrap();
            assert!((&s - &x).norm_max() < 1e-6, "ALM {s:?} vs exact {x:?}");
            let f = |v: &Col<f64>| col_dot(&g, v) + 0.5 * col_dot(v, &(&b * v));
            let phi = cphi.acos();
            let (best_t, best) = (0..=200_000)
                .map(|k| -phi + 2.0 * phi * k as f64 / 200_000.0)
                .map(|t| (t, f(&Col::from_fn(2, |i| r * [t.cos(), t.sin()][i]))))
                .fold((0.0, f64::INFINITY), |a, b| if b.1 < a.1 { b } else { a });
            assert!((f(&s) - best).abs() < 1e-9, "{gv:?}: {} vs {best}", f(&s));
            let t = s[1].atan2(s[0]);
            assert!((t - best_t).abs() < 1e-4, "{gv:?}: angle {t} vs {best_t}");
        }
    }

    #[test]
    fn reactive_bond_axis_stretches_attractive_and_compresses_repulsive_bonds() {
        let sys = MolecularSystem::new(
            vec![Element::H, Element::H],
            &positions::<Bohr>(&[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]),
            SpinCharge::default(),
            None,
        )
        .unwrap();
        let bond = [(AtomId::new(0), AtomId::new(1))];
        // Attractive: atom 1 pulled toward atom 0 (force -x), i.e. gradient +x on atom 1.
        let g = Col::from_fn(6, |i| [-0.1, 0.0, 0.0, 0.1, 0.0, 0.0][i]);
        let e = reactive_bond_axis(&sys, &g, &bond).unwrap();
        assert!(e[0] < 0.0 && e[3] > 0.0, "stretch: {e:?}");
        let g = -g;
        let e = reactive_bond_axis(&sys, &g, &bond).unwrap();
        assert!(e[0] > 0.0 && e[3] < 0.0, "compress: {e:?}");
    }
}
