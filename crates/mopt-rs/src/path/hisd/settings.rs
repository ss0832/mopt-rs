//! Settings of the searches for saddle points of index `k` ([`super::HighIndexSaddleSearch`]).
//!
//! The parts are named after the papers they come from:
//!
//! - the flow of the point: the reflected gradient of high-index saddle dynamics (HiSD; J. Yin,
//!   L. Zhang, P. Zhang, SIAM J. Sci. Comput. 2019, 41, A3576) or its improved form iHiSD,
//!   which starts as a gradient flow and turns into HiSD (H. Su, H. Wang, L. Zhang, J. Zhao,
//!   X. Zheng, SIAM 2025, DOI 10.1137/25M173212X; arXiv:2502.03694) ([`PointFlow`]);
//! - the update of the tracked directions ([`DirectionUpdate`]): the explicit Euler step of
//!   the direction equations of HiSD with Gram-Schmidt, the Oja flow of the subspace with a
//!   QR normalization (M. T. Chu, M. M. Lin, SIAM J. Numer. Anal. 2025, 63, 2343), one step
//!   of the locally optimal block preconditioned conjugate gradient method (LOBPCG; the
//!   alternative of Yin, Zhang, Zhang; A. V. Knyazev, SIAM J. Sci. Comput. 2001, 23, 517),
//!   the deflated directions of the generalized gentlest ascent dynamics (GGAD; W. Quapp,
//!   J. M. Bofill, Theor. Chem. Acc. 2014, 133, 1510), or the eigenvectors of an exact
//!   Hessian;
//! - the curvatures along the directions ([`Curvature`]): differences of gradients (the
//!   shrinking dimer of HiOSD, forward or central) or exact Hessians.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use crate::error::{MoptError, MoptResult};
use crate::opt::convergence::ConvergenceCriteria;

/// The flow of the point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PointFlow {
    /// HiSD: `dx/dt = -(I - 2 V V^T) g`, the gradient reflected across the tracked
    /// directions (ascent along them, descent along the rest).
    Reflected,
    /// iHiSD: `dx/dt = -(alpha (I - 2 V V^T) + (1 - alpha) s I) g` with `s = 1` downward
    /// (descent) and `s = -1` upward (ascent) at the start; `alpha` grows from `alpha0` as
    /// the logistic curve `d alpha / dt = 2 rate alpha (1 - alpha)` per iteration, the
    /// schedule of the examples of the paper.
    Improved {
        /// `alpha` at the start (the paper uses 1e-11 to 1e-7).
        alpha0: f64,
        /// `rate` of the logistic schedule, per iteration.
        rate: f64,
        /// Ascent instead of descent in the gradient-flow part (an upward search).
        upward: bool,
    },
}

/// How the tracked directions follow the lowest curvatures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirectionUpdate {
    /// HiSD (Yin, Zhang, Zhang 2019): `v_i <- v_i + gamma_i d_i` with
    /// `d_i = -(I - v_i v_i^T - 2 sum_(j<i) v_j v_j^T) H v_i`, then modified Gram-Schmidt in
    /// order; the directions become the individual eigenvectors, lowest first.
    HisdEuler,
    /// The Oja flow of the subspace, `V <- V - gamma (I - V V^T) H V`, then a QR
    /// normalization (Chu, Lin 2025: the first and third of their systems): only the
    /// subspace is tracked, not its basis, so near-degenerate curvatures inside it do not
    /// matter.
    OjaQr,
    /// One LOBPCG step: the Rayleigh-Ritz problem on `span[V, W, V_prev]` (`W` the
    /// residuals), as in section 7 of Yin, Zhang, Zhang; `k` more products per iteration
    /// for `W` and `k` for `V_prev`.
    Lobpcg,
    /// GGAD (Quapp, Bofill 2014, eqs. 16-21): each direction follows the Rayleigh quotient
    /// of the Hessian deflated by the earlier directions, `d_s = -(I - v_s v_s^T) D_s H D_s
    /// v_s` with `D_s = I - sum_(r<s) v_r v_r^T`, then a Schmidt orthogonalization.
    Ggad,
    /// The eigenvectors of the lowest eigenvalues of an exact Hessian (needs
    /// [`Curvature::Hessian`]; the choice of the algorithm of iHiSD).
    Eigenvectors,
}

impl DirectionUpdate {
    /// The update called `name` (`euler`, `oja`, `lobpcg`, `ggad`, `eigenvectors`).
    pub fn from_name(name: &str) -> MoptResult<Self> {
        Ok(match name.trim().to_ascii_lowercase().as_str() {
            "euler" | "hisd" | "euler_mgs" => Self::HisdEuler,
            "oja" | "oja_qr" | "qr" => Self::OjaQr,
            "lobpcg" => Self::Lobpcg,
            "ggad" => Self::Ggad,
            "eigenvectors" | "exact" => Self::Eigenvectors,
            other => {
                return Err(MoptError::parse(format!(
                    "direction update '{other}': euler, oja, lobpcg, ggad, or eigenvectors"
                )))
            }
        })
    }

    /// Name of the update.
    pub fn name(self) -> &'static str {
        match self {
            Self::HisdEuler => "euler",
            Self::OjaQr => "oja",
            Self::Lobpcg => "lobpcg",
            Self::Ggad => "ggad",
            Self::Eigenvectors => "eigenvectors",
        }
    }
}

/// Where the products of the Hessian with the directions come from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Curvature {
    /// `(g(x + l v) - g(x)) / l`: one gradient per direction besides the gradient at `x`.
    Forward,
    /// `(g(x + l v) - g(x - l v)) / 2 l`: two gradients per direction (the dimer of HiOSD).
    Central,
    /// The Hessian at `x` (one Hessian per iteration, no extra gradients).
    Hessian,
}

impl Curvature {
    /// The source called `name` (`forward`, `central`, `hessian`).
    pub fn from_name(name: &str) -> MoptResult<Self> {
        Ok(match name.trim().to_ascii_lowercase().as_str() {
            "forward" => Self::Forward,
            "central" | "dimer" => Self::Central,
            "hessian" | "exact" => Self::Hessian,
            other => {
                return Err(MoptError::parse(format!(
                    "curvature '{other}': forward, central, or hessian"
                )))
            }
        })
    }
}

/// The length of the step of the point: Barzilai-Borwein steps on the reflected gradient
/// (the steps of the optimization-based shrinking dimer methods; L. Zhang, Q. Du, Z. Zheng,
/// SIAM J. Sci. Comput. 2016, 38, A528, and HiOSD).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepSettings {
    /// Length of the first step (and of the step after a reset), in the units of the
    /// points (Bohr for structures): the largest displacement of an atom, or of the whole
    /// point for plain variables.
    pub first: f64,
    /// Largest displacement of an atom (or of the whole point) in one step.
    pub max_step: f64,
}

impl Default for StepSettings {
    fn default() -> Self {
        Self {
            first: 0.05,
            max_step: 0.2,
        }
    }
}

/// When the search hands its point over to a tighter method (RS-I-RFO at order `k`) before
/// it has converged on its own criteria.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HandOff {
    /// RMS of the gradient below this (Hartree/Bohr for structures).
    pub rms_gradient: f64,
    /// The subspace turned by less than this from the last iteration: the sine of the
    /// largest principal angle, `||P_n - P_(n-1)||_2`.
    pub projector_change: f64,
}

impl Default for HandOff {
    fn default() -> Self {
        Self {
            rms_gradient: 1e-3,
            projector_change: 0.1,
        }
    }
}

/// Settings of a search for a saddle point of index `order`.
#[derive(Debug, Clone, PartialEq)]
pub struct HisdSettings {
    /// The index `k` of the saddle point sought (1 for a transition state).
    pub order: usize,
    /// The flow of the point.
    pub flow: PointFlow,
    /// The update of the directions.
    pub update: DirectionUpdate,
    /// The products of the Hessian with the directions.
    pub curvature: Curvature,
    /// Track one direction more than `order` and reflect only `order` of them: its
    /// curvature tells whether the `(k + 1)`-th curvature is positive.
    pub guard: bool,
    /// GGAD: the first direction is taken along the gradient at the start (eq. 14 of Quapp
    /// and Bofill), replacing the first given direction.
    pub first_along_gradient: bool,
    /// Length of the displacement of the difference of gradients at the start (Bohr for
    /// structures: the largest displacement of an atom).
    pub dimer_length: f64,
    /// The displacement shrinks by `exp(-shrink)` per iteration (the shrinking dimer,
    /// `dl/dt = -l`) ...
    pub shrink: f64,
    /// ... down to this.
    pub min_dimer_length: f64,
    /// The steps of the point.
    pub step: StepSettings,
    /// Step of the Euler updates of the directions at the start, as a multiple of the
    /// inverse of the largest curvature magnitude along the tracked directions; later
    /// steps are Barzilai-Borwein steps.
    pub rotation_rate: f64,
    /// Largest change of a direction in one update (the length of `gamma d` before the
    /// normalization).
    pub max_rotation: f64,
    /// Updates of the directions per step of the point (each costs the products again).
    pub sweeps: usize,
    /// Iterations (steps of the point) at most.
    pub max_iterations: usize,
    /// The force thresholds of convergence (the displacement thresholds are not used).
    pub criteria: ConvergenceCriteria,
    /// An earlier end that hands the point over (`None`: the search runs to its criteria).
    pub hand_off: Option<HandOff>,
    /// After all tracked curvatures were negative once, the search ends when the highest
    /// of them stays non-negative for more than this many iterations (it left the region
    /// of index `k`).
    pub lost_after: usize,
    /// Seed of the directions that fill a guard or a lost direction.
    pub seed: u64,
}

impl HisdSettings {
    /// HiOSD (Yin, Zhang, Zhang 2019) for index `order`: the reflected gradient,
    /// Barzilai-Borwein steps, the Euler update of the directions with Gram-Schmidt, and a
    /// shrinking dimer of forward differences.
    pub fn hiosd(order: usize) -> Self {
        Self {
            order,
            flow: PointFlow::Reflected,
            update: DirectionUpdate::HisdEuler,
            curvature: Curvature::Forward,
            guard: false,
            first_along_gradient: false,
            dimer_length: 1e-3,
            shrink: 0.0,
            min_dimer_length: 1e-4,
            step: StepSettings::default(),
            rotation_rate: 0.5,
            max_rotation: 0.5,
            sweeps: 1,
            max_iterations: 1000,
            criteria: ConvergenceCriteria::NORMAL,
            hand_off: None,
            lost_after: 20,
            seed: 1,
        }
    }

    /// GGAD (Quapp, Bofill 2014) for index `order`: the same flow of the point with the
    /// deflated update of the directions, the first direction along the gradient at the
    /// start.
    pub fn ggad(order: usize) -> Self {
        Self {
            update: DirectionUpdate::Ggad,
            first_along_gradient: true,
            ..Self::hiosd(order)
        }
    }

    /// Number of tracked directions (`order`, plus one with the guard).
    pub fn tracked(&self) -> usize {
        self.order + usize::from(self.guard)
    }

    /// Check the settings.
    pub fn validate(&self) -> MoptResult<()> {
        let positive = [
            self.dimer_length,
            self.min_dimer_length,
            self.step.first,
            self.step.max_step,
            self.rotation_rate,
            self.max_rotation,
        ];
        if self.order == 0 {
            return Err(MoptError::config(
                "high-index saddle search: the order must be at least 1 (use a minimizer for 0)",
            ));
        }
        if positive.iter().any(|v| !(v.is_finite() && *v > 0.0))
            || !(self.shrink.is_finite() && self.shrink >= 0.0)
            || self.max_iterations == 0
            || self.sweeps == 0
        {
            return Err(MoptError::config(
                "high-index saddle search: lengths, steps, rates, iterations and sweeps must be \
                 positive (the shrink rate non-negative)",
            ));
        }
        if self.update == DirectionUpdate::Eigenvectors && self.curvature != Curvature::Hessian {
            return Err(MoptError::config(
                "high-index saddle search: the eigenvector update needs exact Hessians \
                 (curvature = hessian)",
            ));
        }
        if let PointFlow::Improved { alpha0, rate, .. } = self.flow {
            if !(alpha0 > 0.0 && alpha0 < 1.0 && rate.is_finite() && rate > 0.0) {
                return Err(MoptError::config(
                    "iHiSD: alpha0 must lie in (0, 1) and the rate be positive",
                ));
            }
        }
        if let Some(h) = self.hand_off {
            if !(h.rms_gradient > 0.0 && h.projector_change > 0.0) {
                return Err(MoptError::config(
                    "high-index saddle search: the hand-off thresholds must be positive",
                ));
            }
        }
        Ok(())
    }
}
