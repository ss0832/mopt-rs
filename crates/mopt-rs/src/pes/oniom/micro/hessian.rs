//! The Hessian of the relaxed surface of the core, `H_xx - H_xy H_yy^-1 H_yx`, by one of
//! three routes ([`HessianRoutes`], chosen when the surface is set up):
//!
//! - Every level has an analytic Hessian and the matrices fit in the memory a matrix may take
//!   ([`crate::core_infra::memory`]; [`whole_entries`]): the analytic Hessian of the whole
//!   system at the relaxed environment, reduced to the core with the Cholesky factor of
//!   `H_yy` (positive definite at a relaxed environment; the pseudo-inverse where it is not).
//! - Mechanical embedding with an analytic Hessian for every term of the model systems: those
//!   terms depend on the core alone (the outermost region and the outer atoms of its links),
//!   so `H_xx` is their analytic Hessian on the core plus central differences of the gradient
//!   of the lowest level, which also give the coupling `H_yx` (the model terms have none).
//!   The higher levels are computed once per Hessian instead of at `2n` displaced cores.
//! - Otherwise central differences of the ONIOM gradient give `H_xx` and `H_yx`.
//!
//! On the routes with differences the response of the environment is taken off with
//! `H_yy^-1` applied by preconditioned conjugate gradients on Hessian-vector products by
//! differences of the lowest level. A failed stability solve on a responding block no larger
//! than the core can use central differences and a direct factorization. The memory is
//! that of a few `m x n` blocks ([`difference_entries`]). A Hessian that does not fit there
//! either is refused with what it would take.
//!
//! With a shell (`-oniom_opt micro_hessian_shell=R`) only the atoms of the environment within
//! `R` of an atom of the core respond, `H_xx - H_xS H_SS^-1 H_Sx`, and the rest is held: the
//! blocks of the analytic route, the rows of the coupling, and the conjugate gradients are
//! those of the shell `S` alone. The coupling of an atom of the environment to the core falls
//! off with its distance, and so does what it adds to the Hessian of the core.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::linalg::solvers::Solve;
use faer::{Col, Mat};

use super::{
    assemble, gather, CoreHessian, LowEnvironment, MicroOniomSettings, RelaxedEnvironment,
};
use crate::core_infra::memory;
use crate::domain::{EvalRequest, SpinCharge};
use crate::error::{MoptError, MoptResult};
use crate::math::linalg::{col_dot, col_norm, symmetric_eigen};
use crate::pes::calculator::Calculator;
use crate::pes::microiter::{InnerCoordinates, MicroProblem};
use crate::pes::oniom::ModelPart;

/// Displacement of the central differences of the gradient of the environment (Bohr; the
/// Hessian-vector products along unit vectors and the columns of the dense block). Central
/// differences err by `h^2 f''' / 6` and by the noise of the gradient over `2h`: at 1e-3 Bohr
/// both are near 1e-6 Hartree/Bohr^2 for a force field in double precision (gradients to
/// about 1e-9), well below the soft curvatures of an environment (1e-4 and up).
const PRODUCT_STEP: f64 = 1e-3;

/// The response comes from the dense block of the responding environment when its `2 m`
/// gradients cost no more than this share of the products the conjugate gradients may take
/// (`n` solves of at most [`SOLVE_PRODUCTS`] products of two gradients each).
const DENSE_SHARE: f64 = 1.0;

/// How the response of the environment in the Hessian of the core is solved
/// (`-oniom_opt micro_hessian_solve`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ResponseSolve {
    /// The dense block of the responding environment where it costs fewer gradients than the
    /// conjugate gradients may take (and fits in the memory), else conjugate gradients.
    #[default]
    Auto,
    /// Always the dense block (refused where it does not fit in the memory).
    Dense,
    /// Always conjugate gradients on Hessian-vector products.
    ConjugateGradients,
}

/// Relative residual of the solves with `H_yy` (the Hessian is a model for the
/// quasi-Newton steps of the core; its finite differences are not more accurate).
const SOLVE_TOLERANCE: f64 = 1e-3;

/// Products of one solve at most.
const SOLVE_PRODUCTS: usize = 100;

/// A solve stops when its residual grows to this many times the lowest one it had: the
/// products are then dominated by the noise of the differences (an engine whose gradient has
/// few digits, such as OpenMM's CPU platform in single precision), and further iterations
/// only amplify it.
const RUNAWAY: f64 = 10.0;

/// How far above the largest curvature of the core block the response of the environment may
/// reach (Hartree/Bohr^2): `H_xx - H_xy H_yy^-1 H_yx` is the Schur complement of the Hessian of
/// the whole system, whose lowest eigenvalue bounds its own from below, and no stationary
/// point of a molecular surface has a curvature below minus this.
const RESPONSE_MARGIN: f64 = 1.0;

/// The entries of the dense matrices of the analytic route for a whole system of `whole`
/// coordinates and an environment of `m`: the Hessian of the whole system three times while
/// the terms are summed (the sum, the Hessian of a term, and its copy with the sign of the
/// term), then the block of the environment and its Cholesky factor.
pub fn whole_entries(whole: usize, m: usize) -> f64 {
    let (whole, m) = (whole as f64, m as f64);
    3.0 * whole * whole + 2.0 * m * m
}

/// The entries of the dense matrices of the routes with differences for a core of `n`
/// coordinates and an environment of `m`: the coupling `H_yx`, the solutions and right-hand
/// sides kept for the Galerkin starts, and the response of the environment (`m x n` each),
/// with three blocks of the core.
pub fn difference_entries(n: usize, m: usize) -> f64 {
    let (n, m) = (n as f64, m as f64);
    4.0 * m * n + 3.0 * n * n
}

/// The entries of the reduction of a given Hessian of the whole system (which is there
/// already): the block of the environment and its factor, the coupling and the response, and
/// two blocks of the core.
fn given_entries(n: usize, m: usize) -> f64 {
    let (n, m) = (n as f64, m as f64);
    2.0 * m * m + 2.0 * m * n + 2.0 * n * n
}

/// The coordinates of the environment that respond in the Hessian of the core (indices into
/// the flat coordinates of the environment, three per atom, ascending): all of them, or with
/// `shell` (Bohr) those of the atoms within that distance of an atom of the core at `full`
/// (minimum images in a cell).
pub(super) fn responding_rows(
    full: &crate::domain::MolecularSystem,
    split: &super::CoreSplit,
    shell: Option<f64>,
) -> Vec<usize> {
    let all = 3 * split.environment.len();
    let Some(radius) = shell else {
        return (0..all).collect();
    };
    let cell = full.active_cell();
    let core: Vec<[f64; 3]> = split
        .core
        .iter()
        .map(|&a| full.position(crate::domain::AtomId::new(a)).value())
        .collect();
    let mut out = Vec::new();
    for (k, &a) in split.environment.iter().enumerate() {
        let p = full.position(crate::domain::AtomId::new(a)).value();
        let near = core.iter().any(|c| {
            let d = match cell {
                Some(cell) => cell.mic_delta(p, *c),
                None => std::array::from_fn(|i| p[i] - c[i]),
            };
            d.iter().map(|v| v * v).sum::<f64>() <= radius * radius
        });
        if near {
            out.extend(3 * k..3 * k + 3);
        }
    }
    out
}

/// `v` (one entry per row of `rows`) in the `m` coordinates of the environment, zero
/// elsewhere.
fn embed(v: &Col<f64>, rows: &[usize], m: usize) -> Col<f64> {
    let mut out = Col::<f64>::zeros(m);
    for (i, &r) in rows.iter().enumerate() {
        out[r] = v[i];
    }
    out
}

/// The entries of `v` (all coordinates of the environment) on `rows`.
fn restrict(v: &Col<f64>, rows: &[usize]) -> Col<f64> {
    Col::from_fn(rows.len(), |i| v[rows[i]])
}

/// How the Hessian of the core is made, chosen when the surface is set up.
#[derive(Clone)]
pub(super) struct HessianRoutes {
    /// Every level has an analytic Hessian and the analytic route fits in the memory.
    pub whole: bool,
    /// The terms of the model systems, where every one has an analytic Hessian and the
    /// embedding is mechanical (the split route).
    pub model: Option<ModelPart>,
    /// The memory a Hessian may take (bytes) and how it was had.
    pub budget: (f64, memory::BudgetSource),
    /// What the run log says about the choice (once, when the surface is set up).
    pub notes: Vec<String>,
}

impl HessianRoutes {
    /// The routes of the surface `oniom` (with `model` its terms of the model systems) on a
    /// whole system of `whole` coordinates, a core of `n`, and an environment of `m` of which
    /// `s` respond (the shell at the start).
    pub(super) fn choose(
        oniom: &dyn Calculator,
        model: Option<ModelPart>,
        whole: usize,
        (n, m, s): (usize, usize, usize),
        settings: &MicroOniomSettings,
    ) -> Self {
        let budget = memory::budget_bytes(settings.hessian_memory);
        let analytic = oniom.analytic_hessian();
        let need = memory::f64_bytes(whole_entries(whole, s));
        let fits = need <= budget.0;
        let model = model.filter(|p| !settings.embedded && p.analytic_hessian());
        let mut notes = Vec::new();
        if let Some(radius) = settings.hessian_shell {
            notes.push(format!(
                "ONIOM microiterations: in the Hessian of the core only the environment within \
                 {:.2} Angstrom of the core responds ({} of its {} coordinates at the start); \
                 the rest is held.",
                radius * crate::domain::Units::BOHR_TO_ANGSTROM,
                s,
                m
            ));
        }
        let instead = |model: bool| {
            if model {
                format!(
                    "the analytic Hessians of the model systems on the core and central \
                     differences of the gradient of the lowest level ({} of its gradients per \
                     Hessian)",
                    2 * n
                )
            } else {
                format!(
                    "central differences of the ONIOM gradient ({} gradients per Hessian)",
                    2 * n
                )
            }
        };
        if analytic && !fits && settings.core_hessian != CoreHessian::Approximate {
            notes.push(format!(
                "ONIOM microiterations: the analytic Hessian of the whole system ({whole} \
                 coordinates, {m} of them relaxed) would take about {:.2} GiB, more than the \
                 {} a Hessian may take (-oniom_opt micro_hessian_memory=GiB); the Hessian of \
                 the core comes from {} instead.",
                need / memory::GIB,
                memory::describe(budget.0, budget.1),
                instead(model.is_some())
            ));
        } else if !analytic && model.is_some() {
            notes.push(format!(
                "ONIOM microiterations: the Hessian of the core from {}.",
                instead(true)
            ));
        }
        Self {
            whole: analytic && fits,
            model,
            budget,
            notes,
        }
    }

    /// An error where `entries` entries of dense matrices do not fit in the budget.
    fn check(&self, entries: f64, what: &str) -> MoptResult<()> {
        let need = memory::f64_bytes(entries);
        if need <= self.budget.0 {
            return Ok(());
        }
        Err(MoptError::unsupported(format!(
            "ONIOM microiterations: {what} would take about {:.2} GiB, more than the {} a \
             Hessian may take; give more (-oniom_opt micro_hessian_memory=GiB), a smaller core, \
             or the core block alone (-oniom_opt micro_hessian=held)",
            need / memory::GIB,
            memory::describe(self.budget.0, self.budget.1)
        )))
    }
}

/// `H_yy^-1 H_yx`: by the Cholesky factor of `hyy` (positive definite at a relaxed
/// environment: about a third of the work of an eigendecomposition, and stable), by its
/// pseudo-inverse where the factorization fails (an environment not at a minimum, or one
/// with a direction of no curvature).
fn environment_response(hyy: &Mat<f64>, hyx: &Mat<f64>) -> MoptResult<Mat<f64>> {
    if let Some(out) = positive_environment_response(hyy, hyx) {
        return Ok(out);
    }
    let inverse = crate::math::linalg::pseudo_inverse(hyy)?;
    Ok(&inverse * hyx)
}

/// Solve the environment response by Cholesky factorization on a positive definite block.
fn positive_environment_response(hyy: &Mat<f64>, hyx: &Mat<f64>) -> Option<Mat<f64>> {
    let factor = hyy.llt(faer::Side::Lower).ok()?;
    let response = factor.solve(hyx.as_ref());
    let finite = response.col_iter().all(|c| c.iter().all(|v| v.is_finite()));
    finite.then_some(response)
}

/// `H_SS v` on the responding coordinates `rows` of the environment `y` by central
/// differences of the gradient along `v` (one entry per row; two gradients). The lowest
/// level gives no Hessian-vector products of its own (a force field of another program), so
/// they are differences; central ones, since forward differences along a step of 1e-4 Bohr
/// were off by 0.15 percent on chorismate mutase (docs/source/oniom.md).
fn product(
    problem: &LowEnvironment<'_>,
    y: &[f64],
    v: &Col<f64>,
    rows: &[usize],
) -> MoptResult<Col<f64>> {
    let s = v.nrows();
    let norm = col_norm(v);
    if norm < 1e-300 {
        return Ok(Col::zeros(s));
    }
    let mut plus = y.to_vec();
    let mut minus = y.to_vec();
    for (i, &r) in rows.iter().enumerate() {
        plus[r] += PRODUCT_STEP * v[i] / norm;
        minus[r] -= PRODUCT_STEP * v[i] / norm;
    }
    let gp = problem.evaluate(&plus, false)?.gradient;
    let gm = problem.evaluate(&minus, false)?.gradient;
    Ok(Col::from_fn(s, |i| {
        (gp[rows[i]] - gm[rows[i]]) * norm / (2.0 * PRODUCT_STEP)
    }))
}

/// How the block `H_SS` of the responding environment is had on the dense route.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BlockSource {
    /// The analytic Hessian of the lowest level on the whole system.
    Analytic,
    /// Central differences of the gradient of the lowest level, one pair per coordinate.
    Differences,
}

/// The response of the environment on the dense route, kept for the correction of the
/// gradient of a relaxation that stopped short of a minimum ([`RelaxedEnvironment`]): with
/// `H_SS = V diag(w) V^T` over its positive curvatures and `Z = H_SS^-1 H_Sx`, the residual
/// `g_S` of the environment moves the minimum by `-H_SS^-1 g_S`, the gradient of the core by
/// `-Z^T g_S`, and the energy by `-g_S^T H_SS^-1 g_S / 2` (to first and second order).
#[derive(Debug, Clone)]
pub(super) struct DenseResponse {
    /// The responding coordinates of the environment (indices into its flat coordinates).
    pub rows: Vec<usize>,
    /// The positive eigenvalues of `H_SS` kept for its inverse.
    values: Vec<f64>,
    /// Their eigenvectors (`m x k`).
    vectors: Mat<f64>,
    /// `Z = H_SS^-1 H_Sx` (`m x n`).
    z: Mat<f64>,
    /// The lowest curvature of the block and its unit mode (one entry per responding
    /// coordinate): where it is negative, the environment stands on a saddle point of its own.
    lowest: (f64, Col<f64>),
}

/// The environment of a Hessian of the core stands on a saddle point of its own: the block of
/// its responding coordinates has a curvature below minus the threshold of the mode kill of
/// the environment ([`crate::pes::microiter::ModeKill::threshold`]).
#[derive(Debug, Clone)]
pub(super) struct EnvironmentSaddle {
    /// The lowest curvature of the block (Hartree/Bohr^2).
    pub curvature: f64,
    /// Its unit mode in the coordinates of the whole environment (zero outside the
    /// responding ones).
    pub mode: Col<f64>,
}

/// What a Hessian of the core gives the evaluation: the Hessian (`None`: a check of stability
/// that cannot be judged) and, where the environment it was taken at stands on a saddle point
/// of its own, that saddle (the evaluation then leaves it and takes the Hessian again).
pub(super) struct CoreHessianOut {
    /// The Hessian of the core.
    pub hessian: Option<Mat<f64>>,
    /// The saddle point of the environment, if it stands on one.
    pub saddle: Option<EnvironmentSaddle>,
}

impl DenseResponse {
    /// The response of the block `hss` of the responding coordinates `rows` (symmetrized)
    /// coupled to the core by `hsx`: `hss` decomposed and inverted over its positive
    /// curvatures (relative to the largest, [`POSITIVE_SHARE`]), `Z = H_SS^+ H_Sx`, and the
    /// number of curvatures left out with the lowest one. `None` where the decomposition
    /// fails.
    pub(super) fn from_block(
        rows: &[usize],
        hss: &Mat<f64>,
        hsx: &Mat<f64>,
    ) -> Option<(Self, usize, f64)> {
        let m = rows.len();
        let n = hsx.ncols();
        let (w, v) = symmetric_eigen(&symmetric(hss)).ok()?;
        let top = w.iter().fold(0.0_f64, |a, x| a.max(*x));
        let lowest = if m > 0 { w[0] } else { 0.0 };
        let lowest_mode = Col::from_fn(m, |i| v[(i, 0)]);
        let keep: Vec<usize> = (0..m).filter(|&k| w[k] > POSITIVE_SHARE * top).collect();
        let vectors = Mat::from_fn(m, keep.len(), |i, c| v[(i, keep[c])]);
        let values: Vec<f64> = keep.iter().map(|&k| w[k]).collect();
        // Z = V diag(1/w) V^T H_Sx.
        let vt_b = vectors.transpose() * hsx;
        let scaled = Mat::from_fn(keep.len(), n, |c, k| vt_b[(c, k)] / values[c]);
        let z = &vectors * &scaled;
        Some((
            Self {
                rows: rows.to_vec(),
                values,
                vectors,
                z,
                lowest: (lowest, lowest_mode),
            },
            m - keep.len(),
            lowest,
        ))
    }

    /// The step of the environment (all `all` of its coordinates, zero outside the responding
    /// ones) to the minimum of the quadratic model of this block from an environment whose
    /// gradient is `g_y`: `-H_SS^-1 g_S`, the move of [`Self::correction`]. `None` where it is
    /// beyond [`CORRECTION_REACH`] (or the gradient does not fit).
    pub(super) fn minimum_step(&self, g_y: &Col<f64>, all: usize) -> Option<Col<f64>> {
        if self.rows.iter().any(|&r| r >= g_y.nrows() || r >= all) {
            return None;
        }
        let h = self.inverse_times(&restrict(g_y, &self.rows));
        let reach = crate::math::linalg::col_max_abs(&h);
        (reach.is_finite() && reach <= CORRECTION_REACH).then(|| embed(&(-&h), &self.rows, all))
    }

    /// The saddle point of the environment this block says it stands on: its lowest curvature
    /// where that is below `-threshold`, with the mode in the `all` coordinates of the
    /// environment.
    pub(super) fn saddle(&self, threshold: f64, all: usize) -> Option<EnvironmentSaddle> {
        let (curvature, mode) = &self.lowest;
        (*curvature < -threshold && self.rows.iter().all(|&r| r < all)).then(|| EnvironmentSaddle {
            curvature: *curvature,
            mode: embed(mode, &self.rows, all),
        })
    }

    /// `H_SS^-1 g` (one entry per responding coordinate).
    fn inverse_times(&self, g: &Col<f64>) -> Col<f64> {
        let m = self.vectors.nrows();
        let mut out = Col::<f64>::zeros(m);
        for (k, &w) in self.values.iter().enumerate() {
            let c: f64 = (0..m).map(|i| self.vectors[(i, k)] * g[i]).sum::<f64>() / w;
            for i in 0..m {
                out[i] += c * self.vectors[(i, k)];
            }
        }
        out
    }

    /// The first-order correction of a relaxation whose environment gradient is `g_y` (all
    /// coordinates of the environment): the change of the core gradient, `-Z^T g_S`, and of
    /// the energy, `-g_S^T H_SS^-1 g_S / 2`. `Err` with the predicted move of the environment
    /// (largest component, Bohr) where it is beyond [`CORRECTION_REACH`]: the quadratic model
    /// of the environment does not reach that far, and the correction is not trusted.
    pub(super) fn correction(&self, g_y: &Col<f64>) -> Result<(Col<f64>, f64), f64> {
        if self.rows.iter().any(|&r| r >= g_y.nrows()) {
            return Err(f64::NAN);
        }
        let g = restrict(g_y, &self.rows);
        let h = self.inverse_times(&g);
        let reach = crate::math::linalg::col_max_abs(&h);
        if reach.is_nan() || reach > CORRECTION_REACH {
            return Err(reach);
        }
        let energy = -0.5 * col_dot(&g, &h);
        let n = self.z.ncols();
        let m = self.z.nrows();
        let gradient = Col::from_fn(n, |k| -(0..m).map(|i| self.z[(i, k)] * g[i]).sum::<f64>());
        if energy.is_finite() && gradient.iter().all(|v| v.is_finite()) {
            Ok((gradient, energy))
        } else {
            Err(reach)
        }
    }
}

/// The largest move of a coordinate of the environment (Bohr) that the correction of the
/// gradient of a relaxation trusts its quadratic model for: the response of the newest
/// Hessian of the core is that of the environment it was computed at, and its curvatures
/// hold over tenths of a Bohr (bonds stretch by about that much at the thermal energy).
pub(super) const CORRECTION_REACH: f64 = 0.05;

impl DenseResponse {
    /// The number of responding coordinates.
    pub(super) fn size(&self) -> usize {
        self.rows.len()
    }
}

/// A direct stability check for a small responding environment. Central differences give
/// a common symmetric matrix for all right-hand sides, including soft rotations for which
/// direction-dependent forward differences can stall conjugate gradients.
fn direct_stability_response(
    problem: &LowEnvironment<'_>,
    env: &[f64],
    rows: &[usize],
    coupling: &Mat<f64>,
) -> MoptResult<Option<Mat<f64>>> {
    let m = rows.len();
    let mut h = Mat::<f64>::zeros(m, m);
    for (j, &row) in rows.iter().enumerate() {
        let mut moved = env.to_vec();
        moved[row] += PRODUCT_STEP;
        let plus = problem.evaluate(&moved, false)?.gradient;
        moved[row] = env[row] - PRODUCT_STEP;
        let minus = problem.evaluate(&moved, false)?.gradient;
        for (i, &r) in rows.iter().enumerate() {
            h[(i, j)] = (plus[r] - minus[r]) / (2.0 * PRODUCT_STEP);
        }
    }
    let h = Mat::from_fn(m, m, |i, j| 0.5 * (h[(i, j)] + h[(j, i)]));
    let Some(response) = positive_environment_response(&h, coupling) else {
        return Ok(None);
    };
    let residual = &h * &response - coupling;
    let accurate = residual
        .col_iter()
        .zip(coupling.col_iter())
        .all(|(r, b)| r.norm_l2() <= SOLVE_TOLERANCE * b.norm_l2().max(1e-14));
    Ok(accurate.then_some(response))
}

/// The solutions of earlier right-hand sides, the start of the next solve: with `Z` the
/// solutions and `B = H Z` their right-hand sides, the Galerkin start of `H z = b` in the
/// span of `Z` is `Z (Z^T B)^-1 Z^T b` (the columns of the effective Hessian are solves with
/// one matrix and related right-hand sides, so the later ones start close).
#[derive(Default)]
struct Solved {
    pairs: Vec<(Col<f64>, Col<f64>)>,
}

impl Solved {
    fn start(&self, b: &Col<f64>) -> MoptResult<Option<Col<f64>>> {
        let p = self.pairs.len();
        if p == 0 {
            return Ok(None);
        }
        let mut zb = Mat::<f64>::zeros(p, p);
        for i in 0..p {
            for j in 0..p {
                zb[(i, j)] = 0.5
                    * (col_dot(&self.pairs[i].0, &self.pairs[j].1)
                        + col_dot(&self.pairs[j].0, &self.pairs[i].1));
            }
        }
        let rhs: Vec<f64> = self.pairs.iter().map(|(z, _)| col_dot(z, b)).collect();
        let (w, v) = symmetric_eigen(&zb)?;
        let wmax = w.iter().fold(0.0_f64, |a, x| a.max(x.abs()));
        let mut c = vec![0.0; p];
        for k in (0..p).filter(|&k| w[k] > 1e-10 * wmax) {
            let a: f64 = (0..p).map(|i| v[(i, k)] * rhs[i]).sum::<f64>() / w[k];
            for (i, ci) in c.iter_mut().enumerate() {
                *ci += a * v[(i, k)];
            }
        }
        let m = b.nrows();
        Ok(Some(Col::from_fn(m, |j| {
            self.pairs
                .iter()
                .zip(&c)
                .map(|((z, _), ci)| ci * z[j])
                .sum()
        })))
    }
}

/// How a solve with `H_SS` ended.
#[derive(Debug, Clone, Copy, Default)]
struct SolveEnd {
    /// Hessian-vector products of the environment it took (the Galerkin start included).
    products: usize,
    /// It stopped at a direction without positive curvature (`p.H p <= 0`).
    not_positive: bool,
    /// It stopped at the limit of products.
    at_limit: bool,
    /// Residual relative to the right-hand side at the end.
    residual: f64,
    /// Lowest curvature `p.H p / p.p` met along its directions (Hartree/Bohr^2).
    lowest: f64,
    /// It stopped because its residual ran away (see [`RUNAWAY`]) or the preconditioner was
    /// not positive along the residual.
    ran_away: bool,
}

/// What the solves of one Hessian of the core did, for the log.
#[derive(Debug, Clone, Copy)]
struct SolveTally {
    solves: usize,
    products: usize,
    not_positive: usize,
    at_limit: usize,
    ran_away: usize,
    worst_residual: f64,
    lowest: f64,
    largest_response: f64,
}

impl Default for SolveTally {
    fn default() -> Self {
        Self {
            solves: 0,
            products: 0,
            not_positive: 0,
            at_limit: 0,
            ran_away: 0,
            worst_residual: 0.0,
            lowest: f64::INFINITY,
            largest_response: 0.0,
        }
    }
}

impl SolveTally {
    fn add(&mut self, end: &SolveEnd, response: f64) {
        self.solves += 1;
        self.products += end.products;
        self.not_positive += usize::from(end.not_positive);
        self.at_limit += usize::from(end.at_limit);
        self.ran_away += usize::from(end.ran_away);
        self.worst_residual = self.worst_residual.max(end.residual);
        self.lowest = self.lowest.min(end.lowest);
        self.largest_response = self.largest_response.max(response);
    }
}

/// The lowest and the highest eigenvalue of a symmetric matrix (`None` where it has none).
fn extreme_curvatures(h: &Mat<f64>) -> Option<(f64, f64)> {
    let sym = Mat::from_fn(h.nrows(), h.ncols(), |i, j| 0.5 * (h[(i, j)] + h[(j, i)]));
    let (w, _) = symmetric_eigen(&sym).ok()?;
    let n = w.nrows();
    (n > 0).then(|| (w[0], w[n - 1]))
}

/// The lowest wavenumber (cm^-1, negative: imaginary) of a Hessian of the core with the
/// masses `masses` (amu, one per coordinate), the rigid motions not taken out (the core sits
/// in its environment).
fn lowest_wavenumber(h: &Mat<f64>, masses: &[f64]) -> Option<f64> {
    let n = h.nrows();
    if masses.len() != n || n == 0 {
        return None;
    }
    let weighted = Mat::from_fn(n, n, |i, j| {
        0.5 * (h[(i, j)] + h[(j, i)]) / libm::sqrt(masses[i] * masses[j])
    });
    let (w, _) = symmetric_eigen(&weighted).ok()?;
    Some(crate::irc::modekill::eigenvalue_to_wavenumber(w[0]))
}

/// `H_SS^-1 b` on the responding coordinates `rows` of the environment `y` of `problem`
/// (gradient `g0` there; all of them without a shell): preconditioned conjugate gradients on
/// Hessian-vector products from forward differences of the gradient,
/// with `precondition` (the inverse of the limited-memory BFGS of the relaxations) as the
/// preconditioner, from the Galerkin start of `solved`, to a residual of [`SOLVE_TOLERANCE`]
/// of `b` (at most [`SOLVE_PRODUCTS`] products). The environment is at a minimum, so `H_yy`
/// is positive definite there.
fn solve_environment(
    problem: &LowEnvironment<'_>,
    y: &[f64],
    b: &Col<f64>,
    rows: &[usize],
    precondition: &dyn Fn(&Col<f64>) -> Col<f64>,
    solved: &Solved,
    max_products: usize,
) -> MoptResult<(Col<f64>, SolveEnd)> {
    let m = b.nrows();
    let b_norm = col_norm(b);
    let mut end = SolveEnd {
        lowest: f64::INFINITY,
        ..SolveEnd::default()
    };
    if b_norm < 1e-14 {
        return Ok((Col::zeros(m), end));
    }
    // The Galerkin start from the earlier solutions, where it is closer than zero.
    let (mut z, mut r) = match solved.start(b)? {
        Some(z0) => {
            let hz = product(problem, y, &z0, rows)?;
            end.products += 1;
            let r0 = b - &hz;
            if col_norm(&r0) < b_norm {
                (z0, r0)
            } else {
                (Col::<f64>::zeros(m), b.clone())
            }
        }
        None => (Col::<f64>::zeros(m), b.clone()),
    };
    end.residual = col_norm(&r) / b_norm;
    if end.residual <= SOLVE_TOLERANCE {
        return Ok((z, end));
    }
    // The solution of the lowest residual so far: with noisy products the residual of
    // conjugate gradients does not fall monotonically, and the last iterate may be the worst.
    let mut best = (end.residual, z.clone());
    let mut s = precondition(&r);
    let mut p = s.clone();
    let mut rs = col_dot(&r, &s);
    if !(rs > 0.0 && rs.is_finite()) {
        end.ran_away = true;
        return Ok((z, end));
    }
    end.at_limit = true;
    for _ in 0..max_products {
        let q = product(problem, y, &p, rows)?;
        end.products += 1;
        let pq = col_dot(&p, &q);
        let pp = col_dot(&p, &p);
        if pp > 0.0 {
            end.lowest = end.lowest.min(pq / pp);
        }
        if pq <= 0.0 {
            // Not positive along p (noise of the differences): stop with what was found.
            end.not_positive = true;
            end.at_limit = false;
            break;
        }
        let alpha = rs / pq;
        z = Col::from_fn(m, |i| z[i] + alpha * p[i]);
        r = Col::from_fn(m, |i| r[i] - alpha * q[i]);
        let residual = col_norm(&r) / b_norm;
        if residual < best.0 {
            best = (residual, z.clone());
        }
        if residual <= SOLVE_TOLERANCE {
            end.at_limit = false;
            break;
        }
        if !residual.is_finite() || residual > RUNAWAY * best.0 {
            end.ran_away = true;
            end.at_limit = false;
            break;
        }
        s = precondition(&r);
        let rs_next = col_dot(&r, &s);
        if !(rs_next > 0.0 && rs_next.is_finite()) {
            end.ran_away = true;
            end.at_limit = false;
            break;
        }
        let beta = rs_next / rs;
        rs = rs_next;
        p = Col::from_fn(m, |i| s[i] + beta * p[i]);
    }
    end.residual = best.0;
    Ok((best.1, end))
}

/// The response of the environment in the span of `solutions` (the solves of
/// `H_SS z_k = b_k`, the columns `b_k` of `hyx`): with `V` an orthonormal basis of the
/// solutions, `T = V^T H_SS V` from one Hessian-vector product per column of `V`, and
/// `B = V^T H_Sx`, the coupling `C = B^T T^+ B` (`T^+` over the positive curvatures of `T`) and
/// the solutions `z_k = V T^+ B e_k` that go with it. `C` is positive semidefinite and below
/// the exact `H_xS H_SS^-1 H_Sx` in the Loewner order for any `V` (the Galerkin approximation
/// of a Schur complement), and equal to it where the solves converged: a solve that stopped
/// early leaves part of the response out but puts no curvature into the core that it does not
/// have. `None` where the solutions span nothing. The products are counted in `tally`.
#[allow(clippy::too_many_arguments)]
fn galerkin_response(
    problem: &LowEnvironment<'_>,
    y: &[f64],
    rows: &[usize],
    hyx: &Mat<f64>,
    solutions: &[Col<f64>],
    tally: &mut SolveTally,
) -> MoptResult<Option<(Mat<f64>, Vec<Col<f64>>)>> {
    let m = rows.len();
    let n = hyx.ncols();
    let basis = crate::math::linalg::orthonormalize_columns(m, solutions, 1e-8);
    let r = basis.ncols();
    if r == 0 {
        return Ok(None);
    }
    let mut hv = Mat::<f64>::zeros(m, r);
    for c in 0..r {
        let v = Col::from_fn(m, |i| basis[(i, c)]);
        let q = product(problem, y, &v, rows)?;
        tally.products += 1;
        for i in 0..m {
            hv[(i, c)] = q[i];
        }
    }
    let t = basis.transpose() * &hv;
    let t = Mat::from_fn(r, r, |i, j| 0.5 * (t[(i, j)] + t[(j, i)]));
    let (w, u) = symmetric_eigen(&t)?;
    let top = w.iter().fold(0.0_f64, |a, v| a.max(*v));
    if top.is_nan() || top <= 0.0 {
        return Ok(None);
    }
    // T^+ over the positive curvatures (a direction without one carries no response).
    let keep: Vec<usize> = (0..r).filter(|&k| w[k] > 1e-10 * top).collect();
    let b = basis.transpose() * hyx;
    // W = T^+ B, column by column.
    let mut wb = Mat::<f64>::zeros(r, n);
    for &k in &keep {
        for j in 0..n {
            let coefficient: f64 = (0..r).map(|i| u[(i, k)] * b[(i, j)]).sum::<f64>() / w[k];
            for i in 0..r {
                wb[(i, j)] += coefficient * u[(i, k)];
            }
        }
    }
    let coupling = b.transpose() * &wb;
    let coupling = Mat::from_fn(n, n, |i, j| 0.5 * (coupling[(i, j)] + coupling[(j, i)]));
    let z = (0..n)
        .map(|k| {
            let col = Col::from_fn(r, |i| wb[(i, k)]);
            &basis * &col
        })
        .collect();
    Ok(Some((coupling, z)))
}

/// Whether `coupling = H_xy H_yy^-1 H_yx` (the response of the environment, `n x n`) is one
/// that a positive definite `H_yy` can give next to the core block with the largest curvature
/// `block_highest`: positive semidefinite within rounding and the noise of the differences,
/// and not above `block_highest + RESPONSE_MARGIN` (the Schur complement of the Hessian of the
/// whole system has no curvature far below that of a stationary point). `Err` says why not.
fn plausible_response(coupling: &Mat<f64>, block_highest: f64) -> Result<(), String> {
    let Some((lowest, highest)) = extreme_curvatures(coupling) else {
        return Err("its eigenvalues could not be had".to_string());
    };
    if !(lowest.is_finite() && highest.is_finite()) {
        return Err("it is not finite".to_string());
    }
    let scale = highest.abs().max(block_highest.abs()).max(1e-12);
    if lowest < -1e-2 * scale {
        return Err(format!(
            "it is not positive semidefinite (lowest eigenvalue {lowest:+.3e} Hartree/Bohr^2)"
        ));
    }
    if highest > block_highest + RESPONSE_MARGIN {
        return Err(format!(
            "its largest eigenvalue {highest:+.3e} Hartree/Bohr^2 is beyond the largest \
             curvature of the core block ({block_highest:+.3e}) by more than {RESPONSE_MARGIN}"
        ));
    }
    Ok(())
}

impl RelaxedEnvironment {
    /// The Hessian of the core from a given matrix: one of the core as it is, one of the
    /// whole system reduced to the core with the response of the environment taken off,
    /// `H_xx - H_xy H_yy^-1 H_yx` (the rows of fixed atoms of the environment are left out:
    /// they do not respond). Refused where the reduction does not fit in the memory a Hessian
    /// may take.
    pub fn core_hessian_from(&self, given: &Mat<f64>) -> MoptResult<Mat<f64>> {
        let n = 3 * self.split.core.len();
        let whole = self.template.dof();
        if given.nrows() != given.ncols() {
            return Err(MoptError::config(format!(
                "a Hessian is square; this one has {} rows and {} columns",
                given.nrows(),
                given.ncols()
            )));
        }
        if given.nrows() == n {
            return Ok(given.clone());
        }
        if given.nrows() != whole {
            return Err(MoptError::config(format!(
                "a Hessian of {} coordinates; with ONIOM microiterations the first Hessian is \
                 that of the core ({n} coordinates) or of the whole system ({whole})",
                given.nrows()
            )));
        }
        let at = |atoms: &[usize], k: usize| 3 * atoms[k / 3] + k % 3;
        let (core, env) = (&self.split.core, &self.split.environment);
        let rows = responding_rows(&self.template, &self.split, self.settings.hessian_shell);
        let m = rows.len();
        self.routes.check(
            given_entries(n, m),
            &format!("the reduction of the given Hessian ({whole} coordinates) to the core"),
        )?;
        let hyy = Mat::from_fn(m, m, |i, j| given[(at(env, rows[i]), at(env, rows[j]))]);
        let hyx = Mat::from_fn(m, n, |i, j| given[(at(env, rows[i]), at(core, j))]);
        let response = environment_response(&hyy, &hyx)?;
        drop(hyy);
        let out = Mat::from_fn(n, n, |i, j| {
            let coupling: f64 = (0..m).map(|k| hyx[(k, i)] * response[(k, j)]).sum();
            given[(at(core, i), at(core, j))] - coupling
        });
        Ok(Mat::from_fn(n, n, |i, j| 0.5 * (out[(i, j)] + out[(j, i)])))
    }

    /// The analytic Hessian of the whole system at `full` (the analytic route): that of the
    /// ONIOM surface, or for a replica of a surface with mechanical embedding that of the lowest
    /// level with the Hessian of the model systems, which is computed once per core for all
    /// replicas.
    fn whole_hessian(
        &self,
        full: &crate::domain::MolecularSystem,
        request: &EvalRequest,
    ) -> MoptResult<Option<Mat<f64>>> {
        let Some(model) = self
            .model_part
            .as_ref()
            .filter(|_| self.model_hessians.is_some())
        else {
            return Ok(self.oniom.evaluate(full, request)?.hessian_bohr);
        };
        let low = EvalRequest {
            excited_state: 0,
            ..*request
        };
        let Some(h) = self.low.evaluate(full, &low)?.hessian_bohr else {
            return Ok(None);
        };
        let atoms: Vec<usize> = (0..full.n_atoms()).collect();
        Ok(self
            .model_hessian_on(model, full, &atoms, request.excited_state)?
            .map(|m| h + m))
    }

    /// Whether the Hessian of the core comes from the analytic Hessian of the whole system:
    /// every level has one, and its matrices fit in the memory a Hessian may take
    /// ([`whole_entries`]).
    pub(super) fn analytic_core_hessian(&self) -> bool {
        self.routes.whole
    }

    /// The Hessian of the core from the Hessian `whole` of the whole system at the relaxed
    /// environment: the block of the core with [`CoreHessian::Held`], the response of the
    /// environment taken off otherwise (whose columns, `dy/dx = -H_yy^-1 H_yx`, are kept for
    /// the prediction). The response is the one of the dense route: by the Cholesky factor of
    /// `H_yy` where it is positive definite, over its positive curvatures where it is not
    /// (an environment on a saddle point of its own, which is said with the saddle; an
    /// inverse over the negative curvatures as well would be the response of a stationary
    /// point of the environment that is not its minimum).
    fn reduced(
        &self,
        whole: &Mat<f64>,
        rows: &[usize],
    ) -> MoptResult<(Mat<f64>, Option<EnvironmentSaddle>)> {
        let n = 3 * self.split.core.len();
        let at = |atoms: &[usize], k: usize| 3 * atoms[k / 3] + k % 3;
        let (core, env) = (&self.split.core, &self.split.environment);
        let hxx = Mat::from_fn(n, n, |i, j| whole[(at(core, i), at(core, j))]);
        let m = rows.len();
        if self.settings.core_hessian == CoreHessian::Held || m == 0 {
            return Ok((
                Mat::from_fn(n, n, |i, j| 0.5 * (hxx[(i, j)] + hxx[(j, i)])),
                None,
            ));
        }
        let hyy = Mat::from_fn(m, m, |i, j| whole[(at(env, rows[i]), at(env, rows[j]))]);
        let hyx = Mat::from_fn(m, n, |i, j| whole[(at(env, rows[i]), at(core, j))]);
        // The block of the environment over its positive curvatures, for the correction of
        // the gradient of the relaxations after this Hessian.
        let dense = DenseResponse::from_block(rows, &hyy, &hyx).map(|(d, _, _)| d);
        let all = 3 * env.len();
        let saddle = dense
            .as_ref()
            .and_then(|d| d.saddle(self.settings.inner.mode_kill.threshold, all));
        let response = match (positive_environment_response(&hyy, &hyx), &dense) {
            (Some(r), _) => r,
            (None, Some(d)) => d.z.clone(),
            (None, None) => environment_response(&hyy, &hyx)?,
        };
        drop(hyy);
        if let Ok(mut state) = self.state.lock() {
            if let Some(r) = state.response.as_mut() {
                let all = 3 * env.len();
                for k in 0..n {
                    let column = Col::from_fn(m, |j| -response[(j, k)]);
                    r.set_column(k, &embed(&column, rows, all));
                }
            }
            state.dense = dense.map(std::sync::Arc::new);
        }
        let out = Mat::from_fn(n, n, |i, j| {
            hxx[(i, j)] - (0..m).map(|k| hyx[(k, i)] * response[(k, j)]).sum::<f64>()
        });
        Ok((
            Mat::from_fn(n, n, |i, j| 0.5 * (out[(i, j)] + out[(j, i)])),
            saddle,
        ))
    }

    /// The block `H_xx` of the core and the coupling `H_yx` of the responding coordinates
    /// `rows` of the environment to it (`rows.len() x n`) at `x` with the environment at `env`,
    /// by central differences of the gradient of `calc` asked with `request`.
    #[allow(clippy::too_many_arguments)]
    fn difference_columns(
        &self,
        calc: &dyn Calculator,
        request: &EvalRequest,
        x: &Col<f64>,
        env: &[f64],
        spin: SpinCharge,
        rows: &[usize],
    ) -> MoptResult<(Mat<f64>, Mat<f64>)> {
        let n = x.nrows();
        let m = rows.len();
        let h = self.settings.hessian_step_bohr;
        let mut hxx = Mat::<f64>::zeros(n, n);
        let mut hyx = Mat::<f64>::zeros(m, n);
        for k in 0..n {
            let mut columns = Vec::with_capacity(2);
            for sign in [1.0, -1.0] {
                let mut moved = x.clone();
                moved[k] += sign * h;
                let mut full = assemble(&self.template, &self.split, &moved, env)?;
                full.spin_charge = spin;
                columns.push(calc.evaluate(&full, request)?.gradient_bohr);
            }
            let core_plus = gather(&columns[0], &self.split.core);
            let core_minus = gather(&columns[1], &self.split.core);
            for i in 0..n {
                hxx[(i, k)] = (core_plus[i] - core_minus[i]) / (2.0 * h);
            }
            if m > 0 {
                let env_plus = gather(&columns[0], &self.split.environment);
                let env_minus = gather(&columns[1], &self.split.environment);
                for (j, &r) in rows.iter().enumerate() {
                    hyx[(j, k)] = (env_plus[r] - env_minus[r]) / (2.0 * h);
                }
            }
        }
        Ok((hxx, hyx))
    }

    /// `H_xx` and `H_yx` of the split route: the analytic Hessians of the terms of the model
    /// systems on the core and central differences of the gradient of the lowest level
    /// (`None` where a term gave no Hessian after all).
    #[allow(clippy::too_many_arguments)]
    fn split_columns(
        &self,
        model: &ModelPart,
        x: &Col<f64>,
        env: &[f64],
        spin: SpinCharge,
        excited_state: usize,
        rows: &[usize],
    ) -> MoptResult<Option<(Mat<f64>, Mat<f64>)>> {
        let mut full = assemble(&self.template, &self.split, x, env)?;
        full.spin_charge = spin;
        let Some(terms) = self.model_hessian_on(model, &full, &self.split.core, excited_state)?
        else {
            return Ok(None);
        };
        if let Some(c) = self
            .counters
            .as_ref()
            .filter(|_| self.model_hessians.is_none())
        {
            c.record_hessian();
        }
        let (low, hyx) = self.difference_columns(
            self.low.as_ref(),
            &self.inner_request(excited_state),
            x,
            env,
            spin,
            rows,
        )?;
        Ok(Some((low + terms, hyx)))
    }

    /// The Hessian of the core at `x` with the environment relaxed at `env`: from the
    /// analytic Hessian of the whole system where it is had ([`Self::analytic_core_hessian`]),
    /// otherwise from central differences (of the lowest level with the analytic Hessians of
    /// the model systems, or of the ONIOM gradient), which give the coupling `H_yx` of the
    /// environment to the core as well. With [`CoreHessian::Effective`] the response of the
    /// environment is taken off, `H_xx - H_xy H_yy^-1 H_yx` (the Hessian of `E*`, the curvature
    /// the optimizer sees), with `H_yy^-1` applied by conjugate gradients on Hessian-vector
    /// products of the lowest level, preconditioned by the limited-memory BFGS of the
    /// relaxations (in the coordinates of its pairs: `H_y^-1 = B^-1 H_q^-1 B^-T` for the square
    /// Wilson matrix of the fragment coordinates); with [`CoreHessian::Approximate`] by that
    /// inverse alone (no further gradients). The solutions are the columns of the response of
    /// the environment, `dy/dx = -H_yy^-1 H_yx`, kept for the prediction. Refused where the
    /// matrices of the differences do not fit in the memory a Hessian may take. Where the
    /// block of the responding environment says that `env` stands on a saddle point of the
    /// environment (the analytic and the dense routes), the saddle comes with the Hessian.
    pub(super) fn core_hessian(
        &self,
        x: &Col<f64>,
        env: &[f64],
        spin: SpinCharge,
        excited_state: usize,
        stability: bool,
    ) -> MoptResult<CoreHessianOut> {
        let mut full = assemble(&self.template, &self.split, x, env)?;
        full.spin_charge = spin;
        let rows = if self.settings.core_hessian == CoreHessian::Held {
            Vec::new()
        } else {
            responding_rows(&full, &self.split, self.settings.hessian_shell)
        };
        if self.settings.core_hessian != CoreHessian::Approximate && self.analytic_core_hessian() {
            let request = EvalRequest {
                excited_state,
                ..EvalRequest::ENERGY_GRAD_HESS
            };
            if let Some(whole) = self.whole_hessian(&full, &request)? {
                let (hessian, saddle) = self.reduced(&whole, &rows)?;
                return Ok(CoreHessianOut {
                    hessian: Some(hessian),
                    saddle,
                });
            }
        }
        let n = x.nrows();
        let m = rows.len();
        let coupling = m > 0;
        self.routes.check(
            difference_entries(n, m),
            &format!("the Hessian of the core ({n} coordinates, {m} responding around it)"),
        )?;
        let split = match &self.routes.model {
            Some(model) => self.split_columns(model, x, env, spin, excited_state, &rows)?,
            None => None,
        };
        let (hxx, hyx) = match split {
            Some(columns) => columns,
            None => {
                let request = EvalRequest {
                    excited_state,
                    ..EvalRequest::ENERGY_GRAD
                };
                self.difference_columns(self.oniom.as_ref(), &request, x, env, spin, &rows)?
            }
        };
        if !coupling {
            return Ok(CoreHessianOut {
                hessian: Some(symmetric(&hxx)),
                saddle: None,
            });
        }
        // What the solves take from the state, which is not locked across an evaluation: the
        // pairs of the limited-memory BFGS (read), the fragment coordinates (taken out and put
        // back), the newest preconditioner (shared), and the log.
        let poisoned = |_| MoptError::calc("ONIOM microiterations: the state is poisoned");
        let (memory, mut fragments, preconditioner, progress) = {
            let mut guard = self.state.lock().map_err(poisoned)?;
            let state = &mut *guard;
            (
                state.memory.clone(),
                state.fragments.take(),
                state.preconditioners.back().cloned(),
                state.progress.clone(),
            )
        };
        let outcome = self.with_response(
            hxx,
            &hyx,
            x,
            env,
            spin,
            excited_state,
            stability,
            &rows,
            &memory,
            &mut fragments,
            preconditioner.as_deref(),
            progress.as_ref().map(|(log, _)| log),
        );
        let mut guard = self.state.lock().map_err(poisoned)?;
        let state = &mut *guard;
        state.fragments = fragments;
        let taken = outcome?;
        if let (Some(columns), Some(r)) = (taken.columns.as_ref(), state.response.as_mut()) {
            for (k, column) in columns.iter().enumerate() {
                r.set_column(k, column);
            }
        }
        // The dense response of the newest Hessian corrects the relaxations after it.
        state.dense = taken.dense.map(std::sync::Arc::new);
        Ok(CoreHessianOut {
            hessian: taken.hessian.map(|h| symmetric(&h)),
            saddle: taken.saddle,
        })
    }

    /// The Hessian of the core with the response of the responding environment `rows` taken
    /// off the core block `out` (coupling `hyx`): by the dense block `H_SS` of the responding
    /// environment where that is chosen ([`ResponseSolve`]), by conjugate gradients
    /// otherwise. `None` for a check of stability whose response cannot be verified.
    #[allow(clippy::too_many_arguments)]
    fn with_response(
        &self,
        mut out: Mat<f64>,
        hyx: &Mat<f64>,
        x: &Col<f64>,
        env: &[f64],
        spin: SpinCharge,
        excited_state: usize,
        stability: bool,
        rows: &[usize],
        memory: &crate::pes::microiter::MicroMemory,
        fragments: &mut Option<super::fragments::FragmentCoordinates>,
        model: Option<&super::precondition::EnvironmentPreconditioner>,
        progress: Option<&crate::core_infra::log::Logger>,
    ) -> MoptResult<Taken> {
        let n = out.nrows();
        let m = rows.len();
        let all = 3 * self.split.environment.len();
        let exact = self.settings.core_hessian == CoreHessian::Effective;
        let problem = LowEnvironment {
            low: self.inner(),
            template: &self.template,
            split: &self.split,
            core: x,
            spin,
            request: self.inner_request(excited_state),
            report: None,
            tally: super::Tally::default(),
            preconditioner: None,
            guard: None,
        };
        let block = extreme_curvatures(&out);
        let core_masses: Vec<f64> = self
            .split
            .core
            .iter()
            .flat_map(|&a| {
                let m = self.template.elements[a].mass_amu();
                [m, m, m]
            })
            .collect();
        if exact {
            if let Some(source) = self.dense_source(n, m)? {
                return self.dense_response(
                    out,
                    hyx,
                    &problem,
                    env,
                    rows,
                    source,
                    stability,
                    block,
                    &core_masses,
                    progress,
                );
            }
        }
        let held = out.clone();
        if let Some(f) = fragments.as_mut() {
            f.prepare(env)?;
        }
        let fragments = fragments.as_ref();
        // The inverse of the limited-memory BFGS of the relaxations on the responding
        // coordinates (a principal block of a positive definite matrix is one too), on top
        // of the factorized model of the environment that the relaxations use where there
        // is one: the pairs alone are a scaled identity with a few corrections, and the
        // curvatures of an environment span four orders of magnitude.
        let precondition = |v: &Col<f64>| -> Col<f64> {
            let v = embed(v, rows, all);
            let z = match (fragments, memory.generation(), model) {
                (Some(f), Some(_), _) => f.step_linear(&memory.apply_inverse(&f.gradient_to_q(&v))),
                (_, _, Some(p)) => memory.apply_inverse_with(&v, &|u| p.apply(u)),
                _ => memory.apply_inverse(&v),
            };
            restrict(&z, rows)
        };
        // The response of `micro_hessian=approximate` (no solves): the inverse of the pairs
        // alone, which on the test surfaces is closer to `H_yy^-1` than with the model
        // under it (the model is a preconditioner, not an approximation of the inverse).
        let approximate = |v: &Col<f64>| -> Col<f64> {
            let v = embed(v, rows, all);
            let z = match (fragments, memory.generation()) {
                (Some(f), Some(_)) => f.step_linear(&memory.apply_inverse(&f.gradient_to_q(&v))),
                _ => memory.apply_inverse(&v),
            };
            restrict(&z, rows)
        };
        let mut solved = Solved::default();
        let mut tally = SolveTally::default();
        let mut coupling_matrix = Mat::<f64>::zeros(n, n);
        let mut columns: Vec<Col<f64>> = Vec::with_capacity(n);
        let mut solutions: Vec<Col<f64>> = Vec::with_capacity(n);
        // A check of stability solves to convergence (conjugate gradients take at most as
        // many products as the environment has coordinates); the steps take the solves
        // as far as SOLVE_PRODUCTS go, with the Galerkin response below.
        let max_products = if stability {
            SOLVE_PRODUCTS.max(m)
        } else {
            SOLVE_PRODUCTS
        };
        let mut unconverged = 0_usize;
        for k in 0..n {
            let b = Col::from_fn(m, |j| hyx[(j, k)]);
            let mut converged = !exact;
            let z = if exact {
                let (z, end) = solve_environment(
                    &problem,
                    env,
                    &b,
                    rows,
                    &precondition,
                    &solved,
                    max_products,
                )?;
                if end.residual > SOLVE_TOLERANCE || end.not_positive || end.ran_away {
                    unconverged += 1;
                }
                let b_norm = col_norm(&b);
                tally.add(
                    &end,
                    if b_norm > 0.0 {
                        col_norm(&z) / b_norm
                    } else {
                        0.0
                    },
                );
                converged = end.residual <= 10.0 * SOLVE_TOLERANCE;
                z
            } else {
                approximate(&b)
            };
            for i in 0..n {
                let coupling: f64 = (0..m).map(|j| hyx[(j, i)] * z[j]).sum();
                coupling_matrix[(i, k)] = coupling;
            }
            columns.push(embed(&-&z, rows, all));
            // Only a solution that converged is a pair of the matrix for the starts.
            if exact && converged && col_norm(&z) > 0.0 {
                solved.pairs.push((z.clone(), b));
            }
            solutions.push(z);
        }
        // The dense check fits the same memory budget when the responding block is
        // smaller than the core and the old Krylov vectors are released first.
        if stability && exact && unconverged > 0 && m <= n {
            drop(solved);
            solutions.clear();
            if let Some(z) = direct_stability_response(&problem, env, rows, hyx)? {
                coupling_matrix = hyx.transpose() * &z;
                columns = (0..n)
                    .map(|k| {
                        let col = Col::from_fn(m, |i| -z[(i, k)]);
                        embed(&col, rows, all)
                    })
                    .collect();
                unconverged = 0;
                if let Some(log) = progress {
                    log.info(
                        "  Stability check: solved the responding environment directly using \
                         central differences.",
                    );
                }
            }
        }
        // The lowest wavenumbers of the core block and of the effective Hessian with the
        // response assembled column by column, for the log.
        let column_wise = Mat::from_fn(n, n, |i, k| out[(i, k)] - coupling_matrix[(i, k)]);
        let lowest_block = lowest_wavenumber(&out, &core_masses);
        let lowest_columns = lowest_wavenumber(&column_wise, &core_masses);
        // A check of stability is judged on converged solves alone.
        if stability && exact && unconverged > 0 {
            if let Some(log) = progress {
                log.info(&format!(
                    "  Hessian of the core for a check of stability: {unconverged} of {n} \
                     solve(s) with the environment ({m} coordinates) did not converge to a \
                     residual of {SOLVE_TOLERANCE:.0e} within {max_products} product(s) \
                     (largest residual {:.1e}; {} stopped at a direction without positive \
                     curvature, {} at the limit, {} where the residual or the \
                     preconditioner ran away; lowest curvature met {:+.3e} Hartree/Bohr^2); \
                     the stability cannot be judged",
                    tally.worst_residual,
                    tally.not_positive,
                    tally.at_limit,
                    tally.ran_away,
                    tally.lowest
                ));
            }
            return Ok(Taken::none());
        }
        // The Galerkin response in the span of the solutions for the steps: positive
        // semidefinite and never above the exact one, whatever the solves reached (see
        // `galerkin_response`). A check takes the converged solutions as they are.
        if exact && !stability {
            if let Some((c, z)) =
                galerkin_response(&problem, env, rows, hyx, &solutions, &mut tally)?
            {
                coupling_matrix = c;
                columns = z.iter().map(|z| embed(&-z, rows, all)).collect();
            }
        }
        for i in 0..n {
            for k in 0..n {
                out[(i, k)] -= coupling_matrix[(i, k)];
            }
        }
        // A response that no positive definite H_yy gives is noise of the differences:
        // the block of the core is a better model than a matrix with curvatures of 1e20.
        let verdict =
            plausible_response(&coupling_matrix, block.map_or(0.0, |(_, highest)| highest));
        let mut keep_columns = true;
        match &verdict {
            Ok(()) => {}
            Err(why) if stability => {
                if let Some(log) = progress {
                    log.info(&format!(
                        "  Hessian of the core for a check of stability: the response of \
                         the environment is not one a positive definite environment gives \
                         ({why}); the stability cannot be judged"
                    ));
                }
                return Ok(Taken::none());
            }
            Err(why) => {
                out = held;
                keep_columns = false;
                if let Some(log) = progress {
                    log.info(&format!(
                        "  Hessian of the core: the response of the environment is left out and \
                         the core block taken alone, since {why} (conjugate gradients on \
                         differences of the gradient of the lowest level)"
                    ));
                }
            }
        }
        if let Some(log) = progress {
            let show = |w: Option<f64>| w.map_or_else(|| "none".to_string(), |w| format!("{w:.1}"));
            log.info(&format!(
                "  Hessian of the core: lowest wavenumbers (cm^-1, the masses of the core) of \
                 the core block {}, of the effective Hessian with the response column by \
                 column {}, and of the one taken ({}) {}",
                show(lowest_block),
                show(lowest_columns),
                if stability {
                    "converged solves, for a check of stability"
                } else {
                    "the Galerkin response, for the steps"
                },
                show(lowest_wavenumber(&out, &core_masses))
            ));
            let effective = extreme_curvatures(&out);
            let range = |c: Option<(f64, f64)>| match c {
                Some((lo, hi)) => format!("{lo:+.3e} to {hi:+.3e}"),
                None => "none".to_string(),
            };
            log.info(&format!(
                "  Hessian of the core: {} solve(s) with the environment ({} coordinates) by \
                 conjugate gradients, {} Hessian-vector product(s) ({} gradients of the lowest \
                 level); {} stopped at a direction without positive curvature, {} at the limit \
                 of {max_products}, {} where the residual ran away; largest residual {:.1e}, \
                 lowest curvature met {:+.3e} Hartree/Bohr^2, largest response |z|/|b| \
                 {:.3e} Bohr^2/Hartree; curvatures of the core block {}, of the effective \
                 Hessian {}",
                tally.solves,
                m,
                tally.products,
                2 * tally.products,
                tally.not_positive,
                tally.at_limit,
                tally.ran_away,
                tally.worst_residual,
                tally.lowest,
                tally.largest_response,
                range(block),
                range(effective)
            ));
        }
        Ok(Taken {
            hessian: Some(out),
            columns: keep_columns.then_some(columns),
            dense: None,
            saddle: None,
        })
    }

    /// Whether the response is solved with the dense block of the responding environment
    /// (`m` coordinates, a core of `n`), and how the block is had: `None` for conjugate
    /// gradients. The analytic Hessian of the lowest level where it has one and the whole
    /// system fits in the memory; otherwise central differences where their `2 m` gradients
    /// cost no more than the products the conjugate gradients may take
    /// ([`DENSE_SHARE`]). Refused where `micro_hessian_solve=dense` asks for a block that does
    /// not fit in the memory.
    fn dense_source(&self, n: usize, m: usize) -> MoptResult<Option<BlockSource>> {
        let entries = 2.0 * (m as f64) * (m as f64) + 2.0 * (m as f64) * (n as f64);
        let fits = memory::f64_bytes(entries) <= self.routes.budget.0;
        let analytic = !self.settings.embedded
            && self.low.analytic_hessian()
            && memory::f64_bytes(whole_entries(self.template.dof(), m)) <= self.routes.budget.0;
        let source = if analytic {
            BlockSource::Analytic
        } else {
            BlockSource::Differences
        };
        match self.settings.response_solve {
            ResponseSolve::ConjugateGradients => Ok(None),
            ResponseSolve::Dense => {
                self.routes.check(
                    entries,
                    &format!("the dense block of the responding environment ({m} coordinates)"),
                )?;
                Ok(Some(source))
            }
            ResponseSolve::Auto => {
                let cheap =
                    analytic || (2 * m) as f64 <= DENSE_SHARE * (2 * n * SOLVE_PRODUCTS) as f64;
                Ok((fits && cheap).then_some(source))
            }
        }
    }

    /// The Hessian of the core with the response taken off by the dense block `H_SS` of the
    /// responding environment: the analytic Hessian of the lowest level or central
    /// differences of its gradient, symmetrized, decomposed (`V diag(w) V^T`), and inverted
    /// over its positive curvatures. At a relaxed environment `H_SS` is positive definite
    /// and this is `H_xx - H_xS H_SS^-1 H_Sx` to the accuracy of the differences; curvatures
    /// that are not positive (an environment not at a minimum, or noise) are left out and
    /// said in the log, and a check of stability is then not judged.
    #[allow(clippy::too_many_arguments)]
    fn dense_response(
        &self,
        mut out: Mat<f64>,
        hyx: &Mat<f64>,
        problem: &LowEnvironment<'_>,
        env: &[f64],
        rows: &[usize],
        source: BlockSource,
        stability: bool,
        block: Option<(f64, f64)>,
        core_masses: &[f64],
        progress: Option<&crate::core_infra::log::Logger>,
    ) -> MoptResult<Taken> {
        let n = out.nrows();
        let m = rows.len();
        let all = 3 * self.split.environment.len();
        let started = std::time::Instant::now();
        let (hss, gradients) = match source {
            BlockSource::Analytic => {
                let mut full = assemble(&self.template, &self.split, problem.core, env)?;
                let request = EvalRequest::ENERGY_GRAD_HESS;
                full.spin_charge = problem.spin;
                let whole = self
                    .low
                    .evaluate(&full, &request)?
                    .hessian_bohr
                    .ok_or_else(|| {
                        MoptError::calc(
                            "ONIOM microiterations: the lowest level gave no Hessian for the \
                             block of the environment",
                        )
                    })?;
                if let Some(c) = &self.counters {
                    c.record_hessian();
                }
                let at = |k: usize| 3 * self.split.environment[k / 3] + k % 3;
                (
                    Mat::from_fn(m, m, |i, j| whole[(at(rows[i]), at(rows[j]))]),
                    0,
                )
            }
            BlockSource::Differences => {
                let mut h = Mat::<f64>::zeros(m, m);
                for (j, &row) in rows.iter().enumerate() {
                    let mut moved = env.to_vec();
                    moved[row] = env[row] + PRODUCT_STEP;
                    let plus = problem.evaluate(&moved, false)?.gradient;
                    moved[row] = env[row] - PRODUCT_STEP;
                    let minus = problem.evaluate(&moved, false)?.gradient;
                    for (i, &r) in rows.iter().enumerate() {
                        h[(i, j)] = (plus[r] - minus[r]) / (2.0 * PRODUCT_STEP);
                    }
                }
                (h, 2 * m)
            }
        };
        // The inverse over the positive curvatures (relative to the largest: below that the
        // block has no curvature the differences resolve).
        let (dense, dropped, lowest) =
            DenseResponse::from_block(rows, &hss, hyx).ok_or_else(|| {
                MoptError::linalg(
                    "ONIOM microiterations: the eigenvalues of the block of the environment",
                )
            })?;
        let top = dense.values.iter().fold(0.0_f64, |a, x| a.max(*x));
        let z = dense.z.clone();
        let coupling = symmetric(&(hyx.transpose() * &z));
        let seconds = started.elapsed().as_secs_f64();
        let how = match source {
            BlockSource::Analytic => "the analytic Hessian of the lowest level".to_string(),
            BlockSource::Differences => format!(
                "central differences of the gradient of the lowest level ({gradients} \
                 gradients, step {PRODUCT_STEP:.0e} Bohr)"
            ),
        };
        let largest_response = (0..n)
            .map(|k| {
                let b = Col::from_fn(m, |i| hyx[(i, k)]);
                let zk = Col::from_fn(m, |i| z[(i, k)]);
                let bn = col_norm(&b);
                if bn > 0.0 {
                    col_norm(&zk) / bn
                } else {
                    0.0
                }
            })
            .fold(0.0_f64, f64::max);
        let lowest_block = lowest_wavenumber(&out, core_masses);
        // A curvature of the block below minus the threshold of the mode kill of the
        // environment: the environment stands on a saddle point of its own. Its response is
        // not that of a minimum, so the evaluation leaves the saddle and takes the Hessian
        // again; a check of stability is not judged on this one. Curvatures between that and
        // zero are flat directions of the environment (left out of the inverse, as without
        // coupling they add nothing).
        let saddle = dense.saddle(self.settings.inner.mode_kill.threshold, all);
        if saddle.is_some() && stability {
            if let Some(log) = progress {
                log.info(&format!(
                    "  Hessian of the core for a check of stability: the block of the \
                     responding environment ({m} coordinates, from {how}) has the curvature \
                     {lowest:+.3e} Hartree/Bohr^2: the environment stands on a saddle point of \
                     its own, and the stability is not judged on this Hessian"
                ));
            }
            return Ok(Taken {
                saddle,
                ..Taken::none()
            });
        }
        let held = out.clone();
        for i in 0..n {
            for k in 0..n {
                out[(i, k)] -= coupling[(i, k)];
            }
        }
        let verdict = plausible_response(&coupling, block.map_or(0.0, |(_, highest)| highest));
        if let Err(why) = &verdict {
            if stability {
                if let Some(log) = progress {
                    log.info(&format!(
                        "  Hessian of the core for a check of stability: the response of the \
                         environment is not one a positive definite environment gives ({why}); \
                         the stability cannot be judged"
                    ));
                }
                return Ok(Taken::none());
            }
            out = held;
        }
        if let Some(log) = progress {
            let show = |w: Option<f64>| w.map_or_else(|| "none".to_string(), |w| format!("{w:.1}"));
            let range = |c: Option<(f64, f64)>| match c {
                Some((lo, hi)) => format!("{lo:+.3e} to {hi:+.3e}"),
                None => "none".to_string(),
            };
            log.info(&format!(
                "  Hessian of the core: the response of the environment from its block of {m} \
                 coordinates by {how} in {seconds:.1} s; curvatures of the block {lowest:+.3e} \
                 to {top:+.3e} Hartree/Bohr^2{}; largest response |z|/|b| {largest_response:.3e} \
                 Bohr^2/Hartree; curvatures of the core block {}, of the effective Hessian {}; \
                 lowest wavenumbers (cm^-1, the masses of the core) of the core block {}, of \
                 the effective Hessian {}",
                if dropped > 0 {
                    format!(", {dropped} not positive and left out")
                } else {
                    String::new()
                },
                range(block),
                range(extreme_curvatures(&out)),
                show(lowest_block),
                show(lowest_wavenumber(&out, core_masses))
            ));
            if let Err(why) = &verdict {
                log.info(&format!(
                    "  Hessian of the core: the response of the environment is left out and the \
                     core block taken alone, since {why}"
                ));
            }
        }
        if verdict.is_err() {
            return Ok(Taken {
                hessian: Some(out),
                columns: None,
                dense: None,
                saddle,
            });
        }
        let columns = (0..n)
            .map(|k| {
                let col = Col::from_fn(m, |i| -z[(i, k)]);
                embed(&col, rows, all)
            })
            .collect();
        Ok(Taken {
            hessian: Some(out),
            columns: Some(columns),
            dense: Some(dense),
            saddle,
        })
    }
}

/// What a Hessian of the core gives: the Hessian (`None`: a check of stability that cannot
/// be judged), the columns of the response for the prediction of the relaxations, and the
/// dense response for the correction of the gradient.
struct Taken {
    hessian: Option<Mat<f64>>,
    columns: Option<Vec<Col<f64>>>,
    dense: Option<DenseResponse>,
    saddle: Option<EnvironmentSaddle>,
}

impl Taken {
    fn none() -> Self {
        Self {
            hessian: None,
            columns: None,
            dense: None,
            saddle: None,
        }
    }
}

/// `(H + H^T) / 2`.
fn symmetric(h: &Mat<f64>) -> Mat<f64> {
    Mat::from_fn(h.nrows(), h.ncols(), |i, j| 0.5 * (h[(i, j)] + h[(j, i)]))
}

/// An eigenvalue of the dense block of the environment below this share of the largest is
/// left out of its inverse.
const POSITIVE_SHARE: f64 = 1e-10;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_residual_of_a_quadratic_environment_is_corrected_exactly() {
        // E(x, y) = x^T A x / 2 + x^T C^T y + y^T B y / 2 with one core coordinate pair and
        // three environment coordinates: at y0 = y*(x) + d the residual is g_y = B d, and the
        // corrected gradient and energy are those at the minimum y*(x) = -B^-1 C x.
        let a = Mat::from_fn(2, 2, |i, j| [[0.8, 0.1], [0.1, 0.5]][i][j]);
        let b = Mat::from_fn(3, 3, |i, j| {
            [[0.9, 0.2, 0.0], [0.2, 0.4, 0.05], [0.0, 0.05, 0.3]][i][j]
        });
        let c = Mat::from_fn(3, 2, |i, j| [[0.1, -0.2], [0.05, 0.3], [-0.15, 0.1]][i][j]);
        let x = Col::from_fn(2, |i| [0.3, -0.4][i]);
        let energy = |y: &Col<f64>| {
            0.5 * col_dot(&x, &(&a * &x)) + col_dot(&(&c * &x), y) + 0.5 * col_dot(y, &(&b * y))
        };
        let g_x = |y: &Col<f64>| &a * &x + c.transpose() * y;
        let star = -(crate::math::linalg::pseudo_inverse(&b).unwrap() * (&c * &x));
        let d = Col::from_fn(3, |i| [0.01, -0.02, 0.015][i]);
        let y0 = &star + &d;
        let g_y = &b * &d;
        let (dense, dropped, lowest) = DenseResponse::from_block(&[0, 1, 2], &b, &c).unwrap();
        assert_eq!(dropped, 0);
        assert!(lowest > 0.0);
        let (dg, de) = dense.correction(&g_y).unwrap();
        let corrected = g_x(&y0) + &dg;
        let exact = g_x(&star);
        assert!(crate::math::linalg::col_max_abs(&(&corrected - &exact)) < 1e-14);
        assert!((energy(&y0) + de - energy(&star)).abs() < 1e-14);
        // A residual that would move the environment beyond the reach of the model is not
        // corrected.
        let far = Col::from_fn(3, |i| [1.0, 0.0, 0.0][i]);
        let reach = dense.correction(&(&b * &far)).unwrap_err();
        assert!(reach > CORRECTION_REACH);
    }

    #[test]
    fn only_a_response_that_a_positive_definite_environment_gives_is_taken() {
        // H_xy H_yy^-1 H_yx for a positive definite H_yy: positive semidefinite and within
        // the reach of the core block.
        let c = Mat::from_fn(2, 2, |i, j| [[0.3, 0.1], [0.1, 0.2]][i][j]);
        assert!(plausible_response(&c, 0.8).is_ok());
        assert!(plausible_response(&Mat::<f64>::zeros(2, 2), 0.8).is_ok());
        // A negative eigenvalue, curvatures of 1e21, or no finite values: noise.
        let negative = Mat::from_fn(2, 2, |i, j| [[0.3, 0.0], [0.0, -0.2]][i][j]);
        assert!(plausible_response(&negative, 0.8)
            .unwrap_err()
            .contains("not positive semidefinite"));
        let huge = Mat::from_fn(2, 2, |i, j| if i == j { 1.7e21 } else { 0.0 });
        assert!(plausible_response(&huge, 1.5)
            .unwrap_err()
            .contains("beyond"));
        let nan = Mat::from_fn(2, 2, |_, _| f64::NAN);
        assert!(plausible_response(&nan, 1.5).is_err());
    }
}
