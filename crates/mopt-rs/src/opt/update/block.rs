//! Block (multi-secant) Hessian updates (Gao and Goldfarb, arXiv:1609.00318), as in MultiOptPy's
//! `Optimizer/block_hessian_update.py`.
//!
//! The updater keeps the latest `max_window` secant pairs and imposes the last `block_size` of
//! them simultaneously (`B S = Y` for block BFGS / SR1). MultiOptPy dropped each pair right after
//! using it, which reduced every block update to a single-secant one; here the window is kept
//! (see `docs/source/migration_optimizers.md`).
//!
//! Before the pairs are imposed they are prepared (`BlockUpdater::assemble`):
//!
//! - every pair is divided by the length of its step. The block formulas do not change
//!   under a scaling of the pairs, but their safeguards (which pairs count as dependent,
//!   when a matrix counts as singular) are numbers: with the steps as they are, a pair of
//!   a step of `1e-6` is treated differently from one of `0.1`, and the correction of an
//!   optimization that is almost converged depended on which side of a threshold its
//!   steps fell (a chain relaxation showed it: an image at rest jumped by 0.2 Bohr);
//! - going from the newest pair to older ones, a pair is used only if its step makes an
//!   angle of at least `asin(independence)` with the steps already taken. The steps of an
//!   optimization that converges shrink along one line; pairs that are almost parallel
//!   differ by what the surface is not quadratic by, and imposing them together turns
//!   that difference, divided by the small angle, into curvature;
//! - in block SR1 a pair whose residual is perpendicular to its step is left out (the
//!   usual safeguard of SR1).
//!
//! MultiOptPy filters by the singular values of the steps as they are (above `1e-8`) and
//! applies its other thresholds to the pairs as they are; with
//! [`BlockUpdater::scale_pairs`] off the pairs are not scaled, and a pair is left out when
//! its step is not longer than `1e-8`, which is what MultiOptPy's updates do with the one
//! pair they have at a time (the comparisons with MultiOptPy's data use this).
//!
//! Variants: BFGS, FSB (SR1/BFGS mixed with the mean of `phi_j`), Bofill (SR1/PSB mixed with the
//! mean of `phi_j^2`), their CFD forms (`R = 2 (Y - B S)`, i.e. the compact-finite-difference
//! secant condition `B_{k+1} S = 2 Y - B_k S`), "weighted" forms (step-length-weighted mixing),
//! and double-damped forms. PSB is the genuine multi-secant PSB.
//!
//! MultiOptPy: `Optimizer/block_hessian_update.py:BlockHessianUpdate`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::VecDeque;

use faer::{Col, Mat};

use super::formulas::double_damping;
use super::HessianUpdater;
use crate::error::MoptResult;
use crate::math::linalg::{col_dot, safe_inverse, symmetrize};

/// Which block update to apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockKind {
    /// Block BFGS.
    Bfgs,
    /// Block BFGS on damped pairs.
    BfgsDamped,
    /// FSB with the mean mixing weight.
    Fsb,
    /// FSB on damped pairs.
    FsbDamped,
    /// FSB with step-length-weighted mixing.
    FsbWeighted,
    /// CFD-FSB with the mean mixing weight.
    CfdFsb,
    /// CFD-FSB on damped pairs.
    CfdFsbDamped,
    /// CFD-FSB with step-length-weighted mixing.
    CfdFsbWeighted,
    /// Bofill with the mean mixing weight.
    Bofill,
    /// Bofill with step-length-weighted mixing.
    BofillWeighted,
    /// CFD-Bofill with the mean mixing weight.
    CfdBofill,
    /// CFD-Bofill with step-length-weighted mixing.
    CfdBofillWeighted,
}

/// Stateful block updater.
#[derive(Debug, Clone)]
pub struct BlockUpdater {
    name: String,
    kind: BlockKind,
    /// Number of pairs imposed at once.
    pub block_size: usize,
    /// Number of pairs kept.
    pub max_window: usize,
    /// Sine of the smallest angle between the step of a pair and the steps of the newer
    /// pairs that are used with it.
    pub independence: f64,
    /// Divide every pair by the length of its step before the formulas are applied (the
    /// default; off: MultiOptPy's thresholds on the pairs as they are).
    pub scale_pairs: bool,
    history: VecDeque<(Col<f64>, Col<f64>)>,
}

const CURVATURE_THRESHOLD: f64 = 1e-12;
const INV_REG: f64 = 1e-10;
/// Default of [`BlockUpdater::independence`].
pub const INDEPENDENCE: f64 = 0.1;
/// Length of a step up to which a pair is left out when the pairs are not scaled
/// (MultiOptPy's threshold on the singular values of the steps).
const UNSCALED_STEP: f64 = 1e-8;
/// Cosine between the residual and the step of a pair below which block SR1 leaves the
/// pair out.
const SR1_ALIGNMENT: f64 = 1e-8;

/// The pairs that an update imposes.
struct Pairs {
    /// Their steps, each of unit length, the newest last.
    s: Mat<f64>,
    /// Their gradient changes, divided by the lengths of the steps.
    y: Mat<f64>,
    /// The squared lengths of the steps.
    lengths: Vec<f64>,
}

impl BlockUpdater {
    /// New updater with MultiOptPy's sizes (`block_size = 4`, `max_window = 8`).
    pub fn new(name: &str, kind: BlockKind) -> Self {
        Self {
            name: name.to_string(),
            kind,
            block_size: 4,
            max_window: 8,
            independence: INDEPENDENCE,
            scale_pairs: true,
            history: VecDeque::new(),
        }
    }

    /// The pairs to impose (see the module documentation): up to `block_size` of the
    /// stored ones, from the newest on, scaled to unit steps, each independent of the
    /// newer ones.
    fn assemble(&self) -> Pairs {
        let n = self.history.front().map_or(0, |(s, _)| s.nrows());
        let mut picked: Vec<(usize, f64)> = Vec::new();
        let mut basis: Vec<Col<f64>> = Vec::new();
        for (k, (s, _)) in self.history.iter().enumerate().rev() {
            if picked.len() >= self.block_size.max(1) {
                break;
            }
            let length2 = col_dot(s, s);
            if !(length2 > 0.0 && length2.is_finite()) {
                continue;
            }
            if !self.scale_pairs && length2.sqrt() <= UNSCALED_STEP {
                continue;
            }
            // What is left of the unit step after its parts along the newer steps are
            // taken out (twice: the second pass removes what rounding left).
            let mut rest = (1.0 / length2.sqrt()) * s;
            for _ in 0..2 {
                for q in &basis {
                    let along = col_dot(q, &rest);
                    rest -= along * q;
                }
            }
            let left = col_dot(&rest, &rest).sqrt();
            if left >= self.independence.max(f64::EPSILON) {
                basis.push((1.0 / left) * &rest);
                picked.push((k, length2));
            }
        }
        picked.reverse();
        let scale = |c: usize| {
            if self.scale_pairs {
                1.0 / picked[c].1.sqrt()
            } else {
                1.0
            }
        };
        Pairs {
            s: Mat::from_fn(n, picked.len(), |r, c| {
                self.history[picked[c].0].0[r] * scale(c)
            }),
            y: Mat::from_fn(n, picked.len(), |r, c| {
                self.history[picked[c].0].1[r] * scale(c)
            }),
            lengths: picked.iter().map(|(_, length2)| *length2).collect(),
        }
    }
}

fn col_of(m: &Mat<f64>, j: usize) -> Col<f64> {
    Col::from_fn(m.nrows(), |r| m[(r, j)])
}

fn select_cols(m: &Mat<f64>, idx: &[usize]) -> Mat<f64> {
    Mat::from_fn(m.nrows(), idx.len(), |r, c| m[(r, idx[c])])
}

/// `dB = Y (S^T Y)^-1 Y^T - B S (S^T B S)^-1 S^T B` on the columns of positive curvature
/// (the columns are independent: see `BlockUpdater::assemble`).
fn block_bfgs(b: &Mat<f64>, s: &Mat<f64>, y: &Mat<f64>) -> MoptResult<Mat<f64>> {
    let n = b.nrows();
    if s.ncols() == 0 {
        return Ok(Mat::zeros(n, n));
    }
    let (sf, yf) = (s, y);
    // MultiOptPy computes this curvature filter but then ignores it; it is applied here.
    let keep: Vec<usize> = (0..sf.ncols())
        .filter(|&j| col_dot(&col_of(yf, j), &col_of(sf, j)) > CURVATURE_THRESHOLD)
        .collect();
    if keep.is_empty() {
        return Ok(Mat::zeros(n, n));
    }
    let (sf, yf) = (select_cols(sf, &keep), select_cols(yf, &keep));
    let bs = b * &sf;
    let m1 = sf.transpose() * &bs;
    let m2 = sf.transpose() * &yf;
    let inv1 = safe_inverse(&m1, INV_REG)?;
    let inv2 = safe_inverse(&m2, INV_REG)?;
    let mut d = &yf * inv2 * yf.transpose() - &bs * inv1 * bs.transpose();
    symmetrize(&mut d);
    Ok(d)
}

/// Multi-secant symmetric PSB:
/// `dB = R M^-1 S^T + S M^-1 R^T - S M^-1 sym(R^T S) M^-1 S^T` with `M = S^T S`, `R = Y - B S`.
///
/// Satisfies `(B + dB) S = Y` whenever `R^T S` is symmetric (e.g. for quadratic data).
/// MultiOptPy sums independent single-secant PSB corrections instead, which does not satisfy
/// the block secant conditions.
fn block_psb(b: &Mat<f64>, s: &Mat<f64>, y: &Mat<f64>) -> MoptResult<Mat<f64>> {
    let n = b.nrows();
    if s.ncols() == 0 {
        return Ok(Mat::zeros(n, n));
    }
    let (sf, yf) = (s, y);
    let r = yf - b * sf;
    let m_inv = safe_inverse(&(sf.transpose() * sf), INV_REG)?;
    let mut rts_sym = r.transpose() * sf;
    symmetrize(&mut rts_sym);
    let sm = sf * &m_inv;
    let mut d = &r * sm.transpose() + &sm * r.transpose() - &sm * rts_sym * sm.transpose();
    symmetrize(&mut d);
    Ok(d)
}

/// `dB = R (S^T R)^-1 R^T` with `R = scale (Y - B S)`.
fn block_sr1(b: &Mat<f64>, s: &Mat<f64>, y: &Mat<f64>, scale: f64) -> MoptResult<Mat<f64>> {
    let n = b.nrows();
    if s.ncols() == 0 {
        return Ok(Mat::zeros(n, n));
    }
    let r_all = (y - b * s) * scale;
    // Pairs already satisfied by B (negligible residual) make S^T R singular. Dropping them is
    // exact: for symmetric B and consistent data, R^T s_j = S^T r_j vanishes for such a pair, so
    // the correction leaves it satisfied.
    // A residual perpendicular to its step makes S^T R singular as well: such a pair is
    // left out too (it would add a rank-one term of arbitrary size).
    let keep: Vec<usize> = (0..s.ncols())
        .filter(|&j| {
            let rj = col_of(&r_all, j);
            let yj = col_of(y, j);
            let sj = col_of(s, j);
            let residual = col_dot(&rj, &rj).sqrt();
            residual > 1e-10 * col_dot(&yj, &yj).sqrt().max(1.0)
                && col_dot(&rj, &sj).abs() >= SR1_ALIGNMENT * residual * col_dot(&sj, &sj).sqrt()
        })
        .collect();
    if keep.is_empty() {
        return Ok(Mat::zeros(n, n));
    }
    let (s, r) = (select_cols(s, &keep), select_cols(&r_all, &keep));
    let m = s.transpose() * &r;
    let mut d = &r * safe_inverse(&m, INV_REG)? * r.transpose();
    symmetrize(&mut d);
    Ok(d)
}

/// Per-column mixing weights `phi_j^2` (clamped to `[0, 1]`) or `phi_j`.
fn weights(b: &Mat<f64>, s: &Mat<f64>, y: &Mat<f64>, cfd: bool, squared: bool) -> Vec<f64> {
    (0..s.ncols())
        .map(|j| {
            let (sj, yj) = (col_of(s, j), col_of(y, j));
            let mut a = &yj - b * &sj;
            if cfd {
                a *= 2.0;
            }
            let den = col_dot(&a, &a) * col_dot(&sj, &sj);
            let c = if den.abs() > CURVATURE_THRESHOLD {
                col_dot(&a, &sj).powi(2) / den
            } else {
                0.0
            };
            let c = if c.is_nan() { 0.0 } else { c.clamp(0.0, 1.0) };
            if squared {
                c
            } else {
                libm::sqrt(c)
            }
        })
        .collect()
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

fn damp_cols(s: &Mat<f64>, y: &Mat<f64>) -> Mat<f64> {
    let mut out = y.clone();
    for j in 0..s.ncols() {
        let yt = double_damping(&col_of(s, j), &col_of(y, j), 0.2);
        for r in 0..y.nrows() {
            out[(r, j)] = yt[r];
        }
    }
    out
}

/// `w dSR1 + (1 - w) dPartner`.
fn mix(
    b: &Mat<f64>,
    s: &Mat<f64>,
    y: &Mat<f64>,
    w: f64,
    cfd: bool,
    psb_partner: bool,
) -> MoptResult<Mat<f64>> {
    let d_sr1 = block_sr1(b, s, y, if cfd { 2.0 } else { 1.0 })?;
    let d_partner = if psb_partner {
        block_psb(b, s, y)?
    } else {
        block_bfgs(b, s, y)?
    };
    let mut d = d_sr1 * w + d_partner * (1.0 - w);
    symmetrize(&mut d);
    Ok(d)
}

/// Mixture with the arithmetic mean of the per-pair weights (MultiOptPy's default form).
fn mean_mix(
    b: &Mat<f64>,
    s: &Mat<f64>,
    y: &Mat<f64>,
    cfd: bool,
    squared: bool,
    psb_partner: bool,
) -> MoptResult<Mat<f64>> {
    mix(
        b,
        s,
        y,
        mean(&weights(b, s, y, cfd, squared)),
        cfd,
        psb_partner,
    )
}

/// "Weighted" mixture: the per-pair weights are averaged with weights `||s_j||^2`, so long
/// steps dominate the mixing.
///
/// MultiOptPy's weighted-subspace form applies block SR1 to `(S W, Y W)` and the partner to
/// `(S (1 - W), Y (1 - W))` and adds both corrections. Both block formulas are invariant under
/// column scaling, so that sum equals the full SR1 correction plus the full partner correction
/// (double counting; it breaks the secant conditions). A convex combination with a scalar
/// weight keeps them.
fn step_weighted_mix(
    b: &Mat<f64>,
    pairs: &Pairs,
    cfd: bool,
    squared: bool,
    psb_partner: bool,
) -> MoptResult<Mat<f64>> {
    let (s, y, lens) = (&pairs.s, &pairs.y, &pairs.lengths);
    let w = weights(b, s, y, cfd, squared);
    let total: f64 = lens.iter().sum();
    let wbar = if total > 0.0 {
        w.iter().zip(lens).map(|(wi, li)| wi * li).sum::<f64>() / total
    } else {
        0.0
    };
    mix(b, s, y, wbar, cfd, psb_partner)
}

impl HessianUpdater for BlockUpdater {
    fn handles_negative_curvature(&self) -> bool {
        !matches!(self.kind, BlockKind::Bfgs | BlockKind::BfgsDamped)
    }

    fn learns_saddle_curvature(&self) -> bool {
        // The block Bofill forms mix block SR1 and block PSB: no BFGS term. Measured in
        // `docs/source/migration_safeguards.md` ("Saddle searches learn negative curvature").
        matches!(
            self.kind,
            BlockKind::Bofill
                | BlockKind::BofillWeighted
                | BlockKind::CfdBofill
                | BlockKind::CfdBofillWeighted
        )
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn correction(&mut self, b: &Mat<f64>, s: &Col<f64>, y: &Col<f64>) -> MoptResult<Mat<f64>> {
        self.history.push_back((s.clone(), y.clone()));
        while self.history.len() > self.max_window {
            self.history.pop_front();
        }
        let pairs = self.assemble();
        let (sm, ym) = (&pairs.s, &pairs.y);
        use BlockKind as K;
        match self.kind {
            K::Bfgs => block_bfgs(b, sm, ym),
            K::BfgsDamped => block_bfgs(b, sm, &damp_cols(sm, ym)),
            K::Fsb => mean_mix(b, sm, ym, false, false, false),
            K::FsbDamped => mean_mix(b, sm, &damp_cols(sm, ym), false, false, false),
            K::FsbWeighted => step_weighted_mix(b, &pairs, false, false, false),
            K::CfdFsb => mean_mix(b, sm, ym, true, true, false),
            K::CfdFsbDamped => mean_mix(b, sm, &damp_cols(sm, ym), true, true, false),
            K::CfdFsbWeighted => step_weighted_mix(b, &pairs, true, true, false),
            K::Bofill => mean_mix(b, sm, ym, false, true, true),
            K::BofillWeighted => step_weighted_mix(b, &pairs, false, true, true),
            K::CfdBofill => mean_mix(b, sm, ym, true, true, true),
            K::CfdBofillWeighted => step_weighted_mix(b, &pairs, true, true, true),
        }
    }

    fn reset(&mut self) {
        self.history.clear();
    }

    fn map_pairs(&mut self, map: &dyn Fn(&Col<f64>) -> Col<f64>) {
        for (s, y) in &mut self.history {
            *s = map(s);
            *y = map(y);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quadratic() -> Mat<f64> {
        Mat::from_fn(4, 4, |i, j| match (i, j) {
            _ if i == j => [4.0, 3.0, 2.0, 1.5][i],
            (0, 1) | (1, 0) => 0.4,
            (2, 3) | (3, 2) => -0.3,
            _ => 0.0,
        })
    }

    #[test]
    fn block_bfgs_satisfies_all_secant_conditions() {
        let a = quadratic();
        let mut b = Mat::<f64>::identity(4, 4);
        let mut up = BlockUpdater::new("block_bfgs", BlockKind::Bfgs);
        let steps = [
            [0.1, 0.0, 0.05, 0.0],
            [0.0, 0.2, 0.0, -0.1],
            [0.05, -0.05, 0.1, 0.1],
        ];
        for st in steps {
            let s = Col::from_fn(4, |i| st[i]);
            let y = &a * &s;
            b += up.correction(&b, &s, &y).unwrap();
        }
        // The window keeps all three pairs: B s_j = y_j for every stored pair.
        for st in steps {
            let s = Col::from_fn(4, |i| st[i]);
            let r = &b * &s - &a * &s;
            assert!(col_dot(&r, &r).sqrt() < 1e-10);
        }
    }

    #[test]
    fn mixed_block_updates_satisfy_secant_conditions() {
        let a = quadratic();
        for kind in [
            BlockKind::Fsb,
            BlockKind::FsbWeighted,
            BlockKind::Bofill,
            BlockKind::BofillWeighted,
        ] {
            let mut b = Mat::<f64>::identity(4, 4);
            let mut up = BlockUpdater::new("t", kind);
            let steps = [
                [0.1, 0.0, 0.05, 0.0],
                [0.0, 0.2, 0.0, -0.1],
                [0.05, -0.05, 0.1, 0.1],
            ];
            for st in steps {
                let s = Col::from_fn(4, |i| st[i]);
                let y = &a * &s;
                b += up.correction(&b, &s, &y).unwrap();
            }
            for st in steps {
                let s = Col::from_fn(4, |i| st[i]);
                let r = &b * &s - &a * &s;
                assert!(
                    col_dot(&r, &r).sqrt() < 1e-9,
                    "{kind:?} {}",
                    col_dot(&r, &r).sqrt()
                );
            }
        }
    }

    #[test]
    fn all_variants_are_symmetric_and_finite() {
        let a = quadratic();
        for kind in [
            BlockKind::Bfgs,
            BlockKind::BfgsDamped,
            BlockKind::Fsb,
            BlockKind::FsbDamped,
            BlockKind::FsbWeighted,
            BlockKind::CfdFsb,
            BlockKind::CfdFsbDamped,
            BlockKind::CfdFsbWeighted,
            BlockKind::Bofill,
            BlockKind::BofillWeighted,
            BlockKind::CfdBofill,
            BlockKind::CfdBofillWeighted,
        ] {
            let mut b = Mat::<f64>::identity(4, 4);
            let mut up = BlockUpdater::new("t", kind);
            for k in 0..6 {
                let s = Col::from_fn(4, |i| 0.1 * (((i + 2 * k) as f64) * 0.9).sin());
                let y = &a * &s;
                b += up.correction(&b, &s, &y).unwrap();
            }
            for i in 0..4 {
                for j in 0..4 {
                    assert!(b[(i, j)].is_finite(), "{kind:?}");
                    assert!((b[(i, j)] - b[(j, i)]).abs() < 1e-10, "{kind:?}");
                }
            }
        }
    }
}
