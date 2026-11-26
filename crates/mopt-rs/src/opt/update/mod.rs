//! Quasi-Newton Hessian updaters (extension point), single-secant and block (multi-secant).
//!
//! An updater turns the newest secant pair `(s, y)` into a correction of the current Hessian;
//! block updaters additionally keep a window of earlier pairs. Updaters are selected from the
//! optimizer method string by [`UpdaterRegistry`], which reproduces MultiOptPy's substring
//! dispatch (most specific key first; `flowchart` when nothing matches).
//!
//! MultiOptPy: the choice of the update by method name in
//!  `optimizer.py:CalculateMoveVector.initialization` and the optimizers of `Optimizer/`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

pub mod block;
pub mod formulas;

use std::sync::Arc;

use faer::{Col, Mat};

use crate::error::{MoptError, MoptResult};
use crate::math::linalg::{col_dot, col_norm, symmetrize};
use crate::math::projection::project_out_basis_mat;

pub use block::{BlockKind, BlockUpdater};

/// Hessian update rule.
pub trait HessianUpdater: Send {
    /// Canonical name.
    fn name(&self) -> &str;

    /// Record the newest secant pair and return the correction `dB` (`B + dB` is the update).
    fn correction(&mut self, b: &Mat<f64>, s: &Col<f64>, y: &Col<f64>) -> MoptResult<Mat<f64>>;

    /// Whether the formula is valid for pairs with `s . y <= 0`, i.e. can build negative
    /// curvature (SR1, PSB, Bofill, FSB, MSP, TS-BFGS, and their block forms; not BFGS).
    fn handles_negative_curvature(&self) -> bool {
        false
    }

    /// Whether saddle searches should feed the formula pairs with `s . y <= 0`: true for the
    /// formulas without a BFGS term (SR1, PSB, Bofill, MSP, TS-BFGS, and the block Bofill
    /// forms). A BFGS term divides by `s . y`; with a small negative value it puts a large
    /// negative curvature along `y`, which can send a saddle search down a wrong mode (FSB and
    /// its block forms keep MultiOptPy's rule; see `docs/source/migration_optimizers.md`).
    fn learns_saddle_curvature(&self) -> bool {
        false
    }

    /// Forget stored secant pairs.
    fn reset(&mut self) {}

    /// Apply `map` to the steps and gradient changes of the stored secant pairs (the
    /// structure they belong to was turned, and they turn with it).
    fn map_pairs(&mut self, _map: &dyn Fn(&Col<f64>) -> Col<f64>) {}
}

/// Single-secant updater defined by a plain function.
pub struct FnUpdater {
    name: String,
    f: fn(&Mat<f64>, &Col<f64>, &Col<f64>) -> Mat<f64>,
    negative_curvature: bool,
    saddle_curvature: bool,
}

impl FnUpdater {
    /// Wrap `f` under `name`.
    pub fn new(name: &str, f: fn(&Mat<f64>, &Col<f64>, &Col<f64>) -> Mat<f64>) -> Self {
        Self {
            name: name.to_string(),
            f,
            negative_curvature: false,
            saddle_curvature: false,
        }
    }

    /// Wrap `f`, an update that is valid for pairs with `s . y <= 0`.
    pub fn indefinite(name: &str, f: fn(&Mat<f64>, &Col<f64>, &Col<f64>) -> Mat<f64>) -> Self {
        Self {
            negative_curvature: true,
            ..Self::new(name, f)
        }
    }

    /// Mark the formula as free of BFGS terms (see
    /// [`HessianUpdater::learns_saddle_curvature`]).
    pub fn without_bfgs_term(mut self) -> Self {
        self.saddle_curvature = true;
        self
    }
}

impl HessianUpdater for FnUpdater {
    fn name(&self) -> &str {
        &self.name
    }
    fn correction(&mut self, b: &Mat<f64>, s: &Col<f64>, y: &Col<f64>) -> MoptResult<Mat<f64>> {
        Ok((self.f)(b, s, y))
    }
    fn handles_negative_curvature(&self) -> bool {
        self.negative_curvature
    }
    fn learns_saddle_curvature(&self) -> bool {
        self.saddle_curvature
    }
}

/// Apply `updater` to `hessian` in place, with MultiOptPy's guards: the update is skipped when
/// `||s||` or `||y||` is below `1e-10`, or when `s . y <= 0`. Pairs with `s . y <= 0` are used
/// when negative curvature must be learned (`allow_negative_curvature`: CCQN leaving a
/// potential well, and saddle searches with a formula for which
/// [`HessianUpdater::learns_saddle_curvature`] holds) and the formula supports them. Returns
/// whether it was applied.
///
/// MultiOptPy (`Optimizer/rsirfo.py:RSIRFO.update_hessian`) skips `s . y <= 0` always, also in
/// saddle searches, where steps along the transition vector have exactly that sign (see
/// `docs/source/migration_optimizers.md`).
pub fn apply_update(
    updater: &mut dyn HessianUpdater,
    hessian: &mut Mat<f64>,
    s: &Col<f64>,
    y: &Col<f64>,
    allow_negative_curvature: bool,
) -> MoptResult<bool> {
    let negative_ok = allow_negative_curvature && updater.handles_negative_curvature();
    if col_norm(s) < 1e-10 || col_norm(y) < 1e-10 || (col_dot(s, y) <= 0.0 && !negative_ok) {
        return Ok(false);
    }
    let d = updater.correction(hessian, s, y)?;
    *hessian += d;
    symmetrize(hessian);
    Ok(true)
}

/// Remove the rigid motions spanned by the orthonormal columns of `basis` from a stored
/// Hessian, in place (`P H P`, `P = I - D D^T`): called after every update (also where the
/// formula skipped the pair: the Hessian is brought to the newer structure) and after an exact
/// Hessian is taken over, so that the Hessian an optimizer keeps has no curvature along the
/// translations and rotations of the structure it belongs to (`-update_proj`; see
/// `docs/source/optimization.md`). A secant update puts curvature
/// there (the pair `(s, y)` of two structures is not orthogonal to the rigid motions of the
/// newer one), and an exact Hessian has it at the size of its noise. Nothing is done without a
/// basis or with one of another dimension (plain variables, a surface that changes under rigid
/// motions, internal coordinates).
///
/// MultiOptPy: `Utils/calc_tools.py:Calculationtools.project_out_hess_tr_and_rot_for_coord`
/// is the projection; applying it to the stored Hessian after every update is new.
pub fn project_out_rigid(hessian: &mut Mat<f64>, basis: Option<&Mat<f64>>) {
    if let Some(d) = basis {
        if d.ncols() > 0 && d.nrows() == hessian.nrows() && hessian.nrows() == hessian.ncols() {
            *hessian = project_out_basis_mat(d, hessian);
        }
    }
}

/// Curvatures below minus this (Hartree/Bohr^2, or the units of the coordinates) count as
/// negative in [`negative_curvatures`]: rigid motions and directions without curvature sit
/// near zero in an updated Hessian, at about the round-off of the updates.
pub const NEGATIVE_CURVATURE: f64 = 1e-6;

/// The number of eigenvalues of the symmetric part of `hessian` below `-NEGATIVE_CURVATURE`.
pub fn negative_curvatures(hessian: &Mat<f64>) -> MoptResult<usize> {
    let mut h = hessian.clone();
    symmetrize(&mut h);
    let (w, _) = crate::math::linalg::symmetric_eigen(&h)?;
    Ok(w.iter().filter(|v| **v < -NEGATIVE_CURVATURE).count())
}

/// `None` where `hessian` has at most `allowed` negative curvatures, else their number: the
/// guard of a saddle-point search on its updates (new; MultiOptPy has none). A secant update
/// that gives an updated Hessian more negative curvatures than the saddle order and the last
/// exact Hessian had is noise read as curvature where the steps are short and the gradients
/// noisy (the relaxed surface of ONIOM microiterations: soft modes of 1e-4 Hartree/Bohr^2 and a
/// noise of 1e-5 in the change of the gradient over a step of 0.03 Bohr), and the step along
/// such a mode goes as far as the trust radius lets it; the state machine skips such an
/// update and keeps the Hessian it had. This is a heuristic for the updates of the steps, not
/// a criterion of what a stationary point is: whether a structure is a transition state is
/// counted by `opt::mode_kill` alone (mass-weighted eigenvalues of an exact Hessian below
/// `-IMAGINARY_CURVATURE`), and the state machine leaves the rigid motions out of the count here.
pub fn keeps_inertia(hessian: &Mat<f64>, allowed: usize) -> MoptResult<Option<usize>> {
    let mut h = hessian.clone();
    symmetrize(&mut h);
    let (w, _) = crate::math::linalg::symmetric_eigen(&h)?;
    let negative: Vec<f64> = w
        .iter()
        .copied()
        .filter(|v| *v < -NEGATIVE_CURVATURE)
        .collect();
    let count = negative.len();
    if count <= allowed {
        return Ok(None);
    }
    // Only soft new negative curvatures are noise: the ones beyond `allowed` (those nearest
    // to zero) must all be below SOFT_SHARE of the most negative one. A curvature of the size of
    // the followed one is what the update learned of the surface (CCQN and RS-I-RFO on the
    // Muller-Brown surface and on LJ7 need those).
    let most = negative.first().map_or(0.0, |v| v.abs());
    let soft = negative[allowed..]
        .iter()
        .all(|v| v.abs() < SOFT_SHARE * most);
    Ok(soft.then_some(count))
}

/// A negative curvature that an update adds is taken for noise (and the update skipped by
/// [`keeps_inertia`]) when it is below this share of the most negative curvature: the spurious
/// ones on chorismate mutase were -1.7e-4 to -1.4e-3 Hartree/Bohr^2 next to a followed
/// curvature of -0.11 (a share of 1.3e-2 at most).
pub const SOFT_SHARE: f64 = 0.02;

/// Factory producing an updater.
pub type UpdaterFactory = Arc<dyn Fn() -> Box<dyn HessianUpdater> + Send + Sync>;

/// Ordered `(key, factory)` table; the first key contained in the method string wins.
#[derive(Clone)]
pub struct UpdaterRegistry {
    entries: Vec<(String, UpdaterFactory)>,
    default_key: String,
}

fn fn_factory(
    name: &'static str,
    f: fn(&Mat<f64>, &Col<f64>, &Col<f64>) -> Mat<f64>,
) -> UpdaterFactory {
    Arc::new(move || {
        // Every single-secant formula except BFGS remains valid for s . y <= 0.
        let up = if name.starts_with("bfgs") {
            FnUpdater::new(name, f)
        } else if [
            "sr1",
            "psb",
            "msp",
            "bofill",
            "cfd_bofill",
            "pcfd_bofill",
            "ts_bfgs",
        ]
        .contains(&name)
        {
            FnUpdater::indefinite(name, f).without_bfgs_term()
        } else {
            FnUpdater::indefinite(name, f)
        };
        Box::new(up) as Box<dyn HessianUpdater>
    })
}

fn block_factory(name: &'static str, kind: BlockKind) -> UpdaterFactory {
    Arc::new(move || Box::new(BlockUpdater::new(name, kind)) as Box<dyn HessianUpdater>)
}

impl Default for UpdaterRegistry {
    fn default() -> Self {
        use formulas as f;
        use BlockKind as K;
        // Order mirrors MultiOptPy's dispatch list (most specific first).
        let table: Vec<(&'static str, UpdaterFactory)> = vec![
            ("flowchart", fn_factory("flowchart", f::flowchart)),
            ("ts_bfgs", fn_factory("ts_bfgs", f::ts_bfgs)),
            (
                "block_cfd_fsb_dd",
                block_factory("block_cfd_fsb_dd", K::CfdFsbDamped),
            ),
            (
                "block_cfd_fsb_weighted",
                block_factory("block_cfd_fsb_weighted", K::CfdFsbWeighted),
            ),
            ("block_cfd_fsb", block_factory("block_cfd_fsb", K::CfdFsb)),
            (
                "block_cfd_bofill_weighted",
                block_factory("block_cfd_bofill_weighted", K::CfdBofillWeighted),
            ),
            (
                "block_cfd_bofill",
                block_factory("block_cfd_bofill", K::CfdBofill),
            ),
            (
                "block_bfgs_dd",
                block_factory("block_bfgs_dd", K::BfgsDamped),
            ),
            ("block_bfgs", block_factory("block_bfgs", K::Bfgs)),
            ("block_fsb_dd", block_factory("block_fsb_dd", K::FsbDamped)),
            (
                "block_fsb_weighted",
                block_factory("block_fsb_weighted", K::FsbWeighted),
            ),
            ("block_fsb", block_factory("block_fsb", K::Fsb)),
            (
                "block_bofill_weighted",
                block_factory("block_bofill_weighted", K::BofillWeighted),
            ),
            ("block_bofill", block_factory("block_bofill", K::Bofill)),
            (
                "bfgs_dd",
                fn_factory("bfgs_dd", |b, s, y| {
                    f::bfgs(b, s, &f::double_damping(s, y, 0.2))
                }),
            ),
            ("bfgs", fn_factory("bfgs", f::bfgs)),
            ("sr1", fn_factory("sr1", f::sr1)),
            ("pcfd_bofill", fn_factory("pcfd_bofill", f::pcfd_bofill)),
            (
                "cfd_fsb_dd",
                fn_factory("cfd_fsb_dd", |b, s, y| {
                    f::fsb(b, s, &f::double_damping(s, y, 0.2), 2.0)
                }),
            ),
            (
                "cfd_fsb",
                fn_factory("cfd_fsb", |b, s, y| f::fsb(b, s, y, 2.0)),
            ),
            (
                "cfd_bofill",
                fn_factory("cfd_bofill", |b, s, y| f::bofill(b, s, y, 2.0)),
            ),
            (
                "fsb_dd",
                fn_factory("fsb_dd", |b, s, y| {
                    f::fsb(b, s, &f::double_damping(s, y, 0.2), 1.0)
                }),
            ),
            ("fsb", fn_factory("fsb", |b, s, y| f::fsb(b, s, y, 1.0))),
            (
                "bofill",
                fn_factory("bofill", |b, s, y| f::bofill(b, s, y, 1.0)),
            ),
            ("psb", fn_factory("psb", f::psb)),
            ("msp", fn_factory("msp", f::msp)),
        ];
        Self {
            entries: table.into_iter().map(|(k, v)| (k.to_string(), v)).collect(),
            default_key: "flowchart".to_string(),
        }
    }
}

impl UpdaterRegistry {
    /// Register `factory` under `key` with the highest priority.
    pub fn register(&mut self, key: &str, factory: UpdaterFactory) {
        self.entries.insert(0, (key.to_ascii_lowercase(), factory));
    }

    /// Registered keys in priority order.
    pub fn keys(&self) -> Vec<&str> {
        self.entries.iter().map(|(k, _)| k.as_str()).collect()
    }

    /// Resolve the updater for `method` (substring match, default `flowchart`).
    pub fn resolve(&self, method: &str) -> MoptResult<Box<dyn HessianUpdater>> {
        self.resolve_or(method, &self.default_key)
    }

    /// Resolve the updater for `method`, using `default_key` when no key matches.
    pub fn resolve_or(
        &self,
        method: &str,
        default_key: &str,
    ) -> MoptResult<Box<dyn HessianUpdater>> {
        let lower = method.to_ascii_lowercase();
        let hit = self
            .entries
            .iter()
            .find(|(k, _)| lower.contains(k.as_str()))
            .or_else(|| self.entries.iter().find(|(k, _)| k == default_key));
        hit.map(|(_, factory)| factory())
            .ok_or_else(|| MoptError::config(format!("no Hessian updater for '{method}'")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_update_that_adds_a_negative_curvature_breaks_the_inertia() {
        // diag(-0.1, 3e-4, 1): one negative curvature, and a Bofill update from a short step
        // along the soft mode whose gradient change says -2e-4.
        let h = Mat::from_fn(3, 3, |i, j| if i == j { [-0.1, 3e-4, 1.0][i] } else { 0.0 });
        assert_eq!(negative_curvatures(&h).unwrap(), 1);
        assert_eq!(keeps_inertia(&h, 1).unwrap(), None);
        let reg = UpdaterRegistry::default();
        let mut up = reg.resolve("bofill").unwrap();
        let mut updated = h.clone();
        let s = Col::from_fn(3, |i| [0.0, 0.03, 0.0][i]);
        let y = Col::from_fn(3, |i| [0.0, -6e-6, 0.0][i]);
        assert!(apply_update(up.as_mut(), &mut updated, &s, &y, true).unwrap());
        assert_eq!(keeps_inertia(&updated, 1).unwrap(), Some(2));
        // A second negative curvature that the exact Hessian had is allowed.
        assert_eq!(keeps_inertia(&updated, 2).unwrap(), None);
        // Curvatures at the round-off of zero do not count.
        let flat = Mat::from_fn(2, 2, |i, j| if i == j { [-1e-9, 1.0][i] } else { 0.0 });
        assert_eq!(negative_curvatures(&flat).unwrap(), 0);
    }

    #[test]
    fn substring_dispatch_prefers_specific_keys() {
        let reg = UpdaterRegistry::default();
        let name = |m: &str| reg.resolve(m).unwrap().name().to_string();
        assert_eq!(name("rsirfo_cfd_fsb_dd"), "cfd_fsb_dd");
        assert_eq!(name("rsirfo_fsb"), "fsb");
        assert_eq!(name("mf_rsirfo_pcfd_bofill"), "pcfd_bofill");
        assert_eq!(name("rsirfo_bofill"), "bofill");
        assert_eq!(name("rsirfo_block_bfgs"), "block_bfgs");
        assert_eq!(
            name("rsirfo_block_cfd_bofill_weighted"),
            "block_cfd_bofill_weighted"
        );
        assert_eq!(name("rsirfo_block_fsb_dd"), "block_fsb_dd");
        assert_eq!(name("whatever"), "flowchart");
        assert_eq!(name("rsirfo_ts_bfgs"), "ts_bfgs");
        assert_eq!(name("ccqn"), "flowchart");
        assert_eq!(reg.resolve_or("ccqn", "ts_bfgs").unwrap().name(), "ts_bfgs");
        assert_eq!(
            reg.resolve_or("ccqn_bofill", "ts_bfgs").unwrap().name(),
            "bofill"
        );
    }

    #[test]
    fn update_is_skipped_for_negative_curvature() {
        let reg = UpdaterRegistry::default();
        let mut up = reg.resolve("bfgs").unwrap();
        let mut h = Mat::<f64>::identity(2, 2);
        let s = Col::from_fn(2, |i| [1.0, 0.0][i]);
        let y = Col::from_fn(2, |i| [-1.0, 0.0][i]);
        assert!(!apply_update(up.as_mut(), &mut h, &s, &y, true).unwrap());
        let y = Col::from_fn(2, |i| [2.0, 0.0][i]);
        assert!(apply_update(up.as_mut(), &mut h, &s, &y, false).unwrap());
        assert!((h[(0, 0)] - 2.0).abs() < 1e-14);
    }

    #[test]
    fn negative_curvature_only_when_requested_and_supported() {
        let reg = UpdaterRegistry::default();
        let s = Col::from_fn(2, |i| [1.0, 0.0][i]);
        let y = Col::from_fn(2, |i| [-0.5, 0.0][i]);
        for key in ["ts_bfgs", "bofill", "block_bofill", "block_fsb", "sr1"] {
            let mut up = reg.resolve(key).unwrap();
            let mut h = Mat::<f64>::identity(2, 2);
            assert!(
                !apply_update(up.as_mut(), &mut h, &s, &y, false).unwrap(),
                "{key}"
            );
            assert!(
                apply_update(up.as_mut(), &mut h, &s, &y, true).unwrap(),
                "{key}"
            );
            assert!(h[(0, 0)] < 0.0, "{key}: {}", h[(0, 0)]);
        }
        for key in ["bfgs", "block_bfgs"] {
            let mut up = reg.resolve(key).unwrap();
            let mut h = Mat::<f64>::identity(2, 2);
            assert!(
                !apply_update(up.as_mut(), &mut h, &s, &y, true).unwrap(),
                "{key}"
            );
        }
    }
}
