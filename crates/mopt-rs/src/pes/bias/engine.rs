//! Shared machinery of the bias potentials.
//!
//! A [`GeometricBias`] states which atoms it depends on and its energy as a function of their
//! positions, written once over [`Scalar`]. [`AdBias`] turns it into a [`PesTerm`] with exact
//! gradients and Hessians by forward-mode automatic differentiation over the coordinates of
//! those atoms only. Geometry goes through [`Geom`], which applies minimum-image shifts along
//! the periodic axes of the cell (isolated, 1D, 2D, or 3D). The shifts are piecewise constant,
//! so they do not enter the derivatives, and every bias works under periodic boundary
//! conditions without bias-specific code. In a cell [`AdBias`] also returns the stress
//! exactly: under a homogeneous strain `r -> (1 + e) r` every coordinate and every
//! minimum-image shift (a lattice vector combination) scales, so the nine strain components
//! are seeded as dual numbers ([`Geom::strained`]); fixed points of a bias do not move.
//!
//! MultiOptPy: `Potential/potential.py` (`BiasPotentialCalculation`, `gradually_change_param`).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::sync::Arc;

use faer::{Col, Mat};

use crate::domain::{AtomId, EvalOutput, MolecularSystem, PeriodicCell, Quantity};
use crate::error::{MoptError, MoptResult};
use crate::math::ad::{eval_value_and_gradient, eval_value_grad_hessian, Dual, Scalar};
use crate::pes::term::{check_term_output, Frame, History, PesTerm, TermContext};

/// MultiOptPy's parameter ramp (`gradually_change_param`): linear from `start` to `end` over
/// 300 iterations, then constant; a single value is constant.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ramp {
    /// Value at iteration 0.
    pub start: f64,
    /// Final value (`None`: constant).
    pub end: Option<f64>,
}

impl Ramp {
    /// Iterations of the ramp.
    pub const ITERATIONS: f64 = 300.0;

    /// Constant value.
    pub fn constant(v: f64) -> Self {
        Self {
            start: v,
            end: None,
        }
    }

    /// Value at `iteration`.
    pub fn at(&self, iteration: usize) -> f64 {
        let Some(end) = self.end else {
            return self.start;
        };
        let v = self.start + (end - self.start) / Self::ITERATIONS * iteration as f64;
        if self.start < end {
            v.min(end)
        } else {
            v.max(end)
        }
    }

    /// Parse `"v"` or `"v1,v2"`.
    pub fn parse(s: &str) -> MoptResult<Self> {
        let vals = s
            .split(',')
            .map(|t| {
                t.trim()
                    .parse::<f64>()
                    .map_err(|e| MoptError::parse(format!("'{t}': {e}")))
            })
            .collect::<MoptResult<Vec<f64>>>()?;
        match vals.as_slice() {
            [v] => Ok(Self::constant(*v)),
            [a, b] => Ok(Self {
                start: *a,
                end: Some(*b),
            }),
            _ => Err(MoptError::parse(format!(
                "expected one value or two comma-separated values, got '{s}'"
            ))),
        }
    }

    /// Whether any value of the ramp is zero (MultiOptPy then skips the term).
    pub fn has_zero(&self) -> bool {
        self.start == 0.0 || self.end == Some(0.0)
    }
}

/// Iteration and simulated time at which a bias is evaluated.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Clock {
    /// Optimizer iteration (parameter ramps follow it).
    pub iteration: usize,
    /// Simulated time (atomic units) in molecular dynamics; `None` in optimizations.
    pub time_au: Option<f64>,
}

impl Clock {
    /// Clock of a term evaluation.
    pub fn of(ctx: &TermContext<'_>) -> Self {
        Self {
            iteration: ctx.iteration,
            time_au: ctx.time_au,
        }
    }

    /// Clock at an iteration without simulated time.
    pub fn at_iteration(iteration: usize) -> Self {
        Self {
            iteration,
            time_au: None,
        }
    }
}

/// Positions of the atoms a bias depends on, with minimum-image geometry.
pub struct Geom<'a, T: Scalar> {
    atoms: &'a [AtomId],
    x: &'a [T],
    values: &'a [f64],
    cell: Option<&'a PeriodicCell>,
    /// Homogeneous strain `e` applied to the lattice shifts (`s -> (1 + e) s`), for the
    /// derivatives of the energy along the strain; `None`: the shifts are constants.
    strain: Option<[[T; 3]; 3]>,
}

impl<'a, T: Scalar> Geom<'a, T> {
    /// Geometry of the sorted, unique `atoms` with coordinates `x` (three per atom, in the order
    /// of `atoms`), their values `values` (used for the minimum-image shifts), and the cell.
    pub fn new(
        atoms: &'a [AtomId],
        x: &'a [T],
        values: &'a [f64],
        cell: Option<&'a PeriodicCell>,
    ) -> Self {
        Self {
            atoms,
            x,
            values,
            cell,
            strain: None,
        }
    }

    /// [`Self::new`] under the homogeneous strain `strain` (`r -> (1 + e) r`): `x` holds the
    /// strained coordinates, and every minimum-image shift (a lattice vector combination with
    /// the integers chosen at `values`) is strained in the same way. Seeding `strain` with
    /// derivative parts gives the strain derivatives of the energy exactly.
    pub fn strained(
        atoms: &'a [AtomId],
        x: &'a [T],
        values: &'a [f64],
        cell: Option<&'a PeriodicCell>,
        strain: [[T; 3]; 3],
    ) -> Self {
        Self {
            strain: Some(strain),
            ..Self::new(atoms, x, values, cell)
        }
    }

    /// The lattice shift `s` as a variable: `(1 + e) s` under a strain `e`.
    fn lift(&self, s: [f64; 3]) -> [T; 3] {
        match &self.strain {
            None => s.map(T::from_f64),
            Some(e) => std::array::from_fn(|a| {
                let mut v = T::from_f64(s[a]);
                for (b, sb) in s.iter().enumerate() {
                    v += e[a][b] * T::from_f64(*sb);
                }
                v
            }),
        }
    }

    /// Minimum image of a difference vector `v` (shifted by lattice vectors of the periodic
    /// axes; unchanged for isolated systems).
    pub fn mic(&self, v: [T; 3]) -> [T; 3] {
        let s = self.lift(self.shift(v.map(|c| c.val())));
        std::array::from_fn(|c| v[c] + s[c])
    }

    /// Minimum-image vector `r_a - p` to a point `p` that is itself a variable.
    pub fn d_to(&self, a: AtomId, p: [T; 3]) -> [T; 3] {
        let pa = self.pos(a);
        self.mic(std::array::from_fn(|c| pa[c] - p[c]))
    }

    fn slot(&self, a: AtomId) -> usize {
        // `atoms` is sorted and contains every atom the bias declared.
        self.atoms.binary_search(&a).unwrap_or(0)
    }

    fn values_of(&self, a: AtomId) -> [f64; 3] {
        let k = self.slot(a);
        [
            self.values[3 * k],
            self.values[3 * k + 1],
            self.values[3 * k + 2],
        ]
    }

    fn shift(&self, raw: [f64; 3]) -> [f64; 3] {
        match self.cell {
            Some(c) => c.mic_lattice_shift_bohr(raw, [0.0; 3]),
            None => [0.0; 3],
        }
    }

    /// Position of `a` (as stored; use differences for anything periodic).
    pub fn pos(&self, a: AtomId) -> [T; 3] {
        let k = self.slot(a);
        [self.x[3 * k], self.x[3 * k + 1], self.x[3 * k + 2]]
    }

    /// Minimum-image vector `r_a - r_b`.
    pub fn d(&self, a: AtomId, b: AtomId) -> [T; 3] {
        let (va, vb) = (self.values_of(a), self.values_of(b));
        let s = self.lift(self.shift([va[0] - vb[0], va[1] - vb[1], va[2] - vb[2]]));
        let (pa, pb) = (self.pos(a), self.pos(b));
        std::array::from_fn(|c| pa[c] - pb[c] + s[c])
    }

    /// Minimum-image vector `r_a - p` to a fixed point `p` (Bohr). The point is a parameter of
    /// the bias: a strain moves the atom and the lattice, not the point.
    pub fn d_point(&self, a: AtomId, p: [f64; 3]) -> [T; 3] {
        let va = self.values_of(a);
        let raw = self.shift([va[0] - p[0], va[1] - p[1], va[2] - p[2]]);
        let pa = self.pos(a);
        if self.strain.is_none() {
            return std::array::from_fn(|c| pa[c] - T::from_f64(p[c] - raw[c]));
        }
        let s = self.lift(raw);
        std::array::from_fn(|c| pa[c] - T::from_f64(p[c]) + s[c])
    }

    /// Centroid of `atoms`, with each atom taken at its image nearest to the first one.
    pub fn centroid(&self, atoms: &[AtomId]) -> [T; 3] {
        let a0 = atoms[0];
        let p0 = self.pos(a0);
        let n = T::from_f64(atoms.len() as f64);
        let mut acc = [T::zero(); 3];
        for &a in atoms {
            let d = self.d(a, a0);
            for c in 0..3 {
                acc[c] += d[c];
            }
        }
        std::array::from_fn(|c| p0[c] + acc[c] / n)
    }

    /// Minimum-image vector between two centroids.
    pub fn d_centroids(&self, a: &[AtomId], b: &[AtomId]) -> [T; 3] {
        let (ca, cb) = (self.centroid(a), self.centroid(b));
        let raw: [f64; 3] = std::array::from_fn(|c| ca[c].val() - cb[c].val());
        let s = self.lift(self.shift(raw));
        std::array::from_fn(|c| ca[c] - cb[c] + s[c])
    }
}

/// Euclidean norm.
pub fn norm<T: Scalar>(v: [T; 3]) -> T {
    T::norm3(v)
}

/// A bias written as an energy of the positions of a set of atoms.
pub trait GeometricBias: Send + Sync {
    /// Canonical name.
    fn name(&self) -> &str;

    /// Atoms the energy depends on.
    fn atoms(&self) -> Vec<AtomId>;

    /// Energy (Hartree) at `g` and `clock`.
    fn energy<T: Scalar>(&self, g: &Geom<'_, T>, clock: Clock) -> T;

    /// Record an accepted geometry (see [`PesTerm::commit`]).
    fn commit(&self, _g: &Geom<'_, f64>, _clock: Clock) {}

    /// See [`PesTerm::frame`]; the default is the safe [`Frame::Absolute`].
    fn frame(&self) -> Frame {
        Frame::Absolute
    }

    /// See [`PesTerm::history`]; the default is the safe [`History::Stateful`].
    fn history(&self) -> History {
        History::Stateful
    }

    /// A copy with its own state (see [`PesTerm::fork`]); `None`: no state of its own.
    fn fork(&self) -> Option<Self>
    where
        Self: Sized,
    {
        None
    }

    /// Why the energy is not defined at `g`, for the error message when it comes out
    /// non-finite (a geometric variable that does not exist there, a power of a negative
    /// number); `None` when the bias cannot tell.
    fn diagnose(&self, _g: &Geom<'_, f64>, _clock: Clock) -> Option<String> {
        None
    }
}

/// The history of a bias whose constants follow `ramps`: [`History::IterationRamp`] when any
/// of them changes over the iterations, [`History::Static`] otherwise.
pub(crate) fn ramp_history(ramps: &[&Ramp]) -> History {
    if ramps.iter().any(|r| r.end.is_some()) {
        History::IterationRamp
    } else {
        History::Static
    }
}

/// Sorted unique atoms of a bias, checked against the system, and their coordinates.
pub(crate) fn local_coordinates(
    atoms: Vec<AtomId>,
    sys: &MolecularSystem,
) -> MoptResult<(Vec<AtomId>, Vec<f64>)> {
    let mut atoms = atoms;
    atoms.sort_unstable();
    atoms.dedup();
    for a in &atoms {
        a.check(sys.n_atoms())?;
    }
    let x = atoms
        .iter()
        .flat_map(|&a| sys.position(a).value())
        .collect();
    Ok((atoms, x))
}

/// Scatter a gradient (and Hessian) over the coordinates of `atoms` into the full system.
pub(crate) fn scatter_local(
    sys: &MolecularSystem,
    atoms: &[AtomId],
    local_g: &[f64],
    local_h: Option<&Mat<f64>>,
) -> (Col<f64>, Option<Mat<f64>>) {
    let dof = sys.dof();
    let mut gradient = Col::zeros(dof);
    for (k, a) in atoms.iter().enumerate() {
        for c in 0..3 {
            gradient[a.dof(c)] = local_g[3 * k + c];
        }
    }
    let hessian = local_h.map(|lh| {
        let mut h = Mat::zeros(dof, dof);
        for (k, a) in atoms.iter().enumerate() {
            for (l, b) in atoms.iter().enumerate() {
                for p in 0..3 {
                    for q in 0..3 {
                        h[(a.dof(p), b.dof(q))] = lh[(3 * k + p, 3 * l + q)];
                    }
                }
            }
        }
        h
    });
    (gradient, hessian)
}

/// [`PesTerm`] with exact derivatives for a [`GeometricBias`].
#[derive(Debug, Clone)]
pub struct AdBias<B> {
    /// The bias.
    pub bias: B,
}

impl<B: GeometricBias> AdBias<B> {
    /// Wrap `bias`.
    pub fn new(bias: B) -> Self {
        Self { bias }
    }
}

/// The stress `(1 / V) dE / d eps` (Hartree/Bohr^3, symmetric; see `pes::stress`) of `bias` at
/// the local coordinates `x` of `atoms` in `cell`, by forward-mode automatic differentiation
/// in the nine strain components: under `r -> (1 + e) r` every coordinate and every
/// minimum-image lattice shift is strained ([`Geom::strained`]), fixed points of the bias stay.
fn strain_stress<B: GeometricBias>(
    bias: &B,
    atoms: &[AtomId],
    x: &[f64],
    cell: &PeriodicCell,
    clock: Clock,
) -> [[f64; 3]; 3] {
    let (_, de) = eval_value_and_gradient(&[0.0; 9], |e| {
        let strain: [[Dual; 3]; 3] = std::array::from_fn(|a| std::array::from_fn(|b| e[3 * a + b]));
        let xs: Vec<Dual> = (0..x.len())
            .map(|i| {
                let (k, a) = (i / 3, i % 3);
                let mut v = Dual::from_f64(x[i]);
                for (b, e_ab) in strain[a].iter().enumerate() {
                    v += *e_ab * Dual::from_f64(x[3 * k + b]);
                }
                v
            })
            .collect();
        bias.energy(&Geom::strained(atoms, &xs, x, Some(cell), strain), clock)
    });
    let volume = cell.volume_bohr3();
    std::array::from_fn(|a| std::array::from_fn(|b| 0.5 * (de[3 * a + b] + de[3 * b + a]) / volume))
}

impl<B: GeometricBias + 'static> PesTerm for AdBias<B> {
    fn name(&self) -> &str {
        self.bias.name()
    }

    fn atoms(&self) -> Option<Vec<AtomId>> {
        Some(self.bias.atoms())
    }

    fn evaluate(
        &self,
        sys: &MolecularSystem,
        ctx: &TermContext<'_>,
        need_hessian: bool,
    ) -> MoptResult<EvalOutput> {
        let (atoms, x) = local_coordinates(self.bias.atoms(), sys)?;
        let cell = sys.active_cell();
        let it = Clock::of(ctx);
        let f = |v: &[_]| self.bias.energy(&Geom::new(&atoms, v, &x, cell), it);
        let (energy, local_g, local_h) = if need_hessian {
            let (e, g, h) = eval_value_grad_hessian(&x, None, f);
            (e, g, Some(h))
        } else {
            let (e, g) = eval_value_and_gradient(&x, |v| {
                self.bias.energy(&Geom::new(&atoms, v, &x, cell), it)
            });
            (e, g, None)
        };
        let g: Vec<f64> = local_g.iter().copied().collect();
        let (gradient_bohr, hessian_bohr) = scatter_local(sys, &atoms, &g, local_h.as_ref());
        let stress = cell.map(|c| strain_stress(&self.bias, &atoms, &x, c, it));
        let out = EvalOutput {
            energy: Quantity::new(energy),
            gradient_bohr,
            hessian_bohr,
            orbitals: None,
            stress,
            strain_hessian: None,
        };
        if let Err(e) = check_term_output(self.bias.name(), &out) {
            let why = self.bias.diagnose(&Geom::new(&atoms, &x, &x, cell), it);
            return Err(match why {
                Some(why) => MoptError::calc(format!("bias '{}': {why}", self.bias.name())),
                None => e,
            });
        }
        Ok(out)
    }

    fn commit(&self, sys: &MolecularSystem, ctx: &TermContext<'_>) {
        if let Ok((atoms, x)) = local_coordinates(self.bias.atoms(), sys) {
            self.bias.commit(
                &Geom::new(&atoms, &x, &x, sys.active_cell()),
                Clock::of(ctx),
            );
        }
    }

    fn frame(&self) -> Frame {
        self.bias.frame()
    }

    fn history(&self) -> History {
        self.bias.history()
    }

    fn fork(&self) -> Option<Arc<dyn PesTerm>> {
        self.bias
            .fork()
            .map(|b| Arc::new(AdBias::new(b)) as Arc<dyn PesTerm>)
    }

    fn analytic_hessian(&self) -> bool {
        true
    }

    fn analytic_stress(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ramp_like_multioptpy() {
        let r = Ramp::parse("1.0,4.0").unwrap_or(Ramp::constant(0.0));
        assert!((r.at(0) - 1.0).abs() < 1e-15);
        assert!((r.at(150) - 2.5).abs() < 1e-15);
        assert!((r.at(10_000) - 4.0).abs() < 1e-15);
        assert!((Ramp::parse("3").map(|r| r.at(99)).unwrap_or(0.0) - 3.0).abs() < 1e-15);
        assert!(Ramp::parse("1,2,3").is_err());
    }
}
