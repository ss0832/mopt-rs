//! Bias potentials with auxiliary variables minimized at every evaluation.
//!
//! An [`AuxiliaryBias`] is an energy `E(x, y)` of atom positions `x` and auxiliary variables `y`
//! (rotation angles of an ellipsoid, positions of solvent particles). [`MicroBias`] turns it into
//! the surface term `E*(x) = min_y E(x, y)`: the inner minimization runs through the same
//! microiteration machinery as ONIOM ([`crate::pes::microiter`], with mode kill), the gradient is
//! `dE/dx` at the relaxed `y`, and the Hessian is the Schur complement
//! `H_xx - H_xy H_yy^+ H_yx`, all from automatic differentiation over `(x, y)`.
//!
//! MultiOptPy: `Potential/asym_elllipsoidal_potential.py`, `Potential/spacer_model_potential.py`
//!  (the biases whose auxiliary positions MultiOptPy optimizes).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::sync::{Arc, Mutex};

use faer::{Col, Mat};

use super::engine::{local_coordinates, scatter_local, Clock, Geom};
use crate::domain::{AtomId, EvalOutput, MolecularSystem, PeriodicCell, Quantity};
use crate::error::{MoptError, MoptResult};
use crate::math::ad::{eval_value_and_gradient, eval_value_grad_hessian, Dual, HyperDual, Scalar};
use crate::pes::microiter::{effective_hessian, relax, MicroEval, MicroProblem, MicroSettings};
use crate::pes::term::{check_term_output, Frame, History, PesTerm, TermContext};

/// A bias written as an energy of atom positions and auxiliary variables.
pub trait AuxiliaryBias: Send + Sync {
    /// Canonical name.
    fn name(&self) -> &str;

    /// Atoms the energy depends on.
    fn atoms(&self) -> Vec<AtomId>;

    /// Number of auxiliary variables.
    fn n_aux(&self) -> usize;

    /// Energy (Hartree) at `g` and auxiliary variables `aux`.
    fn energy<T: Scalar>(&self, g: &Geom<'_, T>, aux: &[T], clock: Clock) -> T;

    /// Starting values of the auxiliary variables when there is no previous solution (or as a
    /// second candidate next to it).
    fn initial_aux(&self, g: &Geom<'_, f64>, clock: Clock) -> Vec<f64>;

    /// Whether `min_y E(x, y)` depends on the absolute position of the atoms (see
    /// [`PesTerm::frame`]); the default is the safe [`Frame::Absolute`].
    fn frame(&self) -> Frame {
        Frame::Absolute
    }

    /// A copy of the bias for [`PesTerm::fork`] (`None`: the term cannot be copied).
    fn duplicate(&self) -> Option<Self>
    where
        Self: Sized,
    {
        None
    }
}

/// [`PesTerm`] `min_y E(x, y)` of an [`AuxiliaryBias`].
#[derive(Debug)]
pub struct MicroBias<B> {
    /// The bias.
    pub bias: B,
    /// Inner minimization settings.
    pub settings: MicroSettings,
    last: Mutex<Option<Vec<f64>>>,
}

impl<B: AuxiliaryBias> MicroBias<B> {
    /// Wrap `bias` with the default microiteration settings. They have no methods of the
    /// optimizer registry (this layer is below the optimizers), so a relaxation whose line
    /// search fails ends there; `with_settings(bias, opt::micro::settings())` has them, as the
    /// commands do.
    pub fn new(bias: B) -> Self {
        Self::with_settings(bias, MicroSettings::default())
    }

    /// Wrap `bias` with `settings`.
    pub fn with_settings(bias: B, settings: MicroSettings) -> Self {
        Self {
            bias,
            settings,
            last: Mutex::new(None),
        }
    }

    /// Auxiliary variables of the last evaluation.
    pub fn auxiliary(&self) -> Option<Vec<f64>> {
        self.last.lock().ok().and_then(|l| l.clone())
    }

    /// Relaxed auxiliary variables at the local coordinates `x` of `atoms`.
    fn relax_aux(
        &self,
        atoms: &[AtomId],
        x: &[f64],
        cell: Option<&PeriodicCell>,
        clock: Clock,
    ) -> MoptResult<Vec<f64>> {
        let problem = AuxProblem {
            bias: &self.bias,
            atoms,
            x,
            cell,
            clock,
        };
        let mut starts = Vec::new();
        if let Some(prev) = self.auxiliary().filter(|p| p.len() == self.bias.n_aux()) {
            starts.push(prev);
        }
        starts.push(self.bias.initial_aux(&Geom::new(atoms, x, x, cell), clock));
        let mut best: Option<crate::pes::microiter::MicroResult> = None;
        for y0 in &starts {
            let r = relax(&problem, y0, &self.settings)?;
            // The previous solution is kept unless the fresh start is clearly lower.
            if best.as_ref().is_none_or(|b| r.energy < b.energy - 1e-12) {
                best = Some(r);
            }
        }
        let best = best.ok_or_else(|| MoptError::calc("microiteration has no start"))?;
        if !best.converged {
            return Err(MoptError::calc(format!(
                "microiteration of '{}' did not converge (max gradient {:.3e} after {} steps)",
                self.bias.name(),
                best.max_gradient,
                best.iterations
            )));
        }
        if let Ok(mut l) = self.last.lock() {
            *l = Some(best.y.clone());
        }
        Ok(best.y)
    }
}

struct AuxProblem<'a, B> {
    bias: &'a B,
    atoms: &'a [AtomId],
    x: &'a [f64],
    cell: Option<&'a PeriodicCell>,
    clock: Clock,
}

impl<B: AuxiliaryBias> MicroProblem for AuxProblem<'_, B> {
    fn dim(&self) -> usize {
        self.bias.n_aux()
    }

    fn evaluate(&self, y: &[f64], need_hessian: bool) -> MoptResult<MicroEval> {
        let (energy, gradient, hessian) = if need_hessian {
            let xt: Vec<HyperDual> = self.x.iter().map(|&v| HyperDual::from_f64(v)).collect();
            let g = Geom::new(self.atoms, &xt, self.x, self.cell);
            let (e, gr, h) =
                eval_value_grad_hessian(y, None, |v| self.bias.energy(&g, v, self.clock));
            (e, gr, Some(h))
        } else {
            let xt: Vec<Dual> = self.x.iter().map(|&v| Dual::from_f64(v)).collect();
            let g = Geom::new(self.atoms, &xt, self.x, self.cell);
            let (e, gr) = eval_value_and_gradient(y, |v| self.bias.energy(&g, v, self.clock));
            (e, gr, None)
        };
        Ok(MicroEval {
            energy,
            gradient,
            hessian,
        })
    }
}

impl<B: AuxiliaryBias + 'static> PesTerm for MicroBias<B> {
    fn name(&self) -> &str {
        self.bias.name()
    }

    fn frame(&self) -> Frame {
        self.bias.frame()
    }

    /// Stateful: the inner minimization starts from the previous solution and keeps it
    /// unless a fresh start is clearly lower, so where `E(x, y)` has several minima in `y`
    /// the value depends on the structures evaluated before.
    fn history(&self) -> History {
        History::Stateful
    }

    /// The Schur complement of exact second derivatives.
    fn analytic_hessian(&self) -> bool {
        true
    }

    /// The same bias and settings with a copy of the last auxiliary variables.
    fn fork(&self) -> Option<Arc<dyn PesTerm>> {
        let bias = self.bias.duplicate()?;
        Some(Arc::new(Self {
            bias,
            settings: self.settings.clone(),
            last: Mutex::new(self.auxiliary()),
        }))
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
        let y = self.relax_aux(&atoms, &x, cell, it)?;
        let nx = x.len();
        let z: Vec<f64> = x.iter().chain(&y).copied().collect();
        let (energy, g, h): (f64, Col<f64>, Option<Mat<f64>>) = if need_hessian {
            let (e, g, h) = eval_value_grad_hessian(&z, None, |v| {
                self.bias
                    .energy(&Geom::new(&atoms, &v[..nx], &x, cell), &v[nx..], it)
            });
            (e, g, Some(effective_hessian(&h, nx)?))
        } else {
            let (e, g) = eval_value_and_gradient(&z, |v| {
                self.bias
                    .energy(&Geom::new(&atoms, &v[..nx], &x, cell), &v[nx..], it)
            });
            (e, g, None)
        };
        let local_g: Vec<f64> = (0..nx).map(|i| g[i]).collect();
        let (gradient_bohr, hessian_bohr) = scatter_local(sys, &atoms, &local_g, h.as_ref());
        let out = EvalOutput {
            energy: Quantity::new(energy),
            gradient_bohr,
            hessian_bohr,
            orbitals: None,
            stress: None,
            strain_hessian: None,
        };
        check_term_output(self.bias.name(), &out)?;
        Ok(out)
    }
}
