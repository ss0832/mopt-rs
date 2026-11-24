//! The searches of `ieipmain` in other coordinates than Cartesian (`-coord`).
//!
//! [`InternalSearchSurface`] makes a surface of structures a surface of their coordinates:
//! a point of a search is the vector of the coordinates of a structure relative to one
//! reference structure, so that the steps, the tangents, the distances along a string, the
//! scaled coordinates of ADDF and 2PSHS, and the Hessians are all in these coordinates.
//! When a point is evaluated, its structure is found by the iterative back-transformation
//! of [`crate::coord::system::transform`] from the nearest structure already known, the
//! surface behind gets that structure, and the gradient and the Hessian are taken to the
//! coordinates with the Wilson matrix there (as the chains of `nebmain` do,
//! [`crate::workflow::chain_internal`]).
//!
//! The coordinates must be independent (no more of them than Cartesian coordinates), so that
//! every vector is the vector of a structure: `dic`, `tric`, `tric-cart`, and any registered
//! system of that kind. They are set up once, at the reference, and kept for the whole run (a
//! search compares points in one set of coordinates); where the system would ask to be set
//! up anew, the log says so. Where they break down, the evaluation fails.
//!
//! [`step_map`] serves the methods whose distances stay Cartesian (iEIP, DS-AFIR): their
//! steps, made in Cartesian coordinates, are taken through the coordinates, the step `dx`
//! from `x` becoming the structure with the coordinates `q(x) + B(x) dx`.
//!
//! MultiOptPy: no counterpart (its searches are in Cartesian coordinates).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::sync::{Arc, Mutex};

use faer::Col;

use super::surface::{SearchSurface, SurfaceValue};
use crate::coord::system::transform::{back_transform, BackTransformOptions, CoordinateFrame};
use crate::coord::system::CoordinateSystem;
use crate::domain::MolecularSystem;
use crate::error::{MoptError, MoptResult};
use crate::math::linalg::{col_norm, orthonormalize_columns};
use crate::math::projection::build_tr_rot_basis;
use crate::path::points::{PointRequest, StepMap};

/// Largest difference between the coordinates asked for and those reached, relative to the
/// step, beyond [`REACHED`].
const NOT_REACHED: f64 = 1e-2;
/// A difference below this counts as reached whatever the step.
const REACHED: f64 = 1e-4;
/// Curvature given to coordinates that move the whole structure (Hartree per squared unit).
const RIGID_CURVATURE: f64 = 1.0;

/// The words of every error of the coordinates breaking down during a search
/// ([`is_breakdown`]): the command then sets them up anew or falls back to others.
pub const BREAKDOWN: &str = "the coordinates break down";

/// Whether `e` is the coordinates breaking down.
pub fn is_breakdown(e: &MoptError) -> bool {
    e.to_string().contains(BREAKDOWN)
}

fn broken(what: impl std::fmt::Display) -> MoptError {
    MoptError::calc(format!("{BREAKDOWN}: {what}"))
}

struct State {
    coords: Box<dyn CoordinateSystem>,
    known: Vec<(Col<f64>, Col<f64>)>,
}

/// `inner`, a surface of structures, as a surface of their coordinates.
pub struct InternalSearchSurface<'a> {
    inner: &'a dyn SearchSurface,
    reference: MolecularSystem,
    name: String,
    dimension: usize,
    state: Arc<Mutex<State>>,
    notes: Mutex<Vec<String>>,
}

fn lock(state: &Mutex<State>) -> MoptResult<std::sync::MutexGuard<'_, State>> {
    state
        .lock()
        .map_err(|_| MoptError::calc("the coordinates of the search are not available"))
}

/// The structure with the coordinates `q`, from the nearest one known.
fn structure_at(
    state: &State,
    reference: &MolecularSystem,
    q: &Col<f64>,
) -> MoptResult<(Col<f64>, Col<f64>)> {
    let nearest = state
        .known
        .iter()
        .min_by(|a, b| col_norm(&(&a.0 - q)).total_cmp(&col_norm(&(&b.0 - q))))
        .ok_or_else(|| MoptError::calc("no structure of the search is known"))?;
    let step = q - &nearest.0;
    let length = col_norm(&step);
    if length < 1e-14 {
        return Ok(nearest.clone());
    }
    let start = reference.with_flat_coords_bohr(&nearest.1)?;
    let dx = back_transform(
        state.coords.as_ref(),
        &start,
        &step,
        &BackTransformOptions::default(),
    )
    .map_err(broken)?;
    let x = &nearest.1 + dx;
    let structure = reference.with_flat_coords_bohr(&x)?;
    let reached = state
        .coords
        .displacement(&structure, reference)
        .map_err(broken)?;
    let off = col_norm(&(&reached - q));
    if !off.is_finite() || off > NOT_REACHED * length + REACHED {
        return Err(broken(format!(
            "no structure has the coordinates asked for ({}; off by {off:.2e} after a step \
             of {length:.2e})",
            state.coords.name()
        )));
    }
    Ok((reached, x))
}

impl<'a> InternalSearchSurface<'a> {
    /// The surface `inner` in `coords`, set up at `reference`, and `structures` (Cartesian,
    /// Bohr) as vectors of these coordinates relative to `reference`.
    pub fn new(
        inner: &'a dyn SearchSurface,
        mut coords: Box<dyn CoordinateSystem>,
        reference: MolecularSystem,
        structures: &[Col<f64>],
    ) -> MoptResult<(Self, Vec<Col<f64>>)> {
        coords.initialize(&reference)?;
        if coords.is_cartesian() {
            return Err(MoptError::config(
                "Cartesian coordinates need no change of coordinates",
            ));
        }
        let dimension = coords.dim(&reference);
        if dimension > reference.dof() {
            return Err(MoptError::config(format!(
                "the coordinates {} are redundant ({dimension} for {} Cartesian coordinates): a \
                 search needs independent ones (dic, tric, tric-cart)",
                coords.name(),
                reference.dof()
            )));
        }
        let mut known = Vec::with_capacity(structures.len());
        for (k, x) in structures.iter().enumerate() {
            let s = reference.with_flat_coords_bohr(x)?;
            let q = coords.displacement(&s, &reference).map_err(|e| {
                e.context(format!(
                    "structure {} in the coordinates {}",
                    k + 1,
                    coords.name()
                ))
            })?;
            known.push((q, x.clone()));
        }
        let points = known.iter().map(|(q, _)| q.clone()).collect();
        let name = coords.name().to_string();
        Ok((
            Self {
                inner,
                reference,
                name,
                dimension,
                state: Arc::new(Mutex::new(State { coords, known })),
                notes: Mutex::new(Vec::new()),
            },
            points,
        ))
    }

    /// The number of coordinates.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// The structure evaluated last (Cartesian, Bohr).
    pub fn last_structure(&self) -> Option<Col<f64>> {
        lock(&self.state).ok()?.known.last().map(|(_, x)| x.clone())
    }

    /// The name of the coordinates.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The Cartesian coordinates (Bohr) of the point `q`.
    pub fn cartesian(&self, q: &Col<f64>) -> MoptResult<Col<f64>> {
        let state = lock(&self.state)?;
        structure_at(&state, &self.reference, q).map(|(_, x)| x)
    }

    /// A map from points to their Cartesian coordinates that can be kept (a test for
    /// fragments, for example).
    pub fn to_cartesian(&self) -> Arc<dyn Fn(&Col<f64>) -> Option<Col<f64>> + Send + Sync> {
        let state = Arc::clone(&self.state);
        let reference = self.reference.clone();
        Arc::new(move |q: &Col<f64>| {
            let state = state.lock().ok()?;
            structure_at(&state, &reference, q).ok().map(|(_, x)| x)
        })
    }
}

impl SearchSurface for InternalSearchSurface<'_> {
    fn two_surfaces(&self) -> bool {
        self.inner.two_surfaces()
    }

    fn in_order(&self) -> bool {
        self.inner.in_order()
    }

    fn description(&self) -> String {
        format!(
            "{}; the search is in {} coordinates ({})",
            self.inner.description(),
            self.dimension,
            self.name
        )
    }

    fn evaluate(
        &self,
        q: &Col<f64>,
        image: usize,
        request: &PointRequest,
    ) -> MoptResult<SurfaceValue> {
        if q.nrows() != self.dimension {
            return Err(MoptError::config(format!(
                "a point of {} coordinates in a search of {}",
                q.nrows(),
                self.dimension
            )));
        }
        let (reached, x) = {
            let state = lock(&self.state)?;
            let found = structure_at(&state, &self.reference, q)?;
            let structure = self.reference.with_flat_coords_bohr(&found.1)?;
            if state.coords.needs_reinitialization(&structure) {
                return Err(broken(format!(
                    "the coordinates {} ask to be set up anew at this structure",
                    self.name
                )));
            }
            found
        };
        let asked = PointRequest {
            x: x.clone(),
            ..request.clone()
        };
        // Calculated without the lock: the surface behind may take long.
        let v = self.inner.evaluate(&x, image, &asked)?;
        let mut state = lock(&self.state)?;
        let structure = self.reference.with_flat_coords_bohr(&x)?;
        let frame = CoordinateFrame::new(state.coords.as_ref(), &structure).map_err(broken)?;
        let to_q = |g: &Col<f64>| frame.gradient_to_q(g);
        let mut hessian = v.hessian.as_ref().map(|h| frame.hessian_to_q(h));
        if let Some(h) = hessian.as_mut() {
            let rigid = build_tr_rot_basis(&structure, false);
            let b = state.coords.wilson_b(&structure)?;
            let image_of = &b * &rigid;
            let columns: Vec<Col<f64>> = (0..image_of.ncols())
                .map(|c| Col::from_fn(image_of.nrows(), |r| image_of[(r, c)]))
                .collect();
            let stiff = orthonormalize_columns(image_of.nrows(), &columns, 1e-6);
            for c in 0..stiff.ncols() {
                for i in 0..h.nrows() {
                    for j in 0..h.ncols() {
                        h[(i, j)] += RIGID_CURVATURE * stiff[(i, c)] * stiff[(j, c)];
                    }
                }
            }
        }
        let value = SurfaceValue {
            energy: v.energy,
            gradient: to_q(&v.gradient),
            biased_energy: v.biased_energy,
            biased_gradient: to_q(&v.biased_gradient),
            hessian,
        };
        state.known.push((reached, x));
        Ok(value)
    }

    fn structure(&self, q: &Col<f64>, image: usize) -> Option<MolecularSystem> {
        let x = self.cartesian(q).ok()?;
        self.inner.structure(&x, image)
    }

    fn take_notes(&self) -> Vec<String> {
        let mut notes = self.inner.take_notes();
        if let Ok(mut own) = self.notes.lock() {
            notes.append(&mut own);
        }
        notes
    }

    fn begin_iteration(&self, iteration: usize) {
        self.inner.begin_iteration(iteration);
    }
}

/// The structure a step map took its last step from.
pub type LastStart = Arc<Mutex<Option<Col<f64>>>>;

/// The map of Cartesian steps through `coords` (set up at `reference`): the step from `x` to
/// `y` becomes the structure with the coordinates `q(x) + B(x) (y - x)`; with where it took
/// its last step from.
pub fn step_map(
    mut coords: Box<dyn CoordinateSystem>,
    reference: &MolecularSystem,
) -> MoptResult<(StepMap, LastStart)> {
    coords.initialize(reference)?;
    let coords = Arc::new(Mutex::new(coords));
    let template = reference.clone();
    let last: LastStart = Arc::new(Mutex::new(None));
    let seen = Arc::clone(&last);
    let map: StepMap = Arc::new(move |x: &Col<f64>, y: &Col<f64>| -> MoptResult<Col<f64>> {
        let coords = coords
            .lock()
            .map_err(|_| MoptError::calc("the coordinates of the steps are not available"))?;
        if let Ok(mut s) = seen.lock() {
            *s = Some(x.clone());
        }
        let start = template.with_flat_coords_bohr(x)?;
        if coords.needs_reinitialization(&start) {
            return Err(broken(format!(
                "the coordinates {} ask to be set up anew at this structure",
                coords.name()
            )));
        }
        let b = coords.wilson_b(&start).map_err(broken)?;
        let dq = &b * (y - x);
        let dx = back_transform(
            coords.as_ref(),
            &start,
            &dq,
            &BackTransformOptions::default(),
        )
        .map_err(broken)?;
        Ok(x + dx)
    });
    Ok((map, last))
}
