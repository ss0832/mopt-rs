//! The constraints of the images of a chain (`-fix`, `-pc`, `-gfix`, `-uc` in `nebmain`).
//!
//! Every image holds its own resolved constraints ([`Constraint::resolve`]): an explicit value
//! applies to all images, and a value "at the start" is the value of the image in the first
//! chain, resolved once. An image that the chain puts elsewhere (the images are put back
//! along the chain, images are added or taken out, the chain is replaced by an observer)
//! holds the values interpolated at its fraction of the length of the chain it came from:
//! targets linearly (angle-like ones along the shorter way round), positions of frozen atoms
//! and oriented groups linearly, and the lab-frame direction of `atoms_pair` constraints from
//! the nearer image.
//!
//! The surface of the chain takes the constraint directions out of the gradient of every
//! image and projects its Hessian as the optimizer does, `P (H - C) P + s Q Q^T`
//! ([`ConstraintProjection`]); [`KeepConstraints`](super::KeepConstraints) puts the images
//! back onto them after every iteration.
//!
//! MultiOptPy: no counterpart (new in mopt-rs; MultiOptPy's `nebmain` takes the constraint
//! directions out of the gradients of the images and holds nothing).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::f64::consts::PI;
use std::sync::{Arc, Mutex};

use faer::{Col, Mat};

use crate::coord::system::{Cartesian, CoordinateFrame};
use crate::domain::MolecularSystem;
use crate::error::{MoptError, MoptResult, ResultExt};
use crate::opt::constrained::ConstraintProjection;
use crate::path::band::{path_lengths, ChainKind};
use crate::pes::constraint::{Constraint, ConstraintAnchor, ConstraintKind, ConstraintSet};

/// The resolved constraints of one image and their set.
#[derive(Debug, Clone)]
struct ImageConstraints {
    resolved: Vec<Constraint>,
    set: ConstraintSet,
}

/// The constraints of every image of a chain of structures.
#[derive(Debug, Clone)]
pub struct ChainConstraints {
    template: MolecularSystem,
    images: Vec<ImageConstraints>,
}

/// [`ChainConstraints`] shared by the surface of a chain (which projects with them) and the
/// observer that restores the images ([`super::KeepConstraints`]).
pub type SharedChainConstraints = Arc<Mutex<ChainConstraints>>;

impl ChainConstraints {
    /// `constraints` resolved at every image of `chain` (structures with the atoms of
    /// `template`). Eigenvector constraints are refused: a chain has no Hessian of its own
    /// images to resolve them at.
    pub fn new(
        constraints: &[Constraint],
        template: &MolecularSystem,
        chain: &[Col<f64>],
    ) -> MoptResult<Self> {
        if constraints
            .iter()
            .any(|c| matches!(c.kind, ConstraintKind::EigenVector(_)) && !c.is_resolved())
        {
            return Err(MoptError::unsupported(
                "-pc eigvec: the direction of a Hessian eigenvector is resolved at the first \
                 Hessian of an optimization; the images of a chain have none to resolve it at",
            ));
        }
        let mut images = Vec::with_capacity(chain.len());
        for (k, point) in chain.iter().enumerate() {
            let sys = template.with_flat_coords_bohr(point)?;
            let resolved = constraints
                .iter()
                .map(|c| c.resolve(&sys, None))
                .collect::<MoptResult<Vec<_>>>()
                .context(format!("the constraints of image {k}"))?;
            images.push(ImageConstraints::of(resolved, &sys)?);
        }
        Ok(Self {
            template: template.clone(),
            images,
        })
    }

    /// Images.
    pub fn len(&self) -> usize {
        self.images.len()
    }

    /// Whether there are no images.
    pub fn is_empty(&self) -> bool {
        self.images.is_empty()
    }

    /// Scalar constraints of an image.
    pub fn per_image(&self) -> usize {
        self.images.first().map_or(0, |i| i.set.len())
    }

    /// Whether the constraints pin the structures in space (frozen atoms, Cartesian
    /// components, orientations, fixed directions): then the images are not aligned.
    pub fn breaks_rigid_body(&self) -> bool {
        self.images.iter().any(|i| i.set.breaks_rigid_body())
    }

    /// The resolved constraints of image `image` (what a follow-up run from it holds).
    pub fn resolved(&self, image: usize) -> Option<&[Constraint]> {
        self.images.get(image).map(|i| i.resolved.as_slice())
    }

    /// The set of image `image`.
    pub fn set(&self, image: usize) -> Option<&ConstraintSet> {
        self.images.get(image).map(|i| &i.set)
    }

    /// The images `dropped` (sorted numbers) taken out.
    pub fn drop_images(&mut self, dropped: &[usize]) {
        for &image in dropped.iter().rev() {
            if image < self.images.len() {
                self.images.remove(image);
            }
        }
    }

    /// The constraints of the images of `after`, a chain made from `before` (whose images
    /// hold the constraints now): every image holds the values interpolated at its fraction
    /// of the length of its chain, in `before`.
    pub fn follow(&mut self, before: &[Col<f64>], after: &[Col<f64>]) -> MoptResult<()> {
        if before.len() != self.images.len() {
            return Err(MoptError::config(format!(
                "constraints of {} images for a chain of {}",
                self.images.len(),
                before.len()
            )));
        }
        let resolved: Vec<&[Constraint]> =
            self.images.iter().map(|i| i.resolved.as_slice()).collect();
        let followed = follow_resolved(&resolved, before, after, ChainKind::Atoms)?;
        let mut images = Vec::with_capacity(after.len());
        for (k, (point, resolved)) in after.iter().zip(followed).enumerate() {
            let sys = self.template.with_flat_coords_bohr(point)?;
            images.push(
                ImageConstraints::of(resolved, &sys)
                    .context(format!("the constraints of image {k}"))?,
            );
        }
        self.images = images;
        Ok(())
    }

    /// `chain` put back onto the constraints of its images.
    pub fn restore(&self, chain: &[Col<f64>]) -> MoptResult<Vec<Col<f64>>> {
        if chain.len() != self.images.len() {
            return Err(MoptError::config(format!(
                "constraints of {} images for a chain of {}",
                self.images.len(),
                chain.len()
            )));
        }
        chain
            .iter()
            .zip(&self.images)
            .enumerate()
            .map(|(k, (point, image))| {
                let sys = self.template.with_flat_coords_bohr(point)?;
                Ok(image
                    .set
                    .restore(&sys)
                    .context(format!("image {k}"))?
                    .coords_flat_bohr())
            })
            .collect()
    }
}

impl ImageConstraints {
    fn of(resolved: Vec<Constraint>, sys: &MolecularSystem) -> MoptResult<Self> {
        let set = ConstraintSet::new(&resolved, sys, None)?;
        Ok(Self { resolved, set })
    }
}

/// The gradient of image values at `sys` with the directions of `set` taken out, and the
/// Hessian projected for the steps as the optimizer projects it, `P (H - C) P + s Q Q^T`
/// (constraint curvature `C` from the multipliers of the gradient, stiffness `s` along the
/// constrained directions `Q`).
///
/// MultiOptPy: `Constraint/constraint_condition.py:ProjectOutConstrain` (the gradient; the
/// Hessian as in `opt::constrained`).
pub fn project_image(
    set: &ConstraintSet,
    sys: &MolecularSystem,
    gradient: &Col<f64>,
    hessian: Option<Mat<f64>>,
) -> MoptResult<(Col<f64>, Option<Mat<f64>>)> {
    let cf = set.frame(sys, gradient)?;
    let projected = cf.project_gradient(gradient);
    let hessian = match hessian {
        Some(h) => {
            let frame = CoordinateFrame::new(&Cartesian, sys)?;
            Some(ConstraintProjection::new(&frame, &cf)?.hessian(&h, &cf.curvature))
        }
        None => None,
    };
    Ok((projected, hessian))
}

/// The resolved constraints of the images of `after`, a chain made from `before` whose images
/// hold `resolved`: every image holds the values interpolated at its fraction of the length of
/// its chain, in `before` (chains of the kind `kind`).
///
/// MultiOptPy: no counterpart (new in mopt-rs).
pub(crate) fn follow_resolved(
    resolved: &[&[Constraint]],
    before: &[Col<f64>],
    after: &[Col<f64>],
    kind: ChainKind,
) -> MoptResult<Vec<Vec<Constraint>>> {
    if before.len() < 2 || after.is_empty() || resolved.len() != before.len() {
        return Err(MoptError::config(
            "the constraints of a chain follow a chain of at least two images",
        ));
    }
    let from = fractions_of(before, kind);
    let to = fractions_of(after, kind);
    let mut out = Vec::with_capacity(after.len());
    for s in &to {
        // The segment of `before` that holds the fraction `s`.
        let j = from
            .windows(2)
            .position(|w| *s <= w[1])
            .unwrap_or(from.len() - 2);
        let span = from[j + 1] - from[j];
        let t = if span > 0.0 {
            ((s - from[j]) / span).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let (a, b) = (resolved[j], resolved[j + 1]);
        out.push(
            a.iter()
                .zip(b)
                .map(|(a, b)| interpolate(a, b, t))
                .collect::<MoptResult<Vec<_>>>()?,
        );
    }
    Ok(out)
}

/// Fractions of the length of `chain` (of the kind `kind`) at its images (0 to 1; evenly
/// spaced when the chain has no length).
pub(crate) fn fractions_of(chain: &[Col<f64>], kind: ChainKind) -> Vec<f64> {
    let lengths = path_lengths(chain, kind);
    let total = lengths.last().copied().unwrap_or(0.0);
    let n = chain.len().max(2) - 1;
    lengths
        .iter()
        .enumerate()
        .map(|(k, l)| {
            if total > 0.0 {
                l / total
            } else {
                k as f64 / n as f64
            }
        })
        .collect()
}

/// The period of the residual of a constraint kind (`None`: not periodic).
fn period_of(kind: &ConstraintKind) -> Option<f64> {
    match kind {
        ConstraintKind::Dihedral(_) => Some(2.0 * PI),
        ConstraintKind::Function(f) => f.period,
        _ => None,
    }
}

/// The constraint between `a` and `b` (the same constraint resolved at two images) at
/// `t` from `a` (0) to `b` (1).
pub(crate) fn interpolate(a: &Constraint, b: &Constraint, t: f64) -> MoptResult<Constraint> {
    if a.kind != b.kind {
        return Err(MoptError::config(
            "the constraints of two images of a chain differ in kind",
        ));
    }
    let nearer = if t < 0.5 { a } else { b };
    let mut out = a.clone();
    out.target = match (a.target, b.target) {
        (Some(ta), Some(tb)) => {
            if matches!(a.anchor, Some(ConstraintAnchor::Direction(_))) {
                // A target `v . x` belongs to the direction of its image.
                nearer.target
            } else {
                let mut d = tb - ta;
                if let Some(p) = period_of(&a.kind) {
                    d -= p * (d / p).round();
                }
                Some(ta + t * d)
            }
        }
        (target, _) => target,
    };
    out.anchor = match (&a.anchor, &b.anchor) {
        (Some(ConstraintAnchor::Positions(pa)), Some(ConstraintAnchor::Positions(pb)))
            if pa.len() == pb.len() =>
        {
            Some(ConstraintAnchor::Positions(
                pa.iter()
                    .zip(pb)
                    .map(|(p, q)| std::array::from_fn(|c| p[c] + t * (q[c] - p[c])))
                    .collect(),
            ))
        }
        _ => nearer.anchor.clone(),
    };
    Ok(out)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::domain::{positions, AtomId, Bohr, Element, SpinCharge};

    fn template() -> MolecularSystem {
        MolecularSystem::new(
            vec![Element::C; 3],
            &positions::<Bohr>(&[[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 4.0, 0.0]]),
            SpinCharge::default(),
            None,
        )
        .unwrap()
    }

    fn point(d: f64, y: f64) -> Col<f64> {
        Col::from_fn(9, |k| [0.0, 0.0, 0.0, d, 0.0, 0.0, 0.0, y, 0.0][k])
    }

    #[test]
    fn values_at_the_start_are_those_of_every_image_and_follow_by_fraction() {
        let t = template();
        let chain = vec![point(4.0, 4.0), point(5.0, 6.0), point(6.0, 8.0)];
        let bond = Constraint::at_start(ConstraintKind::Distance(
            vec![AtomId::new(0)],
            vec![AtomId::new(1)],
        ));
        let manual = Constraint::with_target(
            ConstraintKind::Distance(vec![AtomId::new(0)], vec![AtomId::new(2)]),
            5.0,
        );
        let mut c = ChainConstraints::new(&[bond, manual], &t, &chain).unwrap();
        let targets = |c: &ChainConstraints| -> Vec<(f64, f64)> {
            (0..c.len())
                .map(|k| {
                    let r = c.resolved(k).unwrap();
                    (r[0].target.unwrap(), r[1].target.unwrap())
                })
                .collect()
        };
        assert_eq!(targets(&c), vec![(4.0, 5.0), (5.0, 5.0), (6.0, 5.0)]);
        // The restored chain holds them.
        let back = c.restore(&chain).unwrap();
        for (k, p) in back.iter().enumerate() {
            let s = t.with_flat_coords_bohr(p).unwrap();
            let r = c.set(k).unwrap().residuals(&s).unwrap();
            assert!(r.iter().all(|v| v.abs() < 1e-10), "{r:?}");
        }
        // Five images on the same line: the targets of the first one at the fractions.
        let after = vec![
            point(4.0, 4.0),
            point(4.5, 5.0),
            point(5.0, 6.0),
            point(5.5, 7.0),
            point(6.0, 8.0),
        ];
        c.follow(&chain, &after).unwrap();
        let found = targets(&c);
        let want = [4.0, 4.5, 5.0, 5.5, 6.0];
        for (f, w) in found.iter().zip(want) {
            assert!(
                (f.0 - w).abs() < 1e-12 && (f.1 - 5.0).abs() < 1e-15,
                "{found:?}"
            );
        }
    }

    #[test]
    fn angles_go_the_shorter_way_and_eigenvectors_are_refused() {
        let k = ConstraintKind::Dihedral([AtomId::new(0); 4]);
        let a = Constraint::with_target(k.clone(), 3.0);
        let b = Constraint::with_target(k, -3.0);
        let mid = interpolate(&a, &b, 0.5).unwrap().target.unwrap();
        assert!((mid.abs() - PI).abs() < 1e-12, "{mid}");
        let eig = Constraint::at_start(ConstraintKind::EigenVector(0));
        let err = ChainConstraints::new(&[eig], &template(), &[point(4.0, 4.0)])
            .unwrap_err()
            .to_string();
        assert!(err.contains("eigvec"), "{err}");
    }
}
