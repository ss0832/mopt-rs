//! DIIS for a chain: the chains of the last iterations are combined so that the combined
//! forces across the chain are smallest, and the step leads to the combination of the
//! chains with their steps.
//!
//! The chain is taken as one vector (all images in a row), its error vector is the force
//! across the chain (which vanishes on the path whatever the step rule is), and the step
//! the relaxation proposes there (along the forces, quasi-Newton, or their mixture) is the
//! relaxation of GDIIS: with coefficients `c` (`sum c = 1`) that minimize `|sum_i c_i F_i|`
//! the next chain is `sum_i c_i (X_i + S_i)` (Csaszar and Pulay, J. Mol. Struct. 1984, 114,
//! 31), accepted by the rules of Farkas and Schlegel (Phys. Chem. Chem. Phys. 2002, 4, 11;
//! [`crate::opt::enhance::diis_core`]); otherwise the proposed steps are taken. It works
//! with every step rule of the chain, because it only combines what that rule proposes.
//!
//! A chain that was redistributed or aligned stays a valid point: its forces were
//! evaluated where it is. The points are forgotten when the number of images changes.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use crate::opt::enhance::diis_core::DiisPoint;
use crate::opt::enhance::gdiis::gdiis_step;

/// Settings of [`ChainDiis`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChainDiisSettings {
    /// Chains kept at most.
    pub points: usize,
    /// Chains needed before the first extrapolation.
    pub min_points: usize,
    /// The extrapolation begins when the largest force of the images that move is below
    /// this (Hartree/Bohr).
    pub start_force: f64,
}

impl Default for ChainDiisSettings {
    fn default() -> Self {
        Self {
            points: 5,
            min_points: 3,
            start_force: 5e-3,
        }
    }
}

/// The chains of the last iterations and their extrapolation.
#[derive(Debug, Clone)]
pub struct ChainDiis {
    settings: ChainDiisSettings,
    points: Vec<DiisPoint>,
    used: usize,
}

fn in_a_row(parts: &[Col<f64>], sign: f64) -> Col<f64> {
    let total: usize = parts.iter().map(Col::nrows).sum();
    let mut all = Col::<f64>::zeros(total);
    let mut at = 0;
    for part in parts {
        for k in 0..part.nrows() {
            all[at + k] = sign * part[k];
        }
        at += part.nrows();
    }
    all
}

impl ChainDiis {
    /// No chains yet.
    pub fn new(settings: ChainDiisSettings) -> Self {
        Self {
            settings: ChainDiisSettings {
                points: settings.points.max(2),
                min_points: settings.min_points.max(2),
                ..settings
            },
            points: Vec::new(),
            used: 0,
        }
    }

    /// Forget the chains.
    pub fn clear(&mut self) {
        self.points.clear();
    }

    /// Number of iterations whose steps were the extrapolated ones.
    pub fn used(&self) -> usize {
        self.used
    }

    /// Record `chain` with the `forces` across it and the `steps` proposed there, and
    /// return the steps to take: the extrapolated ones when the largest force of the
    /// images that move, `largest_force`, is small enough and the extrapolation is
    /// acceptable, else `steps`.
    pub fn steps(
        &mut self,
        chain: &[Col<f64>],
        forces: &[Col<f64>],
        steps: Vec<Col<f64>>,
        largest_force: f64,
    ) -> Vec<Col<f64>> {
        let position = in_a_row(chain, 1.0);
        let proposed = in_a_row(&steps, 1.0);
        if self
            .points
            .last()
            .is_some_and(|last| last.position.nrows() != position.nrows())
        {
            self.points.clear();
        }
        self.points.push(DiisPoint {
            position,
            energy: 0.0,
            // The error vector: minus the force, as a gradient.
            gradient: in_a_row(forces, -1.0),
            step: proposed.clone(),
        });
        if self.points.len() > self.settings.points {
            self.points.remove(0);
        }
        if largest_force > self.settings.start_force || self.points.len() < self.settings.min_points
        {
            return steps;
        }
        let Some(extrapolated) = gdiis_step(&self.points, &proposed) else {
            return steps;
        };
        self.used += 1;
        let mut at = 0;
        steps
            .iter()
            .map(|step| {
                let part = Col::from_fn(step.nrows(), |k| extrapolated[at + k]);
                at += step.nrows();
                part
            })
            .collect()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    /// Two images of one variable each in a quadratic bowl with different curvatures: the
    /// force is `-k (x - x0)`, the proposed step a tenth of it.
    fn bowl(x: &[f64]) -> (Vec<Col<f64>>, Vec<Col<f64>>, Vec<Col<f64>>) {
        let (k, x0) = ([2.0, 0.5], [1.0, -3.0]);
        let one = |v: f64| Col::from_fn(1, |_| v);
        let chain: Vec<_> = x.iter().map(|v| one(*v)).collect();
        let forces: Vec<_> = (0..2).map(|i| one(-k[i] * (x[i] - x0[i]))).collect();
        let steps: Vec<_> = forces.iter().map(|f| f * 0.1).collect();
        (chain, forces, steps)
    }

    #[test]
    fn the_extrapolation_reaches_the_minimum_of_a_quadratic_bowl() {
        let mut diis = ChainDiis::new(ChainDiisSettings {
            start_force: f64::INFINITY,
            ..ChainDiisSettings::default()
        });
        let mut x = [0.0, 0.0];
        let mut plain = [0.0, 0.0];
        for iteration in 0..12 {
            let (chain, forces, steps) = bowl(&x);
            let taken = diis.steps(&chain, &forces, steps, 1.0);
            for i in 0..2 {
                x[i] += taken[i][0];
            }
            let (_, _, steps) = bowl(&plain);
            for i in 0..2 {
                plain[i] += steps[i][0];
            }
            if iteration == 1 {
                // Two chains are not enough yet.
                assert_eq!(diis.used(), 0);
            }
        }
        // The extrapolation was taken where the rules of acceptance allowed it (they turn
        // down steps that leave the direction of the proposed one by too much), and the
        // chain is nearer to the minimum than with the plain steps.
        assert!(diis.used() >= 1, "{}", diis.used());
        let off = |x: &[f64; 2]| (x[0] - 1.0).abs().max((x[1] + 3.0).abs());
        assert!(
            off(&x) < 0.9 * off(&plain),
            "{} against {}",
            off(&x),
            off(&plain)
        );
    }

    #[test]
    fn large_forces_and_a_change_of_the_chain_leave_the_steps_as_they_are() {
        let mut diis = ChainDiis::new(ChainDiisSettings::default());
        for x in [[0.0, 0.0], [0.2, -0.15], [0.36, -0.29]] {
            let (chain, forces, steps) = bowl(&x);
            let taken = diis.steps(&chain, &forces, steps.clone(), 1.0);
            // A force of 1 is above the limit of 5e-3.
            assert_eq!(taken[0][0], steps[0][0]);
        }
        assert_eq!(diis.used(), 0);
        // Another number of images: the chains kept are forgotten.
        let one = |v: f64| Col::from_fn(1, |_| v);
        let chain = vec![one(0.0), one(0.1), one(0.2)];
        let forces = vec![one(1e-4), one(1e-4), one(1e-4)];
        let steps = vec![one(1e-5), one(1e-5), one(1e-5)];
        let taken = diis.steps(&chain, &forces, steps.clone(), 1e-4);
        assert_eq!(taken.len(), 3);
        assert_eq!(taken[1][0], steps[1][0]);
        assert_eq!(diis.used(), 0);
    }
}
