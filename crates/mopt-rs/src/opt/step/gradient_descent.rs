//! Steepest descent with a fixed step factor, plain and mass-weighted.
//!
//! `dx = -delta g` (Cauchy 1847), and `dx_i = -delta g_i / m_i` with the masses of the most
//! abundant isotopes in amu:
//! the Euler step of the mass-weighted steepest-descent path, i.e. of the intrinsic reaction
//! coordinate (Fukui, Acc. Chem. Res. 1981, 14, 363). Both take `delta = 1`; the optimizer's
//! trust radius limits the step length.
//!
//! MultiOptPy: `Optimizer/gradientdescent.py:GradientDescent`, `MassWeightedGradientDescent`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use super::{StepGenerator, StepInput};
use crate::domain::MolecularSystem;
use crate::error::{MoptError, MoptResult};
use crate::parameters::isotope_masses::MOST_ABUNDANT_ISOTOPE_MASS_U;

/// Masses (amu) of the most abundant isotopes of the atoms, as MultiOptPy's `atomic_mass`
/// (the standard atomic weight where the table ends).
pub fn isotope_masses(sys: &MolecularSystem) -> Vec<f64> {
    sys.atoms()
        .map(|a| {
            let el = sys.element(a);
            MOST_ABUNDANT_ISOTOPE_MASS_U
                .get(usize::from(el.atomic_number()))
                .copied()
                .filter(|m| *m > 0.0)
                .unwrap_or_else(|| el.mass_amu())
        })
        .collect()
}

/// Steepest descent `dx = -delta g`.
#[derive(Debug, Clone, Copy)]
pub struct GradientDescent {
    /// Step factor `delta`.
    pub delta: f64,
}

impl Default for GradientDescent {
    fn default() -> Self {
        Self { delta: 1.0 }
    }
}

impl StepGenerator for GradientDescent {
    fn set_step_factor(&mut self, delta: f64) -> bool {
        self.delta = delta;
        true
    }

    fn name(&self) -> &str {
        "gradientdescent"
    }

    fn propose(&mut self, input: &StepInput<'_>) -> MoptResult<Col<f64>> {
        Ok(input.gradient * (-self.delta))
    }
}

/// Mass-weighted steepest descent `dx_i = -delta g_i / m_i` (Cartesian coordinates).
///
/// Without atoms (plain variables) every mass is 1 and the step equals plain steepest descent.
#[derive(Debug, Clone, Copy)]
pub struct MassWeightedGradientDescent {
    /// Step factor `delta`.
    pub delta: f64,
}

impl Default for MassWeightedGradientDescent {
    fn default() -> Self {
        Self { delta: 1.0 }
    }
}

impl StepGenerator for MassWeightedGradientDescent {
    fn set_step_factor(&mut self, delta: f64) -> bool {
        self.delta = delta;
        true
    }

    fn name(&self) -> &str {
        "mwgradientdescent"
    }

    fn propose(&mut self, input: &StepInput<'_>) -> MoptResult<Col<f64>> {
        let g = input.gradient;
        let Some(sys) = input.system else {
            return Ok(g * (-self.delta));
        };
        if !input.cartesian_coordinates || g.nrows() != sys.dof() {
            return Err(MoptError::config(
                "mass-weighted steepest descent needs Cartesian coordinates (-coord cart)",
            ));
        }
        let masses = isotope_masses(sys);
        Ok(Col::from_fn(g.nrows(), |i| {
            -self.delta * g[i] / masses[i / 3]
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{positions, Bohr, Element, MolecularSystem, SpinCharge};
    use crate::opt::stepper::plain_input;
    use faer::Mat;

    #[test]
    fn steps_are_the_negative_gradient_divided_by_the_masses() {
        let h = Mat::identity(6, 6);
        let g = Col::from_fn(6, |i| 0.1 * (i as f64 + 1.0));
        let x = Col::zeros(6);
        let mut input = plain_input(0, &x, 0.0, &g, &h, None, 0);
        let s = GradientDescent::default().propose(&input).unwrap();
        assert!((0..6).all(|i| s[i] == -g[i]));
        // No atoms: unit masses.
        let s = MassWeightedGradientDescent::default()
            .propose(&input)
            .unwrap();
        assert!((0..6).all(|i| s[i] == -g[i]));
        let sys = MolecularSystem::new(
            vec![Element::H, Element::O],
            &positions::<Bohr>(&[[0.0, 0.0, 0.0], [1.8, 0.0, 0.0]]),
            SpinCharge::default(),
            None,
        )
        .unwrap();
        input.system = Some(&sys);
        let s = MassWeightedGradientDescent::default()
            .propose(&input)
            .unwrap();
        let (mh, mo) = (1.00782503223, 15.99491461957);
        assert!((s[0] + g[0] / mh).abs() < 1e-15 && (s[5] + g[5] / mo).abs() < 1e-15);
        input.cartesian_coordinates = false;
        assert!(MassWeightedGradientDescent::default()
            .propose(&input)
            .is_err());
    }
}
