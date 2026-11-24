//! Point charges of an electronic embedding: a calculator that takes them
//! ([`crate::pes::Calculator::evaluate_in_charges`]) computes the system in their field, with
//! the interaction of the system with the charges in its energy (not that of the charges among
//! themselves) and the gradient of that energy with respect to the positions of the charges.
//! ONIOM's electronic embedding puts the model systems in the charges of the atoms around
//! them.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use crate::domain::{Element, EvalOutput};
use crate::error::{MoptError, MoptResult};

/// Fixed point charges: positions (Bohr), charges (e), and the element each one stands for
/// (semiempirical methods give a charge the width of that element; one of no element is a
/// nitrogen there, as in xtb).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PointCharges {
    /// Positions (Bohr).
    pub positions_bohr: Vec<[f64; 3]>,
    /// Charges (e).
    pub charges: Vec<f64>,
    /// The element of every charge, if it has one.
    pub elements: Vec<Option<Element>>,
}

impl PointCharges {
    /// Charges at `positions_bohr` with the elements `elements`, checked: as many of each,
    /// finite.
    pub fn new(
        positions_bohr: Vec<[f64; 3]>,
        charges: Vec<f64>,
        elements: Vec<Option<Element>>,
    ) -> MoptResult<Self> {
        if positions_bohr.len() != charges.len() || elements.len() != charges.len() {
            return Err(MoptError::config(format!(
                "point charges: {} positions, {} charges, {} elements",
                positions_bohr.len(),
                charges.len(),
                elements.len()
            )));
        }
        if charges.iter().any(|q| !q.is_finite())
            || positions_bohr.iter().flatten().any(|x| !x.is_finite())
        {
            return Err(MoptError::config(
                "point charges: a charge or a position that is not finite",
            ));
        }
        Ok(Self {
            positions_bohr,
            charges,
            elements,
        })
    }

    /// Number of charges.
    pub fn len(&self) -> usize {
        self.charges.len()
    }

    /// Whether there are none.
    pub fn is_empty(&self) -> bool {
        self.charges.is_empty()
    }

    /// The Coulomb energy `sum_iK q_i q_K / r_iK` of the point charges `q` at `positions`
    /// (Bohr) with these charges, its gradient with respect to `positions` (flat), and with
    /// respect to the positions of these charges.
    pub fn coulomb_with(
        &self,
        positions: &[[f64; 3]],
        q: &[f64],
    ) -> MoptResult<(f64, Vec<f64>, Vec<[f64; 3]>)> {
        if positions.len() != q.len() {
            return Err(MoptError::config(format!(
                "point charges: {} positions for {} charges",
                positions.len(),
                q.len()
            )));
        }
        let mut energy = 0.0;
        let mut gradient = vec![0.0; 3 * positions.len()];
        let mut of_charges = vec![[0.0; 3]; self.len()];
        for (i, (xi, qi)) in positions.iter().zip(q).enumerate() {
            if *qi == 0.0 {
                continue;
            }
            for (k, (xk, qk)) in self.positions_bohr.iter().zip(&self.charges).enumerate() {
                let d = [xi[0] - xk[0], xi[1] - xk[1], xi[2] - xk[2]];
                let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                if r2 < 1e-12 {
                    return Err(MoptError::calc(format!(
                        "point charges: charge {} sits on atom {}",
                        k + 1,
                        i + 1
                    )));
                }
                let r = r2.sqrt();
                let e = qi * qk / r;
                energy += e;
                // d(1/r)/dx_i = -d / r^3.
                let f = -e / r2;
                for c in 0..3 {
                    gradient[3 * i + c] += f * d[c];
                    of_charges[k][c] -= f * d[c];
                }
            }
        }
        Ok((energy, gradient, of_charges))
    }
}

/// An evaluation in the field of point charges.
#[derive(Debug, Clone)]
pub struct EmbeddedOutput {
    /// The energy (with the interaction with the charges), the gradient of the system, and
    /// its Hessian at fixed charges if asked for.
    pub output: EvalOutput,
    /// The gradient of the energy with respect to the positions of the charges (Hartree/Bohr).
    pub charge_gradient_bohr: Vec<[f64; 3]>,
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    #[test]
    fn the_coulomb_energy_and_its_gradients_agree_with_differences() {
        let charges = PointCharges::new(
            vec![[0.3, -1.2, 2.0], [-2.5, 0.4, 0.1]],
            vec![0.4, -0.7],
            vec![Some(Element::O), None],
        )
        .unwrap();
        let x = vec![[0.0, 0.0, 0.0], [1.4, 0.2, -0.3]];
        let q = [-0.3, 0.25];
        let (e, g, gk) = charges.coulomb_with(&x, &q).unwrap();
        let h = 1e-6;
        for i in 0..2 {
            for c in 0..3 {
                let mut up = x.clone();
                up[i][c] += h;
                let mut down = x.clone();
                down[i][c] -= h;
                let fd = (charges.coulomb_with(&up, &q).unwrap().0
                    - charges.coulomb_with(&down, &q).unwrap().0)
                    / (2.0 * h);
                assert!((fd - g[3 * i + c]).abs() < 1e-8, "{fd} {}", g[3 * i + c]);
            }
        }
        for (k, of_charge) in gk.iter().enumerate() {
            for (c, analytic) in of_charge.iter().enumerate() {
                let mut up = charges.clone();
                up.positions_bohr[k][c] += h;
                let mut down = charges.clone();
                down.positions_bohr[k][c] -= h;
                let fd = (up.coulomb_with(&x, &q).unwrap().0
                    - down.coulomb_with(&x, &q).unwrap().0)
                    / (2.0 * h);
                assert!((fd - analytic).abs() < 1e-8);
            }
        }
        assert!(e.is_finite());
        assert!(PointCharges::new(vec![[0.0; 3]], vec![], vec![]).is_err());
    }
}
