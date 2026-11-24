//! LINCS, the linear constraint solver for bond lengths (Hess, Bekker, Berendsen, and
//! Fraaije, J. Comput. Chem. 1997, 18, 1463), as an alternative to SHAKE/RATTLE
//! ([`super::constraints::Rattle`]) when every constraint is the distance of two atoms.
//!
//! After an unconstrained drift from `x_old` to `x`, with the directions `n_i` of the bonds
//! at `x_old` (`B` the matrix of the rows `+n_i` at atom `a_i` and `-n_i` at `b_i`, `W` the
//! inverse masses) and `S = diag(1 / sqrt(w_a + w_b))`, the positions are moved by
//! `-W B^T S (I - A)^-1 S (B x - d)`, where `I - A = S B W B^T S` and `(I - A)^-1` is the
//! series `I + A + ... + A^order` (eq. 12-14 of the paper). The rotation of the bonds
//! lengthens them to second order; the correction projects them again on
//! `p_i = sqrt(2 d_i^2 - l_i^2)` with the current lengths `l_i` (eq. 15), `iterations` times.
//! The velocities take the displacement over the time of the drift, and after a kick they
//! are projected with the same series at the current directions (`B v = 0`). The defaults,
//! an expansion of order 4 and one correction, are those of the paper for molecular
//! dynamics (and of GROMACS, `lincs-order = 4`, `lincs-iter = 1`).
//!
//! The corrections of the paper stop after a fixed number: the bond lengths keep the residual
//! of the truncated series (2.7e-5 Bohr for three coupled bonds with order 4 and one
//! correction). Every hard constraint is held to 1e-10 Bohr
//! ([`crate::pes::constraint::GUARANTEE`]), so the correction is repeated after those asked
//! for until the largest residual is below 1e-12 or no longer decreases (at most 200 more;
//! each costs one series solve, with the directions of the step kept); a bond left above
//! 1e-10 is an error naming it. Where the series converges quickly a few corrections suffice
//! and LINCS keeps its cost advantage over SHAKE; the test
//! `tests::the_residual_of_lincs_falls_with_its_order_and_corrections` records the counts.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use crate::domain::MolecularSystem;
use crate::error::{MoptError, MoptResult};
use crate::pes::constraint::{Constraint, ConstraintKind, GUARANTEE};

/// Default order of the expansion.
pub const DEFAULT_ORDER: usize = 4;
/// Default number of corrections for the rotation of the bonds.
pub const DEFAULT_ITERATIONS: usize = 1;
/// Residual (Bohr) the corrections go on to after the corrections asked for.
const TARGET: f64 = 1e-12;
/// Corrections at most beyond those asked for.
const MAX_CORRECTIONS: usize = 200;

/// A bond held at its length: atoms `a`, `b` (0-based), length (Bohr).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bond {
    /// First atom.
    pub a: usize,
    /// Second atom.
    pub b: usize,
    /// Length (Bohr).
    pub length: f64,
}

/// The LINCS solver of a set of bonds.
#[derive(Debug, Clone)]
pub struct Lincs {
    bonds: Vec<Bond>,
    /// For each bond, the bonds that share an atom with it: (other bond, shared atom,
    /// product of the signs of the shared atom in the two rows).
    couplings: Vec<Vec<(usize, usize, f64)>>,
    order: usize,
    iterations: usize,
}

fn delta(x: &Col<f64>, a: usize, b: usize) -> [f64; 3] {
    std::array::from_fn(|c| x[3 * a + c] - x[3 * b + c])
}

fn norm(d: [f64; 3]) -> f64 {
    libm::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

impl Lincs {
    /// The solver of `bonds` with the expansion of `order` and `iterations` corrections.
    pub fn new(bonds: Vec<Bond>, order: usize, iterations: usize) -> MoptResult<Self> {
        if bonds
            .iter()
            .any(|b| b.a == b.b || b.length.is_nan() || b.length <= 0.0)
        {
            return Err(MoptError::config(
                "a LINCS bond joins two different atoms at a positive length",
            ));
        }
        let mut couplings = vec![Vec::new(); bonds.len()];
        for (i, p) in bonds.iter().enumerate() {
            for (j, q) in bonds.iter().enumerate() {
                if i == j {
                    continue;
                }
                for (atom, si) in [(p.a, 1.0), (p.b, -1.0)] {
                    for (other, sj) in [(q.a, 1.0), (q.b, -1.0)] {
                        if atom == other {
                            couplings[i].push((j, atom, si * sj));
                        }
                    }
                }
            }
        }
        Ok(Self {
            bonds,
            couplings,
            order,
            iterations,
        })
    }

    /// The bonds of `constraints` (distances of two single atoms, at their targets or at
    /// their lengths in `start`); `None` when any constraint is something else.
    pub fn bonds_of(constraints: &[Constraint], start: &MolecularSystem) -> Option<Vec<Bond>> {
        let x = start.coords_flat_bohr();
        constraints
            .iter()
            .map(|c| match &c.kind {
                ConstraintKind::Distance(a, b) if a.len() == 1 && b.len() == 1 => {
                    let (a, b) = (a[0].index(), b[0].index());
                    Some(Bond {
                        a,
                        b,
                        length: c.target.unwrap_or_else(|| norm(delta(&x, a, b))),
                    })
                }
                _ => None,
            })
            .collect()
    }

    /// Number of bonds.
    pub fn len(&self) -> usize {
        self.bonds.len()
    }

    /// Whether there are none.
    pub fn is_empty(&self) -> bool {
        self.bonds.is_empty()
    }

    /// Largest deviation of a bond from its length (Bohr).
    pub fn max_residual(&self, x: &Col<f64>) -> f64 {
        self.bonds
            .iter()
            .map(|b| (norm(delta(x, b.a, b.b)) - b.length).abs())
            .fold(0.0, f64::max)
    }

    /// `(I - A)^-1 rhs` by the series of `order` terms, for the unit directions `n` and the
    /// scales `s`.
    fn solve(&self, n: &[[f64; 3]], s: &[f64], w: &[f64], rhs: &[f64]) -> Vec<f64> {
        let k = self.bonds.len();
        // A_ij = -s_i s_j (sign) w_c (n_i . n_j) for bonds i, j that share atom c.
        let coefficient = |i: usize, j: usize, atom: usize, sign: f64| {
            -s[i] * s[j] * sign * w[3 * atom] * dot(n[i], n[j])
        };
        let mut sol = rhs.to_vec();
        let mut term = rhs.to_vec();
        for _ in 0..self.order {
            let next: Vec<f64> = (0..k)
                .map(|i| {
                    self.couplings[i]
                        .iter()
                        .map(|&(j, atom, sign)| coefficient(i, j, atom, sign) * term[j])
                        .sum()
                })
                .collect();
            for (s, t) in sol.iter_mut().zip(&next) {
                *s += t;
            }
            term = next;
        }
        sol
    }

    /// Directions and scales of the rows at `x`.
    fn rows(&self, x: &Col<f64>, w: &[f64]) -> MoptResult<(Vec<[f64; 3]>, Vec<f64>)> {
        let mut n = Vec::with_capacity(self.bonds.len());
        let mut s = Vec::with_capacity(self.bonds.len());
        for b in &self.bonds {
            let d = delta(x, b.a, b.b);
            let l = norm(d);
            let inverse = w[3 * b.a] + w[3 * b.b];
            if l <= 0.0 || inverse <= 0.0 {
                return Err(MoptError::conv(
                    "LINCS: a bond of zero length or between two held atoms",
                ));
            }
            n.push(d.map(|c| c / l));
            s.push(1.0 / libm::sqrt(inverse));
        }
        Ok((n, s))
    }

    /// `x -= W B^T S sol`.
    fn apply(&self, x: &mut Col<f64>, n: &[[f64; 3]], s: &[f64], w: &[f64], sol: &[f64]) {
        for (i, b) in self.bonds.iter().enumerate() {
            let f = s[i] * sol[i];
            for c in 0..3 {
                x[3 * b.a + c] -= w[3 * b.a + c] * n[i][c] * f;
                x[3 * b.b + c] += w[3 * b.b + c] * n[i][c] * f;
            }
        }
    }

    /// The positions `x` (after a drift of time `h` from `x_old`) put back on the bonds,
    /// and the velocities `v` changed by the same displacement over `h`.
    pub fn constrain(
        &self,
        x_old: &Col<f64>,
        x: &mut Col<f64>,
        v: &mut Col<f64>,
        h: f64,
        w: &[f64],
    ) -> MoptResult<()> {
        if self.is_empty() {
            return Ok(());
        }
        let free = x.clone();
        let (n, s) = self.rows(x_old, w)?;
        let rhs: Vec<f64> = self
            .bonds
            .iter()
            .enumerate()
            .map(|(i, b)| s[i] * (dot(n[i], delta(x, b.a, b.b)) - b.length))
            .collect();
        let sol = self.solve(&n, &s, w, &rhs);
        self.apply(x, &n, &s, w, &sol);
        let correct = |x: &mut Col<f64>| {
            let rhs: Vec<f64> = self
                .bonds
                .iter()
                .enumerate()
                .map(|(i, b)| {
                    // The component along the old direction is shortened by d - p, so that
                    // the bond, lengthened to l by its rotation, gets its length back:
                    // p^2 + (l^2 - d^2) = d^2.
                    let l = norm(delta(x, b.a, b.b));
                    let p = libm::sqrt((2.0 * b.length * b.length - l * l).max(0.0));
                    s[i] * (b.length - p)
                })
                .collect();
            let sol = self.solve(&n, &s, w, &rhs);
            self.apply(x, &n, &s, w, &sol);
        };
        for _ in 0..self.iterations {
            correct(x);
        }
        // The corrections of the paper end where the truncated series leaves the bonds; more
        // of them go on to the target (their fixed point holds every bond at its length), for
        // as long as the largest residual decreases, so that every bond holds to GUARANTEE.
        let mut worst = self.max_residual(x);
        let mut extra = 0;
        while worst > TARGET && extra < MAX_CORRECTIONS {
            let before = x.clone();
            correct(x);
            extra += 1;
            let now = self.max_residual(x);
            if now >= worst {
                *x = before;
                break;
            }
            worst = now;
        }
        if worst > GUARANTEE {
            let (k, b) = self
                .bonds
                .iter()
                .enumerate()
                .max_by(|(_, p), (_, q)| {
                    let r = |b: &Bond| (norm(delta(x, b.a, b.b)) - b.length).abs();
                    r(p).total_cmp(&r(q))
                })
                .map_or((0, self.bonds[0]), |(k, b)| (k, *b));
            return Err(MoptError::conv(format!(
                "LINCS: the bond {k} ({}-{}) is off by {worst:.3e} Bohr after {} corrections, \
                 above the {GUARANTEE:.0e} hard constraints are held to (a smaller time step, \
                 or SHAKE)",
                b.a + 1,
                b.b + 1,
                self.iterations + extra
            )));
        }
        if h > 0.0 {
            for k in 0..x.nrows() {
                v[k] += (x[k] - free[k]) / h;
            }
        }
        Ok(())
    }

    /// The velocities without components along the bonds at `x` (`B v = 0`).
    pub fn project_velocities(&self, x: &Col<f64>, v: &mut Col<f64>, w: &[f64]) -> MoptResult<()> {
        if self.is_empty() {
            return Ok(());
        }
        let (n, s) = self.rows(x, w)?;
        let rhs: Vec<f64> = self
            .bonds
            .iter()
            .enumerate()
            .map(|(i, b)| {
                let dv: [f64; 3] = std::array::from_fn(|c| v[3 * b.a + c] - v[3 * b.b + c]);
                s[i] * dot(n[i], dv)
            })
            .collect();
        let sol = self.solve(&n, &s, w, &rhs);
        self.apply(v, &n, &s, w, &sol);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_rotating_bond_keeps_its_length_and_its_rate() {
        // Two hydrogens 1.4 Bohr apart moving sideways: the free step lengthens the bond.
        let w = vec![1.0 / 1837.0; 6];
        let lincs = match Lincs::new(
            vec![Bond {
                a: 0,
                b: 1,
                length: 1.4,
            }],
            4,
            2,
        ) {
            Ok(l) => l,
            Err(e) => panic!("{e}"),
        };
        let x_old = Col::from_fn(6, |k| if k == 3 { 1.4 } else { 0.0 });
        let mut v = Col::from_fn(6, |k| match k {
            1 => 1e-3,
            4 => -1e-3,
            _ => 0.0,
        });
        let h = 20.0;
        let mut x = &x_old + &v * h;
        assert!(lincs.max_residual(&x) > 1e-4);
        if let Err(e) = lincs.constrain(&x_old, &mut x, &mut v, h, &w) {
            panic!("{e}");
        }
        assert!(lincs.max_residual(&x) < 1e-6, "{}", lincs.max_residual(&x));
        if let Err(e) = lincs.project_velocities(&x, &mut v, &w) {
            panic!("{e}");
        }
        let d = delta(&x, 0, 1);
        let dv: [f64; 3] = std::array::from_fn(|c| v[c] - v[3 + c]);
        assert!(dot(d, dv).abs() < 1e-14);
    }

    #[test]
    fn the_residual_of_lincs_falls_with_its_order_and_corrections() {
        // Three bonds of a triangle of hydrogens, coupled through the shared atoms, after a
        // drift that turns and stretches them. A lone bond is held to the rounding floor by
        // one correction; coupled bonds keep the residual of the truncated series, which
        // falls with the order of the expansion and the number of corrections and stays
        // far above the floor that SHAKE reaches with the defaults (order 4, one correction).
        let w = vec![1.0 / 1837.0; 9];
        let x_old = Col::from_fn(9, |k| [0.0, 0.0, 0.0, 1.4, 0.0, 0.0, 0.7, 1.212, 0.0][k]);
        let v0 = Col::from_fn(9, |k| {
            [0.0, 1e-3, 3e-4, -1e-3, 0.0, 5e-4, 4e-4, -2e-4, -6e-4][k]
        });
        let h = 20.0;
        let bonds = vec![
            Bond {
                a: 0,
                b: 1,
                length: 1.4,
            },
            Bond {
                a: 1,
                b: 2,
                length: 1.4,
            },
            Bond {
                a: 2,
                b: 0,
                length: 1.4,
            },
        ];
        let residual = |order: usize, iterations: usize| {
            let lincs = match Lincs::new(bonds.clone(), order, iterations) {
                Ok(l) => l,
                Err(e) => panic!("{e}"),
            };
            let mut v = v0.clone();
            let mut x = &x_old + &v * h;
            if let Err(e) = lincs.constrain(&x_old, &mut x, &mut v, h, &w) {
                panic!("{e}");
            }
            lincs.max_residual(&x)
        };
        let defaults = residual(4, 1);
        let (higher, more) = (residual(8, 1), residual(8, 4));
        eprintln!("LINCS residuals (Bohr): order 4 with one correction {defaults:.3e}, order 8 with one {higher:.3e}, order 8 with four {more:.3e}");
        // Every setting holds the bonds to the guarantee: the corrections go on (before they
        // did not, the defaults left 2.7e-5 Bohr).
        for r in [defaults, higher, more] {
            assert!(r <= 1e-10, "{r:e}");
        }
    }

    #[test]
    fn a_step_that_cannot_reach_the_guarantee_is_an_error_naming_the_bond() {
        // Two bonds at lengths that a step from this structure cannot give both (the atoms
        // are on a line and the bonds would need 1 + 1 = 3 Bohr between the ends): the
        // corrections stall, and the bond is named.
        let w = vec![1.0 / 1837.0; 9];
        let lincs = match Lincs::new(
            vec![
                Bond {
                    a: 0,
                    b: 1,
                    length: 1.0,
                },
                Bond {
                    a: 1,
                    b: 2,
                    length: 1.0,
                },
                Bond {
                    a: 0,
                    b: 2,
                    length: 3.0,
                },
            ],
            4,
            1,
        ) {
            Ok(l) => l,
            Err(e) => panic!("{e}"),
        };
        let x_old = Col::from_fn(9, |k| [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.1, 0.0][k]);
        let mut x = x_old.clone();
        let mut v = Col::zeros(9);
        match lincs.constrain(&x_old, &mut x, &mut v, 1.0, &w) {
            Ok(()) => panic!("an impossible set of bonds was accepted"),
            Err(e) => assert!(e.to_string().contains("LINCS: the bond"), "{e}"),
        }
    }
}
