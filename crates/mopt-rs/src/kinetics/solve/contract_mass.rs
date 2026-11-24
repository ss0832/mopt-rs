//! `contract`: the contraction of mass-action networks of any reaction order, the extension of
//! RCMC to non-linear kinetics (`kinetics_beyond_tst.md`, section 3.8.1; the proposal checked
//! by `tools/kinetics/check_nonlinear_contraction.py`, whose algorithm this is).
//!
//! For `dc/dt = f(c) = N v(c)` with the Jacobian `J`, the species are split into eliminated
//! ones `S` and kept ones `T`:
//!
//! - **manifold**: the eliminated species follow the kept ones, `f_S(c_S, c_T) = 0`, solved by
//!   Newton's method (`c_S = h(c_T)`, `h' = -J_SS^-1 J_ST`);
//! - **reduced equation**: `(I + B h') dc_T/dt = f_T(h(c_T), c_T)` with `B = -J_TS J_SS^-1`
//!   (`M dp_T/dt = D p_T` of RCMC for a first-order network; without the factor `I + B h'`
//!   the plain quasi-steady-state approximation, an order less accurate);
//! - **start**: the composition is moved onto the manifold along `[I B]`,
//!   `c_T + B c_S = c_T(0) + B c_S(0)`, which keeps every conserved total (the start of the slow
//!   solution continued back to `t = 0`; a product of an eliminated species may start slightly
//!   below zero, so the rate laws are evaluated as the signed polynomials they are);
//! - **choice**: fastest first, as RCMC: the kept species with the largest effective loss rate
//!   `-D_jj` (`D = J_TT - J_TS J_SS^-1 J_ST` at the current composition) is eliminated while
//!   `rate * t > theta` (`--contract-theta`, 100), the eliminated block relaxes as a whole
//!   within the horizon (`-max Re eig(J_SS) * t > theta`: all forms of a catalyst are
//!   short-lived, their total is not), and the projection onto its manifold exists.
//!
//! The cut is made for every output time; the reduced equation is not stiff on the scale of `t`
//! (its rates are below `theta / t`) and is integrated by the Dormand-Prince pair 5(4) with
//! step-size control (Dormand, Prince, J. Comput. Appl. Math. 1980, 6, 19). With nothing to
//! eliminate the full equations are integrated by ROS2 (the solver `rosenbrock`). The model is
//! `mass_action` (concentrations) or a linear one (populations), where the blocks are those of
//! RCMC.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::linalg::solvers::Solve;
use faer::Mat;

use super::{check_input, Evolution, KineticSolution, KineticSolver, SolveDiagnostics};
use crate::core_infra::rng::Rng;
use crate::error::{MoptError, MoptResult};
use crate::kinetics::mass_action::{ros2, Ode};

/// The default threshold of `rate * t` above which a species is eliminated.
pub const DEFAULT_THETA: f64 = 100.0;

/// The contraction of mass action.
#[derive(Debug, Clone, Copy)]
pub struct ContractMassAction {
    /// Species are eliminated while `rate * t` exceeds this.
    pub theta: f64,
    /// The relative tolerance of the integration of the reduced equation.
    pub tolerance: f64,
}

/// `dp/dt = K p` of a linear model.
struct LinearSystem {
    n: usize,
    k: Vec<f64>,
}

impl Ode for LinearSystem {
    fn dim(&self) -> usize {
        self.n
    }
    fn rhs(&self, y: &[f64], f: &mut [f64]) {
        for (i, fi) in f.iter_mut().enumerate() {
            *fi = (0..self.n).map(|j| self.k[i * self.n + j] * y[j]).sum();
        }
    }
    fn jacobian(&self, _y: &[f64], j: &mut [f64]) {
        j.copy_from_slice(&self.k);
    }
}

fn jacobian(ode: &dyn Ode, y: &[f64]) -> Mat<f64> {
    let n = ode.dim();
    let mut j = vec![0.0; n * n];
    ode.jacobian(y, &mut j);
    Mat::from_fn(n, n, |a, b| j[a * n + b])
}

fn rhs(ode: &dyn Ode, y: &[f64]) -> Vec<f64> {
    let mut f = vec![0.0; ode.dim()];
    ode.rhs(y, &mut f);
    f
}

fn sub(a: &Mat<f64>, rows: &[usize], cols: &[usize]) -> Mat<f64> {
    Mat::from_fn(rows.len(), cols.len(), |i, j| a[(rows[i], cols[j])])
}

fn finite(m: &Mat<f64>) -> bool {
    m.col_iter().all(|c| c.iter().all(|v| v.is_finite()))
}

/// The solution `x` of `a x = b` for square `a`, or an error when `a` is singular.
fn solve(a: &Mat<f64>, b: &Mat<f64>) -> MoptResult<Mat<f64>> {
    let x = a.partial_piv_lu().solve(b.as_ref());
    if finite(&x) {
        Ok(x)
    } else {
        Err(MoptError::linalg(
            "contract: a block of the Jacobian is singular",
        ))
    }
}

/// A split of the species.
struct Split {
    s: Vec<usize>,
    t: Vec<usize>,
}

impl Split {
    fn new(n: usize, s: Vec<usize>) -> Self {
        let t = (0..n).filter(|i| !s.contains(i)).collect();
        Self { s, t }
    }

    /// `B`, `h'`, and `D` at the composition `c`.
    fn blocks(&self, ode: &dyn Ode, c: &[f64]) -> MoptResult<(Mat<f64>, Mat<f64>, Mat<f64>)> {
        let j = jacobian(ode, c);
        let (s, t) = (&self.s, &self.t);
        if s.is_empty() {
            return Ok((
                Mat::zeros(t.len(), 0),
                Mat::zeros(0, t.len()),
                sub(&j, t, t),
            ));
        }
        let jss = sub(&j, s, s);
        let jts = sub(&j, t, s);
        let jst = sub(&j, s, t);
        let hp = -solve(&jss, &jst)?;
        // B = -J_TS J_SS^-1 = -(J_SS^-T J_TS^T)^T.
        let b = -solve(&jss.transpose().to_owned(), &jts.transpose().to_owned())?
            .transpose()
            .to_owned();
        let d = sub(&j, t, t) + &jts * &hp;
        Ok((b, hp, d))
    }

    /// The full composition on the manifold `f_S = 0` for the kept values `c_t` (Newton from
    /// `guess`, the eliminated values kept from going below zero).
    fn manifold(&self, ode: &dyn Ode, c_t: &[f64], guess: &[f64]) -> MoptResult<Vec<f64>> {
        let n = ode.dim();
        let mut c = vec![0.0; n];
        for (x, &i) in c_t.iter().zip(&self.t) {
            c[i] = *x;
        }
        if self.s.is_empty() {
            return Ok(c);
        }
        let scale = c_t.iter().fold(0.0_f64, |m, x| m.max(x.abs())).max(1e-300);
        let floor = 1e-12 * scale;
        for &i in &self.s {
            c[i] = guess[i].max(floor);
        }
        for _ in 0..200 {
            let f = rhs(ode, &c);
            let j = jacobian(ode, &c);
            let jss = sub(&j, &self.s, &self.s);
            let fs = Mat::from_fn(self.s.len(), 1, |a, _| -f[self.s[a]]);
            let step = solve(&jss, &fs)?;
            let mut lambda = 1.0;
            let fits = (0..self.s.len()).all(|a| c[self.s[a]] + step[(a, 0)] >= -floor);
            if fits {
                for (a, &i) in self.s.iter().enumerate() {
                    c[i] = (c[i] + step[(a, 0)]).max(0.0);
                }
            } else {
                while (0..self.s.len()).any(|a| c[self.s[a]] + lambda * step[(a, 0)] <= 0.0)
                    && lambda > 1e-12
                {
                    lambda *= 0.5;
                }
                for (a, &i) in self.s.iter().enumerate() {
                    c[i] += lambda * step[(a, 0)];
                }
            }
            let big = (0..self.s.len())
                .map(|a| (lambda * step[(a, 0)]).abs())
                .fold(0.0, f64::max);
            let size = self.s.iter().map(|&i| c[i].abs()).fold(floor, f64::max);
            if big <= 1e-14 * size {
                return Ok(c);
            }
        }
        // A double root (a dimer at zero concentration) is approached linearly: after 200
        // halvings the remainder is far below every concentration that matters, as in the
        // check script.
        Ok(c)
    }

    /// `c_from` moved onto the manifold along `[I B]` (the conserved totals are kept).
    fn project(&self, ode: &dyn Ode, c_from: &[f64]) -> MoptResult<Vec<f64>> {
        if self.s.is_empty() {
            return Ok(c_from.to_vec());
        }
        let nt = self.t.len();
        let scale = c_from
            .iter()
            .fold(0.0_f64, |m, x| m.max(x.abs()))
            .max(1e-300);
        let mut guess = c_from.to_vec();
        let mut c_t: Vec<f64> = self.t.iter().map(|&i| c_from[i]).collect();
        let residual = |x: &[f64], guess: &[f64]| -> MoptResult<(Vec<f64>, Vec<f64>)> {
            let c = self.manifold(ode, x, guess)?;
            let (b, _, _) = self.blocks(ode, &c)?;
            let r = (0..nt)
                .map(|a| {
                    c[self.t[a]] - c_from[self.t[a]]
                        + (0..self.s.len())
                            .map(|q| b[(a, q)] * (c[self.s[q]] - c_from[self.s[q]]))
                            .sum::<f64>()
                })
                .collect();
            Ok((r, c))
        };
        for _ in 0..100 {
            let (r, c) = residual(&c_t, &guess)?;
            guess = c;
            if r.iter().fold(0.0_f64, |m, x| m.max(x.abs())) <= 1e-13 * scale {
                break;
            }
            let h = 1e-6 * scale;
            let mut jac = Mat::<f64>::zeros(nt, nt);
            for col in 0..nt {
                let mut shifted = c_t.clone();
                shifted[col] += h;
                let (rs, _) = residual(&shifted, &guess)?;
                for a in 0..nt {
                    jac[(a, col)] = (rs[a] - r[a]) / h;
                }
            }
            let rhs_m = Mat::from_fn(nt, 1, |a, _| -r[a]);
            let step = solve(&jac, &rhs_m)?;
            for (a, x) in c_t.iter_mut().enumerate() {
                *x += step[(a, 0)];
            }
        }
        self.manifold(ode, &c_t, &guess)
    }

    /// `dc_T/dt` of the reduced equation at the kept values `c_t`.
    fn reduced(
        &self,
        ode: &dyn Ode,
        c_t: &[f64],
        guess: &[f64],
    ) -> MoptResult<(Vec<f64>, Vec<f64>)> {
        let c = self.manifold(ode, c_t, guess)?;
        let (b, hp, _) = self.blocks(ode, &c)?;
        let nt = self.t.len();
        let m = Mat::<f64>::identity(nt, nt) + &b * &hp;
        let f = rhs(ode, &c);
        let ft = Mat::from_fn(nt, 1, |a, _| f[self.t[a]]);
        let x = solve(&m, &ft)?;
        Ok(((0..nt).map(|a| x[(a, 0)]).collect(), c))
    }
}

/// The largest real part of the eigenvalues of `a` (scaled by a power of two first).
fn largest_real_part(a: &Mat<f64>) -> MoptResult<f64> {
    let n = a.nrows();
    if n == 1 {
        return Ok(a[(0, 0)]);
    }
    let big = (0..n)
        .flat_map(|i| (0..n).map(move |j| (i, j)))
        .map(|(i, j)| a[(i, j)].abs())
        .fold(0.0, f64::max);
    if !(big.is_finite() && big > 0.0) {
        return Ok(0.0);
    }
    let s = 2f64.powi(-(big.log2().round() as i32));
    let scaled = Mat::from_fn(n, n, |i, j| a[(i, j)] * s);
    let evd = scaled
        .eigen()
        .map_err(|e| MoptError::linalg(format!("contract: eigenvalues failed: {e:?}")))?;
    let values = evd.S();
    Ok((0..n)
        .map(|k| values[k].re / s)
        .fold(f64::NEG_INFINITY, f64::max))
}

/// Dormand-Prince 5(4) from `y0` over `[0, t_end]` with the relative tolerance `rtol`.
fn dopri5(
    y0: &[f64],
    t_end: f64,
    rtol: f64,
    f: &mut dyn FnMut(&[f64]) -> MoptResult<Vec<f64>>,
) -> MoptResult<(Vec<f64>, usize)> {
    const A: [&[f64]; 6] = [
        &[1.0 / 5.0],
        &[3.0 / 40.0, 9.0 / 40.0],
        &[44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0],
        &[
            19372.0 / 6561.0,
            -25360.0 / 2187.0,
            64448.0 / 6561.0,
            -212.0 / 729.0,
        ],
        &[
            9017.0 / 3168.0,
            -355.0 / 33.0,
            46732.0 / 5247.0,
            49.0 / 176.0,
            -5103.0 / 18656.0,
        ],
        &[
            35.0 / 384.0,
            0.0,
            500.0 / 1113.0,
            125.0 / 192.0,
            -2187.0 / 6784.0,
            11.0 / 84.0,
        ],
    ];
    const E: [f64; 7] = [
        71.0 / 57600.0,
        0.0,
        -71.0 / 16695.0,
        71.0 / 1920.0,
        -17253.0 / 339200.0,
        22.0 / 525.0,
        -1.0 / 40.0,
    ];
    let n = y0.len();
    let scale = y0.iter().fold(0.0_f64, |m, x| m.max(x.abs())).max(1e-300);
    let atol = rtol * 1e-6 * scale;
    let mut y = y0.to_vec();
    if t_end <= 0.0 || n == 0 {
        return Ok((y, 0));
    }
    let mut k1 = f(&y)?;
    let norm = k1.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
    let mut h = if norm > 0.0 {
        (0.01 * scale / norm).min(t_end)
    } else {
        t_end
    };
    let mut t = 0.0;
    let mut steps = 0;
    while t < t_end {
        if steps > crate::kinetics::mass_action::MAX_STEPS {
            return Err(MoptError::calc(
                "contract: the reduced equation needs too many steps (is it stiff?)",
            ));
        }
        h = h.min(t_end - t);
        let mut k = vec![k1.clone()];
        for row in A.iter().take(5) {
            let yi: Vec<f64> = (0..n)
                .map(|i| {
                    y[i] + h * row
                        .iter()
                        .enumerate()
                        .map(|(q, a)| a * k[q][i])
                        .sum::<f64>()
                })
                .collect();
            k.push(f(&yi)?);
        }
        let y5: Vec<f64> = (0..n)
            .map(|i| {
                y[i] + h * A[5]
                    .iter()
                    .enumerate()
                    .map(|(q, a)| a * k[q][i])
                    .sum::<f64>()
            })
            .collect();
        let k7 = f(&y5)?;
        k.push(k7.clone());
        let mut err: f64 = 0.0;
        for i in 0..n {
            let e = h * (0..7).map(|q| E[q] * k[q][i]).sum::<f64>();
            let sc = atol + rtol * y[i].abs().max(y5[i].abs());
            err = err.max((e / sc).abs());
        }
        if err <= 1.0 || h <= 1e-14 * t_end {
            t += h;
            y = y5;
            k1 = k7;
            steps += 1;
        }
        let factor = if err > 0.0 {
            (0.9 * err.powf(-0.2)).clamp(0.2, 5.0)
        } else {
            5.0
        };
        h *= factor;
    }
    Ok((y, steps))
}

/// The composition at `t` of the contracted model of `ode` from `y0`: the eliminated species,
/// the composition, and the steps of the integration.
pub fn contract_mass_action(
    ode: &dyn Ode,
    y0: &[f64],
    t: f64,
    theta: f64,
    rtol: f64,
) -> MoptResult<(Vec<usize>, Vec<f64>, usize)> {
    let n = ode.dim();
    let mut split = Split::new(n, Vec::new());
    let mut c = y0.to_vec();
    while split.s.len() + 1 < n {
        let (_, _, d) = split.blocks(ode, &c)?;
        let (mut pick, mut rate) = (0usize, f64::NEG_INFINITY);
        for a in 0..split.t.len() {
            if -d[(a, a)] > rate {
                rate = -d[(a, a)];
                pick = a;
            }
        }
        if rate * t <= theta {
            break;
        }
        let mut s = split.s.clone();
        s.push(split.t[pick]);
        let trial = Split::new(n, s);
        let j = jacobian(ode, &c);
        let jss = sub(&j, &trial.s, &trial.s);
        if -largest_real_part(&jss)? * t <= theta {
            break;
        }
        match trial.project(ode, &c) {
            Ok(projected) => {
                c = projected;
                split = trial;
            }
            Err(_) => break,
        }
    }
    if split.s.is_empty() {
        let (ys, stats) = ros2(
            ode,
            y0,
            &[t],
            rtol,
            rtol * 1e-6 * y0.iter().fold(0.0_f64, |m, x| m.max(x.abs())).max(1e-300),
        )?;
        let y = ys.into_iter().next().unwrap_or_default();
        return Ok((Vec::new(), y, stats.steps));
    }
    let mut guess = c.clone();
    let c_t: Vec<f64> = split.t.iter().map(|&i| c[i]).collect();
    let mut f = |x: &[f64]| -> MoptResult<Vec<f64>> {
        let (dx, full) = split.reduced(ode, x, &guess)?;
        guess = full;
        Ok(dx)
    };
    let (end, steps) = dopri5(&c_t, t, rtol, &mut f)?;
    let full = split.manifold(ode, &end, &c)?;
    Ok((split.s.clone(), full, steps))
}

impl KineticSolver for ContractMassAction {
    fn name(&self) -> &str {
        "contract"
    }

    fn solve(
        &mut self,
        ev: &Evolution,
        p0: &[f64],
        times: &[f64],
        _rng: &mut Rng,
    ) -> MoptResult<KineticSolution> {
        if !(self.theta > 0.0 && self.tolerance > 0.0 && self.tolerance < 1.0) {
            return Err(MoptError::config(
                "contract: the threshold must be positive and the tolerance between 0 and 1",
            ));
        }
        let (ode, y0, system): (Box<dyn Ode>, Vec<f64>, Option<_>) = match ev {
            Evolution::MassAction(sys) => {
                check_input(sys.n_states, p0, times)?;
                (Box::new(sys.clone()), sys.initial(p0)?, Some(sys))
            }
            _ => {
                let g = ev.generator()?;
                check_input(g.n, p0, times)?;
                let m = g.rates().matrix();
                let n = g.n;
                let k = (0..n * n).map(|x| m[(x / n, x % n)]).collect();
                (
                    Box::new(LinearSystem { n, k }) as Box<dyn Ode>,
                    p0.to_vec(),
                    None,
                )
            }
        };
        let labels: Vec<String> = match ev {
            Evolution::MassAction(sys) => sys.labels.clone(),
            _ => ev.generator()?.labels.clone(),
        };
        let mut populations = Vec::with_capacity(times.len());
        let mut diagnostics = Vec::with_capacity(times.len());
        for &t in times {
            let (eliminated, y, steps) =
                contract_mass_action(ode.as_ref(), &y0, t, self.theta, self.tolerance)?;
            let conservation = match system {
                Some(sys) => sys.conservation(&y0, &y),
                None => (y.iter().sum::<f64>() - y0.iter().sum::<f64>()).abs(),
            };
            let names: Vec<&str> = eliminated.iter().map(|&i| labels[i].as_str()).collect();
            diagnostics.push(SolveDiagnostics {
                time: t,
                eliminated: eliminated.len(),
                conservation,
                most_negative: y.iter().copied().fold(0.0, f64::min),
                notes: vec![format!(
                    "contract: {} eliminated ({}), {steps} steps of the reduced equation",
                    eliminated.len(),
                    if names.is_empty() {
                        "none".to_string()
                    } else {
                        names.join(", ")
                    }
                )],
                ..SolveDiagnostics::default()
            });
            populations.push(y);
        }
        Ok(KineticSolution {
            solver: "contract".to_string(),
            times: times.to_vec(),
            populations,
            diagnostics,
            superstates: vec![None; times.len()],
            notes: vec![format!(
                "contract: species with an effective loss rate times t above {} are on their \
                 quasi-steady manifold; relative tolerance {:e}",
                self.theta, self.tolerance
            )],
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::kinetics::mass_action::{MassActionSystem, Reaction};

    /// A system of entities `labels` with reactions `(reactants, products, k)`.
    fn system(labels: &[&str], reactions: &[(&[usize], &[usize], f64)]) -> MassActionSystem {
        MassActionSystem {
            labels: labels.iter().map(|s| s.to_string()).collect(),
            n_states: labels.len(),
            species: Vec::new(),
            reactions: reactions
                .iter()
                .enumerate()
                .map(|(step, (r, p, k))| Reaction {
                    reactants: r.to_vec(),
                    products: p.to_vec(),
                    ln_k: k.ln(),
                    step,
                })
                .collect(),
            species_initial: Vec::new(),
            total: 1.0,
            conserved: Vec::new(),
        }
    }

    /// The stiff integration of the full equations with the relative tolerance `rtol`.
    fn reference(sys: &MassActionSystem, y0: &[f64], t: f64, rtol: f64) -> Vec<f64> {
        let (ys, _) = ros2(sys, y0, &[t], rtol, rtol * 1e-6).unwrap();
        ys.into_iter().next().unwrap()
    }

    fn error(a: &[f64], b: &[f64]) -> f64 {
        let big = b.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
        a.iter()
            .zip(b)
            .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()))
            / big
    }

    #[test]
    fn association_converges_at_second_order_in_the_separation() {
        // A + B <-> X -> P with k_assoc = 1, k_diss + k_2 = fast (table of section 3.8.1:
        // 7e-4, 7e-6, 7e-8 for 1e1, 1e2, 1e3).
        let mut last = f64::INFINITY;
        for fast in [1e1, 1e2, 1e3] {
            let sys = system(
                &["A", "B", "X", "P"],
                &[
                    (&[0, 1], &[2], 1.0),
                    (&[2], &[0, 1], 0.7 * fast),
                    (&[2], &[3], 0.3 * fast),
                ],
            );
            let y0 = [1.0, 0.6, 0.0, 0.0];
            let (s, y, _) = contract_mass_action(&sys, &y0, 3.0, 10.0, 1e-11).unwrap();
            assert_eq!(s, vec![2]);
            let exact = reference(&sys, &y0, 3.0, 1e-10);
            let e = error(&y, &exact);
            assert!(e < 3e-2 / fast / fast * 10.0, "{fast}: {e}");
            assert!(e < last / 50.0, "{fast}: {e} after {last}");
            last = e;
            // A + X + P and B + X + P are conserved.
            assert!((y[0] + y[2] + y[3] - 1.0).abs() < 1e-12);
            assert!((y[1] + y[2] + y[3] - 0.6).abs() < 1e-12);
        }
    }

    #[test]
    fn dimerization_enzyme_and_third_order() {
        // 2 A <-> D fast, D -> P slow: 8e-7 in the table.
        let sys = system(
            &["A", "D", "P"],
            &[
                (&[0, 0], &[1], 1e5),
                (&[1], &[0, 0], 1e3),
                (&[1], &[2], 1e-1),
            ],
        );
        let y0 = [1.0, 0.0, 0.0];
        let (s, y, _) = contract_mass_action(&sys, &y0, 10.0, 10.0, 1e-11).unwrap();
        assert!(s.contains(&0), "{s:?}");
        let e = error(&y, &reference(&sys, &y0, 10.0, 1e-8));
        assert!(e < 1e-5, "{e}");
        assert!((y[0] + 2.0 * y[1] + 2.0 * y[2] - 1.0).abs() < 1e-12);
        // E + S <-> ES -> E + P with the substrate in excess: 6e-7.
        let sys = system(
            &["E", "S", "ES", "P"],
            &[
                (&[0, 1], &[2], 1e4),
                (&[2], &[0, 1], 1e3),
                (&[2], &[0, 3], 1e2),
            ],
        );
        let y0 = [1e-3, 1.0, 0.0, 0.0];
        let (s, y, _) = contract_mass_action(&sys, &y0, 20.0, 10.0, 1e-11).unwrap();
        assert!(!s.is_empty());
        let e = error(&y, &reference(&sys, &y0, 20.0, 1e-8));
        assert!(e < 1e-5, "{e}");
        // 2 A + B <-> T fast, T -> P slow: 8e-7.
        let sys = system(
            &["A", "B", "T", "P"],
            &[
                (&[0, 0, 1], &[2], 1e5),
                (&[2], &[0, 0, 1], 1e3),
                (&[2], &[3], 5e-2),
            ],
        );
        let y0 = [1.0, 0.8, 0.0, 0.0];
        let (s, y, _) = contract_mass_action(&sys, &y0, 10.0, 10.0, 1e-11).unwrap();
        assert!(!s.is_empty());
        let e = error(&y, &reference(&sys, &y0, 10.0, 1e-8));
        assert!(e < 1e-5, "{e}");
    }

    #[test]
    fn a_first_order_network_has_the_blocks_of_rcmc() {
        use crate::kinetics::contract::contract;
        use crate::kinetics::generator::Generator;
        // A <-> I <-> P with a short-lived I: the blocks of the contraction of I are D and M of
        // RCMC, and the propagated populations those of the matrix exponential.
        let k = vec![
            vec![0.0, 40.0, 0.0],
            vec![1.0, 0.0, 0.5],
            vec![0.0, 60.0, 0.0],
        ];
        let g = Generator::from_matrix(&k).unwrap();
        let rates = g.rates();
        let m = rates.matrix();
        let lin = LinearSystem {
            n: 3,
            k: (0..9).map(|x| m[(x / 3, x % 3)]).collect(),
        };
        let split = Split::new(3, vec![1]);
        let (b, hp, d) = split.blocks(&lin, &[1.0, 0.0, 0.0]).unwrap();
        let c = contract(&rates, &[1]).unwrap();
        let dr = c.d_matrix();
        let mr = c.m_matrix();
        let mm = Mat::<f64>::identity(2, 2) + &b * &hp;
        for i in 0..2 {
            for j in 0..2 {
                assert!((d[(i, j)] - dr[(i, j)]).abs() < 1e-12);
                assert!((mm[(i, j)] - mr[(i, j)]).abs() < 1e-12);
            }
        }
        let ev = Evolution::Linear(g);
        let mut rng = Rng::from_seed(1);
        let mut solver = ContractMassAction {
            theta: 10.0,
            tolerance: 1e-11,
        };
        let sol = solver
            .solve(&ev, &[1.0, 0.0, 0.0], &[5.0], &mut rng)
            .unwrap();
        let exact = super::super::Expm
            .solve(&ev, &[1.0, 0.0, 0.0], &[5.0], &mut rng)
            .unwrap();
        assert_eq!(sol.diagnostics[0].eliminated, 1);
        let e = error(&sol.populations[0], &exact.populations[0]);
        assert!(e < 1e-3, "{e}");
        assert!((sol.populations[0].iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }
}
