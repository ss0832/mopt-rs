//! Collective variables (extension point) and the harmonic umbrella restraint on them.
//!
//! A collective variable is a scalar function of the structure with its gradient and
//! Hessian. It is held as a [`PesTerm`] whose "energy" is the value, so that any term can
//! be one: the built-in ones are written over automatic differentiation
//! ([`GeometricBias`]): distances (between atoms or the centroids of groups), angles,
//! dihedral angles (periodic), coordination numbers, the path variables `s` and `z` of
//! Branduardi, Gervasio, and Parrinello, and expressions of the `-ubias` language (any
//! user-defined variable). [`Umbrella`] is the restraint `k / 2 (xi - xi_0)^2` of one window
//! of umbrella sampling.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::f64::consts::PI;
use std::sync::Arc;

use faer::{Col, Mat};

use crate::domain::{AtomId, EvalOutput, MolecularSystem, Quantity};
use crate::error::{MoptError, MoptResult};
use crate::math::ad::{angle_between, Scalar};
use crate::pes::bias::engine::norm;
use crate::pes::bias::{AdBias, Clock, ExpressionBias, Geom, GeometricBias};
use crate::pes::term::{Frame, History, PesTerm, TermContext};

/// A collective variable.
#[derive(Clone)]
pub struct Cv {
    name: String,
    term: Arc<dyn PesTerm>,
    period: Option<f64>,
}

impl std::fmt::Debug for Cv {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Cv")
            .field("name", &self.name)
            .field("period", &self.period)
            .finish()
    }
}

/// The built-in variables over automatic differentiation.
#[derive(Debug, Clone)]
enum Geometric {
    Distance(Vec<AtomId>, Vec<AtomId>),
    Angle([AtomId; 3]),
    Dihedral([AtomId; 4]),
    /// `sum_{i in A, j in B, i != j} (1 - (r/r0)^n) / (1 - (r/r0)^m)`.
    Coordination {
        a: Vec<AtomId>,
        b: Vec<AtomId>,
        r0: f64,
        n: u32,
        m: u32,
    },
    /// Path variables over the atoms `atoms` and the frames (their flat positions).
    Path {
        atoms: Vec<AtomId>,
        frames: Vec<Vec<f64>>,
        lambda: f64,
        progress: bool,
    },
}

/// `sum_{k < n} q^k`, the numerator `(1 - q^n) / (1 - q)` without its 0/0 at `q = 1`.
fn geometric_sum<T: Scalar>(q: T, n: u32) -> T {
    let mut s = T::zero();
    let mut p = T::from_f64(1.0);
    for _ in 0..n {
        s += p;
        p = p * q;
    }
    s
}

/// `ln sum_i exp(a_i)` without overflow.
fn log_sum_exp<T: Scalar>(values: &[T]) -> T {
    let top = values.iter().fold(f64::NEG_INFINITY, |m, v| m.max(v.val()));
    let sum = values
        .iter()
        .fold(T::zero(), |s, v| s + (*v - T::from_f64(top)).exp());
    sum.ln() + T::from_f64(top)
}

impl Geometric {
    fn path<T: Scalar>(&self, g: &Geom<'_, T>) -> T {
        let Self::Path {
            atoms,
            frames,
            lambda,
            progress,
        } = self
        else {
            return T::zero();
        };
        // -lambda |x - x_i|^2 over the atoms, per atom (mean square displacement).
        let exponents: Vec<T> = frames
            .iter()
            .map(|frame| {
                let mut d2 = T::zero();
                for (k, &a) in atoms.iter().enumerate() {
                    let p = g.pos(a);
                    for c in 0..3 {
                        let d = p[c] - T::from_f64(frame[3 * k + c]);
                        d2 += d * d;
                    }
                }
                T::from_f64(-lambda / atoms.len().max(1) as f64) * d2
            })
            .collect();
        let total = log_sum_exp(&exponents);
        if !progress {
            // z = -(1 / lambda) ln sum_i exp(-lambda d_i^2).
            return T::from_f64(-1.0 / lambda) * total;
        }
        // s = sum_i i w_i / sum_i w_i / (N - 1) with w_i = exp(a_i - ln sum).
        let last = (frames.len().max(2) - 1) as f64;
        exponents.iter().enumerate().fold(T::zero(), |s, (i, a)| {
            s + T::from_f64(i as f64 / last) * (*a - total).exp()
        })
    }
}

impl GeometricBias for Geometric {
    fn name(&self) -> &str {
        "collective_variable"
    }

    fn atoms(&self) -> Vec<AtomId> {
        match self {
            Self::Distance(a, b) => a.iter().chain(b).copied().collect(),
            Self::Angle(a) => a.to_vec(),
            Self::Dihedral(a) => a.to_vec(),
            Self::Coordination { a, b, .. } => a.iter().chain(b).copied().collect(),
            Self::Path { atoms, .. } => atoms.clone(),
        }
    }

    fn energy<T: Scalar>(&self, g: &Geom<'_, T>, _clock: Clock) -> T {
        match self {
            Self::Distance(a, b) => norm(g.d_centroids(a, b)),
            // atan2(|v1 x v2|, v1 . v2): accurate up to 0 and 180 degrees, where the arc
            // cosine of the normalized dot product loses the angle and its derivative.
            Self::Angle([a, b, c]) => angle_between(g.d(*a, *b), g.d(*c, *b)),
            Self::Dihedral([a, b, c, d]) => {
                crate::pes::bias::restraints::dihedral(g.d(*b, *a), g.d(*c, *b), g.d(*d, *c)).0
            }
            Self::Coordination { a, b, r0, n, m } => {
                let mut total = T::zero();
                for &i in a {
                    for &j in b {
                        if i == j {
                            continue;
                        }
                        let q = norm(g.d(i, j)) / T::from_f64(*r0);
                        total += geometric_sum(q, *n) / geometric_sum(q, *m);
                    }
                }
                total
            }
            Self::Path { .. } => self.path(g),
        }
    }

    /// Distances, angles, dihedrals, and coordination numbers are invariant; the path
    /// variables compare the positions with fixed frames without a superposition, so they
    /// depend on where the structure is.
    fn frame(&self) -> Frame {
        match self {
            Self::Path { .. } => Frame::Absolute,
            _ => Frame::Invariant,
        }
    }

    fn history(&self) -> History {
        History::Static
    }
}

impl Cv {
    fn geometric(name: String, g: Geometric, period: Option<f64>) -> Self {
        Self {
            name,
            term: Arc::new(AdBias::new(g)),
            period,
        }
    }

    /// The distance between atoms `a` and `b` (Bohr).
    pub fn distance(a: AtomId, b: AtomId) -> Self {
        Self::distance_of_groups(vec![a], vec![b])
    }

    /// The distance between the centroids of two groups (Bohr).
    pub fn distance_of_groups(a: Vec<AtomId>, b: Vec<AtomId>) -> Self {
        let name = format!("distance({:?}, {:?})", ones(&a), ones(&b));
        Self::geometric(name, Geometric::Distance(a, b), None)
    }

    /// The angle `a-b-c` at `b` (radians).
    pub fn angle(a: AtomId, b: AtomId, c: AtomId) -> Self {
        let name = format!("angle{:?}", ones(&[a, b, c]));
        Self::geometric(name, Geometric::Angle([a, b, c]), None)
    }

    /// The dihedral angle `a-b-c-d` (radians, IUPAC sign, period `2 pi`).
    pub fn dihedral(a: AtomId, b: AtomId, c: AtomId, d: AtomId) -> Self {
        let name = format!("dihedral{:?}", ones(&[a, b, c, d]));
        Self::geometric(name, Geometric::Dihedral([a, b, c, d]), Some(2.0 * PI))
    }

    /// The coordination number of group `a` with group `b`:
    /// `sum (1 - (r/r0)^n) / (1 - (r/r0)^m)` (with its limit `n / m` at `r = r0`).
    pub fn coordination(
        a: Vec<AtomId>,
        b: Vec<AtomId>,
        r0: f64,
        n: u32,
        m: u32,
    ) -> MoptResult<Self> {
        if !(r0.is_finite() && r0 > 0.0) || n == 0 || m <= n {
            return Err(MoptError::config(
                "a coordination number needs r0 > 0 and exponents 0 < n < m",
            ));
        }
        let name = format!("coordination({:?}, {:?})", ones(&a), ones(&b));
        Ok(Self::geometric(
            name,
            Geometric::Coordination { a, b, r0, n, m },
            None,
        ))
    }

    /// The progress `s` (0 at the first frame, 1 at the last) or the distance `z` from a
    /// path of frames (flat positions of `atoms`, Bohr), with `lambda` (1/Bohr^2) per atom.
    pub fn path(
        atoms: Vec<AtomId>,
        frames: Vec<Vec<f64>>,
        lambda: f64,
        progress: bool,
    ) -> MoptResult<Self> {
        if frames.len() < 2 || frames.iter().any(|f| f.len() != 3 * atoms.len()) {
            return Err(MoptError::config(
                "path variables need two or more frames of the positions of the atoms",
            ));
        }
        if !(lambda.is_finite() && lambda > 0.0) {
            return Err(MoptError::config(
                "lambda of a path variable must be positive",
            ));
        }
        let name = if progress { "path_s" } else { "path_z" }.to_string();
        Ok(Self::geometric(
            name,
            Geometric::Path {
                atoms,
                frames,
                lambda,
                progress,
            },
            None,
        ))
    }

    /// An expression of the `-ubias` language (atomic units), e.g. `dist(1,2) - dist(2,3)`.
    pub fn expression(source: &str, period: Option<f64>) -> MoptResult<Self> {
        let e = ExpressionBias::parse(source)?;
        if e.constant_value().is_some() {
            return Err(MoptError::config(format!(
                "the collective variable '{source}' does not depend on the structure"
            )));
        }
        Ok(Self {
            name: source.trim().to_string(),
            term: Arc::new(AdBias::new(e)),
            period,
        })
    }

    /// Any term as a collective variable (its energy is the value).
    pub fn from_term(name: &str, term: Arc<dyn PesTerm>, period: Option<f64>) -> Self {
        Self {
            name: name.to_string(),
            term,
            period,
        }
    }

    /// Name for files and logs.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Period of the value (`None`: not periodic).
    pub fn period(&self) -> Option<f64> {
        self.period
    }

    /// The value at `sys`.
    pub fn value(&self, sys: &MolecularSystem) -> MoptResult<f64> {
        let ctx = TermContext::at_iteration(0, sys);
        Ok(self.term.evaluate(sys, &ctx, false)?.energy.value())
    }

    /// Value, gradient, and (on request) Hessian at `sys`.
    pub fn evaluate(&self, sys: &MolecularSystem, need_hessian: bool) -> MoptResult<EvalOutput> {
        let ctx = TermContext::at_iteration(0, sys);
        self.term.evaluate(sys, &ctx, need_hessian)
    }

    /// `value - center` wrapped into half a period for periodic variables.
    pub fn difference(&self, value: f64, center: f64) -> f64 {
        let d = value - center;
        match self.period {
            Some(p) => d - p * (d / p).round(),
            None => d,
        }
    }
}

fn ones(atoms: &[AtomId]) -> Vec<usize> {
    atoms.iter().map(|a| a.index() + 1).collect()
}

/// The umbrella restraint `k / 2 (xi - xi_0)^2` on a collective variable.
#[derive(Debug, Clone)]
pub struct Umbrella {
    /// The variable.
    pub cv: Cv,
    /// Its centre `xi_0`.
    pub center: f64,
    /// The force constant `k` (Hartree per unit of the variable squared).
    pub k: f64,
    label: String,
}

impl Umbrella {
    /// The restraint on `cv` at `center` with force constant `k`.
    pub fn new(cv: Cv, center: f64, k: f64) -> MoptResult<Self> {
        if !(k.is_finite() && k > 0.0 && center.is_finite()) {
            return Err(MoptError::config(
                "an umbrella needs a finite centre and a positive force constant",
            ));
        }
        let label = format!("umbrella({}, {center})", cv.name());
        Ok(Self {
            cv,
            center,
            k,
            label,
        })
    }

    /// The bias (Hartree) of the restraint at the value `xi`.
    pub fn bias_at(&self, xi: f64) -> f64 {
        let d = self.cv.difference(xi, self.center);
        0.5 * self.k * d * d
    }
}

impl PesTerm for Umbrella {
    fn name(&self) -> &str {
        &self.label
    }

    fn evaluate(
        &self,
        sys: &MolecularSystem,
        _ctx: &TermContext<'_>,
        need_hessian: bool,
    ) -> MoptResult<EvalOutput> {
        let out = self.cv.evaluate(sys, need_hessian)?;
        let d = self.cv.difference(out.energy.value(), self.center);
        let g = &out.gradient_bohr;
        let hessian = match (need_hessian, &out.hessian_bohr) {
            (true, Some(h)) => {
                let n = g.nrows();
                Some(Mat::from_fn(n, n, |i, j| {
                    self.k * (g[i] * g[j] + d * h[(i, j)])
                }))
            }
            (true, None) => {
                return Err(MoptError::calc(format!(
                    "the variable '{}' gave no Hessian",
                    self.cv.name()
                )))
            }
            (false, _) => None,
        };
        Ok(EvalOutput {
            energy: Quantity::new(0.5 * self.k * d * d),
            gradient_bohr: Col::from_fn(g.nrows(), |i| self.k * d * g[i]),
            hessian_bohr: hessian,
            orbitals: None,
            stress: None,
            strain_hessian: None,
        })
    }

    /// Those of the variable: the restraint is a function of its value alone.
    fn frame(&self) -> Frame {
        self.cv.term.frame()
    }

    fn history(&self) -> History {
        self.cv.term.history()
    }

    fn fork(&self) -> Option<Arc<dyn PesTerm>> {
        let term = self.cv.term.fork()?;
        Some(Arc::new(Self {
            cv: Cv {
                term,
                ..self.cv.clone()
            },
            ..self.clone()
        }))
    }

    fn analytic_hessian(&self) -> bool {
        self.cv.term.analytic_hessian()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::domain::{positions, Angstrom, Element, SpinCharge};
    use crate::pes::term_check::check_term;

    fn three() -> MolecularSystem {
        MolecularSystem::new(
            vec![
                Element::parse_token("O").unwrap(),
                Element::parse_token("H").unwrap(),
                Element::parse_token("H").unwrap(),
            ],
            &positions::<Angstrom>(&[[0.0, 0.0, 0.1], [0.0, 0.76, -0.47], [0.0, -0.74, -0.49]]),
            SpinCharge::default(),
            None,
        )
        .unwrap()
    }

    #[test]
    fn the_angle_is_accurate_near_180_degrees() {
        // Atoms 1-2-3 with atom 3 at a small height y above the line: theta = pi - atan(y / l).
        let (l, y) = (1.3_f64, 2e-9_f64);
        let sys = MolecularSystem::new(
            vec![Element::parse_token("C").unwrap(); 3],
            &positions::<crate::domain::Bohr>(&[[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [l, y, 0.0]]),
            SpinCharge::default(),
            None,
        )
        .unwrap();
        let cv = Cv::angle(AtomId::new(0), AtomId::new(1), AtomId::new(2));
        let out = cv.evaluate(&sys, false).unwrap();
        let want = PI - (y / l).atan();
        assert!(
            (out.energy.value() - want).abs() < 1e-14,
            "{}",
            out.energy.value()
        );
        // d theta / d y_3 = -l / (l^2 + y^2).
        let slope = out.gradient_bohr[7];
        assert!((slope + l / (l * l + y * y)).abs() < 1e-9, "{slope}");
    }

    #[test]
    fn the_variables_have_their_values_and_exact_derivatives() {
        let sys = three();
        let (o, h1, h2) = (AtomId::new(0), AtomId::new(1), AtomId::new(2));
        let d = Cv::distance(o, h1).value(&sys).unwrap();
        assert!((d - sys.distance(o, h1).value()).abs() < 1e-12);
        // At r = r0 every pair contributes n / m.
        let cn = Cv::coordination(vec![o], vec![h1], d, 6, 12).unwrap();
        assert!((cn.value(&sys).unwrap() - 0.5).abs() < 1e-12);
        let expr = Cv::expression("dist(1,2) - dist(1,3)", None).unwrap();
        let reference = d - sys.distance(o, h2).value();
        assert!((expr.value(&sys).unwrap() - reference).abs() < 1e-12);
        // The path through the structure and a moved copy: s = 0 at the first frame.
        let x: Vec<f64> = sys.coords_flat_bohr().iter().copied().collect();
        let moved: Vec<f64> = x.iter().map(|v| v + 0.5).collect();
        let s = Cv::path(vec![o, h1, h2], vec![x.clone(), moved], 50.0, true).unwrap();
        assert!(s.value(&sys).unwrap() < 1e-3);
        for cv in [Cv::angle(h1, o, h2), cn, s, expr] {
            let umbrella = Umbrella::new(cv.clone(), 0.3, 0.2).unwrap();
            let report = check_term(&umbrella, &sys, 1e-5).unwrap();
            assert!(report.passes(1e-5), "{}: {report:?}", cv.name());
        }
        // Periodic differences are wrapped.
        let dih = Cv::dihedral(h1, o, h2, h1);
        assert!((dih.difference(3.1, -3.1) - (6.2 - 2.0 * PI)).abs() < 1e-12);
    }
}
