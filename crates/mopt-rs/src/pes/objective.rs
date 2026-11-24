//! Objectives of plain variables: functions `x in R^n -> (E, dE/dx, d2E/dx2)` without atoms,
//! optimized by [`crate::opt::vector::VectorOptimizer`].
//!
//! Built in: the two-variable Muller-Brown surface (`mb2d`), quadratic forms, and expressions of
//! the variables `q1`, `q2`, ... (`expr:...`, the grammar of `-ubias`). Rust code adds others by
//! implementing [`Objective`] and registering a factory in an [`ObjectiveRegistry`]; Python
//! passes a function to the ask/tell optimizer.
//!
//! MultiOptPy: no counterpart (new in mopt-rs; MultiOptPy's model surfaces act on atoms).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::BTreeMap;
use std::sync::Arc;

use faer::{Col, Mat};

use crate::error::{MoptError, MoptResult};
use crate::math::ad::eval_value_grad_hessian;
use crate::pes::bias::VariableExpression;
use crate::pes::counting::{CountingObjective, EvaluationCounters, EvaluationCounts};
use crate::pes::model_objectives::{Model, ModelObjective};
use crate::pes::native::muller_brown::muller_brown;
use crate::pes::registry::CalculatorOptions;

/// Value, gradient, and (on request) Hessian of an objective.
#[derive(Debug, Clone)]
pub struct ObjectiveValue {
    /// Objective value.
    pub value: f64,
    /// Gradient.
    pub gradient: Col<f64>,
    /// Hessian, when requested.
    pub hessian: Option<Mat<f64>>,
}

/// A function of `n` plain variables.
pub trait Objective: Send + Sync {
    /// Short name.
    fn name(&self) -> &str;

    /// Where the objective comes from, for the run log.
    fn description(&self) -> String {
        format!("{}: built into mopt-rs {}", self.name(), crate::VERSION)
    }

    /// The number of variables it takes (`None`: any number).
    fn dimension(&self) -> Option<usize>;

    /// Names of the variables (default `q1`, `q2`, ...).
    fn variable_names(&self, n: usize) -> Vec<String> {
        (1..=n).map(|k| format!("q{k}")).collect()
    }

    /// Evaluate at `x`.
    fn evaluate(&self, x: &Col<f64>, need_hessian: bool) -> MoptResult<ObjectiveValue>;
}

/// Check `x` against the objective's dimension.
pub fn check_dimension(obj: &dyn Objective, n: usize) -> MoptResult<()> {
    match obj.dimension() {
        Some(d) if d != n => Err(MoptError::config(format!(
            "objective '{}' takes {d} variables, got {n}",
            obj.name()
        ))),
        _ => Ok(()),
    }
}

/// The Muller-Brown surface of two variables (Muller and Brown, Theor. Chim. Acta 1979, 53, 75),
/// scaled by `energy_scale` (default `1e-3`, as the one-atom `muller_brown` calculator).
#[derive(Debug, Clone, Copy)]
pub struct MullerBrown2D {
    /// Factor applied to the original potential.
    pub energy_scale: f64,
}

impl Default for MullerBrown2D {
    fn default() -> Self {
        Self { energy_scale: 1e-3 }
    }
}

impl Objective for MullerBrown2D {
    fn name(&self) -> &str {
        "mb2d"
    }

    fn dimension(&self) -> Option<usize> {
        Some(2)
    }

    fn variable_names(&self, _n: usize) -> Vec<String> {
        vec!["x".into(), "y".into()]
    }

    fn evaluate(&self, x: &Col<f64>, need_hessian: bool) -> MoptResult<ObjectiveValue> {
        check_dimension(self, x.nrows())?;
        let p = [x[0], x[1]];
        let (e, g, h) = eval_value_grad_hessian(&p, None, |v| muller_brown(v[0], v[1]));
        let s = self.energy_scale;
        Ok(ObjectiveValue {
            value: s * e,
            gradient: Col::from_fn(2, |i| s * g[i]),
            hessian: need_hessian.then(|| Mat::from_fn(2, 2, |i, j| s * h[(i, j)])),
        })
    }
}

/// `E = e0 + (x - m)^T A (x - m) / 2` with a symmetric `A`.
#[derive(Debug, Clone)]
pub struct Quadratic {
    /// The matrix `A`.
    pub matrix: Mat<f64>,
    /// The stationary point `m`.
    pub center: Col<f64>,
    /// The value `e0` at `m`.
    pub value: f64,
}

impl Quadratic {
    /// A quadratic form; `matrix` must be square, symmetric, and match `center`.
    pub fn new(matrix: Mat<f64>, center: Col<f64>, value: f64) -> MoptResult<Self> {
        let n = center.nrows();
        if matrix.nrows() != n || matrix.ncols() != n {
            return Err(MoptError::config(format!(
                "quadratic form: a {}x{} matrix for {n} variables",
                matrix.nrows(),
                matrix.ncols()
            )));
        }
        for i in 0..n {
            for j in 0..i {
                if (matrix[(i, j)] - matrix[(j, i)]).abs() > 1e-12 * (1.0 + matrix[(i, j)].abs()) {
                    return Err(MoptError::config(
                        "quadratic form: the matrix is not symmetric",
                    ));
                }
            }
        }
        Ok(Self {
            matrix,
            center,
            value,
        })
    }
}

impl Objective for Quadratic {
    fn name(&self) -> &str {
        "quadratic"
    }

    fn dimension(&self) -> Option<usize> {
        Some(self.center.nrows())
    }

    fn evaluate(&self, x: &Col<f64>, need_hessian: bool) -> MoptResult<ObjectiveValue> {
        check_dimension(self, x.nrows())?;
        let d = x - &self.center;
        let g = &self.matrix * &d;
        let value = self.value + 0.5 * (0..d.nrows()).map(|i| d[i] * g[i]).sum::<f64>();
        Ok(ObjectiveValue {
            value,
            gradient: g,
            hessian: need_hessian.then(|| self.matrix.clone()),
        })
    }
}

/// An expression of `q1`, `q2`, ... (`-os expr:...`).
#[derive(Debug, Clone)]
pub struct ExpressionObjective {
    expr: VariableExpression,
}

impl ExpressionObjective {
    /// Parse `source`.
    pub fn parse(source: &str) -> MoptResult<Self> {
        Ok(Self {
            expr: VariableExpression::parse(source)?,
        })
    }
}

impl Objective for ExpressionObjective {
    fn name(&self) -> &str {
        "expr"
    }

    fn description(&self) -> String {
        format!(
            "expr: the expression '{}' (built into mopt-rs {})",
            self.expr.source(),
            crate::VERSION
        )
    }

    fn dimension(&self) -> Option<usize> {
        None
    }

    fn evaluate(&self, x: &Col<f64>, need_hessian: bool) -> MoptResult<ObjectiveValue> {
        let n = x.nrows();
        if self.expr.variables_used() > n {
            return Err(MoptError::config(format!(
                "the expression uses q{} but there are {n} variables",
                self.expr.variables_used()
            )));
        }
        let p: Vec<f64> = x.iter().copied().collect();
        let (value, gradient, h) = eval_value_grad_hessian(&p, None, |v| self.expr.value(v, 0));
        Ok(ObjectiveValue {
            value,
            gradient,
            hessian: need_hessian.then_some(h),
        })
    }
}

/// What a factory receives.
#[derive(Debug, Clone, Copy)]
pub struct ObjectiveRequest<'a> {
    /// The full name as given (`mb2d`, `expr:q1^2 + q2^2`).
    pub name: &'a str,
    /// The part after `:` (empty when absent).
    pub argument: &'a str,
    /// Settings (`-calc_opt key=value` in `extra`).
    pub options: &'a CalculatorOptions,
}

/// Factory of an objective.
pub type ObjectiveFactory =
    Arc<dyn Fn(&ObjectiveRequest<'_>) -> MoptResult<Box<dyn Objective>> + Send + Sync>;

/// Objectives by name.
#[derive(Clone, Default)]
pub struct ObjectiveRegistry {
    factories: BTreeMap<String, ObjectiveFactory>,
    /// The evaluations of the objectives created here (shared by clones).
    counters: Arc<EvaluationCounters>,
}

impl std::fmt::Debug for ObjectiveRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ObjectiveRegistry")
            .field("names", &self.names())
            .finish()
    }
}

fn number(key: &str, value: &str) -> MoptResult<f64> {
    value
        .trim()
        .parse()
        .map_err(|_| MoptError::config(format!("{key} must be a number, got '{value}'")))
}

impl ObjectiveRegistry {
    /// The built-in objectives: `mb2d` (`muller_brown_2d`; `-calc_opt energy_scale=S`), the model
    /// surfaces of [`crate::pes::model_objectives`] (`wolfe_quapp`, `three_hole`, `himmelblau`,
    /// `rosenbrock`, `styblinski_tang`; `-calc_opt energy_scale=S`), and `expr:...`.
    pub fn builtin() -> Self {
        let mut r = Self::default();
        r.register("mb2d", |q| {
            let mut o = MullerBrown2D::default();
            for (k, v) in &q.options.extra {
                match k.as_str() {
                    "energy_scale" => o.energy_scale = number(k, v)?,
                    other => {
                        return Err(MoptError::config(format!(
                            "mb2d: unknown setting '{other}' (known: energy_scale)"
                        )))
                    }
                }
            }
            Ok(Box::new(o))
        });
        r.alias("muller_brown_2d", "mb2d");
        for model in Model::ALL {
            r.register(model.name(), move |q| {
                let mut o = ModelObjective::new(model);
                for (k, v) in &q.options.extra {
                    match k.as_str() {
                        "energy_scale" => o.energy_scale = number(k, v)?,
                        other => {
                            return Err(MoptError::config(format!(
                                "{}: unknown setting '{other}' (known: energy_scale)",
                                model.name()
                            )))
                        }
                    }
                }
                Ok(Box::new(o))
            });
        }
        r.register("expr", |q| {
            if q.argument.trim().is_empty() {
                return Err(MoptError::config(
                    "expr: give the expression after the colon, e.g. expr:(q1-1)^2+q2^2",
                ));
            }
            Ok(Box::new(ExpressionObjective::parse(q.argument)?))
        });
        r
    }

    /// Register `factory` under `name` (case-insensitive).
    pub fn register(
        &mut self,
        name: &str,
        factory: impl Fn(&ObjectiveRequest<'_>) -> MoptResult<Box<dyn Objective>>
            + Send
            + Sync
            + 'static,
    ) {
        self.factories
            .insert(name.to_ascii_lowercase(), Arc::new(factory));
    }

    /// Another name for a registered objective.
    pub fn alias(&mut self, alias: &str, target: &str) {
        if let Some(f) = self.factories.get(&target.to_ascii_lowercase()).cloned() {
            self.factories.insert(alias.to_ascii_lowercase(), f);
        }
    }

    /// Registered names.
    pub fn names(&self) -> Vec<String> {
        self.factories.keys().cloned().collect()
    }

    /// Whether `name` (its part before `:`) is an objective.
    pub fn contains(&self, name: &str) -> bool {
        let head = name.split_once(':').map_or(name, |(h, _)| h);
        self.factories
            .contains_key(&head.trim().to_ascii_lowercase())
    }

    /// The objective `name`.
    pub fn create(
        &self,
        name: &str,
        options: &CalculatorOptions,
    ) -> MoptResult<Box<dyn Objective>> {
        Ok(Box::new(CountingObjective::new(
            self.create_uncounted(name, options)?,
            self.counters.clone(),
        )))
    }

    /// The objective `name`, whose evaluations are not counted in [`Self::counts`]
    /// (the maps of a surface in the figures, which are drawings and not part of the
    /// search).
    pub fn create_uncounted(
        &self,
        name: &str,
        options: &CalculatorOptions,
    ) -> MoptResult<Box<dyn Objective>> {
        let name = name.trim();
        let (head, argument) = name.split_once(':').unwrap_or((name, ""));
        let f = self
            .factories
            .get(&head.to_ascii_lowercase())
            .ok_or_else(|| {
                MoptError::config(format!(
                    "unknown objective '{head}' (known: {})",
                    self.names().join(", ")
                ))
            })?;
        f(&ObjectiveRequest {
            name,
            argument,
            options,
        })
    }

    /// The evaluations the objectives created by this registry (and its clones) made so
    /// far.
    pub fn counts(&self) -> EvaluationCounts {
        self.counters.counts()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::pes::native::muller_brown::{MB_MINIMA, MB_SADDLES};

    #[test]
    fn muller_brown_stationary_points_and_derivatives() {
        let mb = MullerBrown2D::default();
        for p in MB_MINIMA.iter().chain(MB_SADDLES.iter()) {
            let v = mb.evaluate(&Col::from_fn(2, |i| p[i]), true).unwrap();
            assert!(v.gradient.iter().all(|g| g.abs() < 2e-6), "{p:?}");
        }
        let x = Col::from_fn(2, |i| [0.3, 0.7][i]);
        let v = mb.evaluate(&x, true).unwrap();
        let h = 1e-6;
        for k in 0..2 {
            let mut xp = x.clone();
            xp[k] += h;
            let mut xm = x.clone();
            xm[k] -= h;
            let fd = (mb.evaluate(&xp, false).unwrap().value
                - mb.evaluate(&xm, false).unwrap().value)
                / (2.0 * h);
            assert!((fd - v.gradient[k]).abs() < 1e-8);
        }
        assert!(mb.evaluate(&Col::zeros(3), false).is_err());
    }

    #[test]
    fn expression_and_quadratic_objectives() {
        let reg = ObjectiveRegistry::builtin();
        let opts = CalculatorOptions::default();
        let e = reg
            .create("expr:(q1 - 1)^2 + 3*q2^2 + q1*q3", &opts)
            .unwrap();
        let x = Col::from_fn(3, |i| [2.0, 0.5, -1.0][i]);
        let v = e.evaluate(&x, true).unwrap();
        assert!((v.value - (1.0 + 0.75 - 2.0)).abs() < 1e-14);
        assert!((v.gradient[0] - (2.0 + -1.0)).abs() < 1e-14);
        assert!((v.gradient[1] - 3.0).abs() < 1e-14);
        assert!((v.gradient[2] - 2.0).abs() < 1e-14);
        let h = v.hessian.unwrap();
        assert!((h[(0, 2)] - 1.0).abs() < 1e-14 && (h[(1, 1)] - 6.0).abs() < 1e-14);
        assert!(e.evaluate(&Col::zeros(2), false).is_err());
        assert!(reg.create("expr:dist(1, 2)", &opts).is_err());
        assert!(reg.create("expr:", &opts).is_err());
        assert!(reg.contains("MB2D") && reg.contains("expr:q1") && !reg.contains("lj"));
        for name in [
            "wolfe_quapp",
            "three_hole",
            "himmelblau",
            "rosenbrock",
            "styblinski_tang",
        ] {
            let o = reg.create(name, &opts).unwrap();
            assert_eq!(o.name(), name);
        }
        let mut scaled = CalculatorOptions::default();
        scaled
            .extra
            .push(("energy_scale".to_string(), "0.5".to_string()));
        let h = reg.create("himmelblau", &scaled).unwrap();
        let v = h.evaluate(&Col::zeros(2), false).unwrap();
        assert!((v.value - 0.5 * (121.0 + 49.0)).abs() < 1e-12);
        scaled.extra.push(("nonsense".to_string(), "1".to_string()));
        assert!(reg.create("rosenbrock", &scaled).is_err());

        let a = Mat::from_fn(2, 2, |i, j| [[2.0, 0.5], [0.5, 1.0]][i][j]);
        let q = Quadratic::new(a, Col::from_fn(2, |i| [1.0, -1.0][i]), 0.25).unwrap();
        let v = q
            .evaluate(&Col::from_fn(2, |i| [1.0, -1.0][i]), false)
            .unwrap();
        assert_eq!(v.value, 0.25);
        let bad = Mat::from_fn(2, 2, |i, j| [[2.0, 0.5], [0.4, 1.0]][i][j]);
        assert!(Quadratic::new(bad, Col::zeros(2), 0.0).is_err());
    }
}
