//! Model surfaces of plain variables, for tests of the optimizers, saddle searches, path
//! methods, IRC, and the reaction-network mapper on surfaces whose stationary points are known:
//!
//! - `wolfe_quapp` (two variables): `x^4 + y^4 - 2x^2 - 4y^2 + xy + 0.3x + 0.1y`, the surface on
//!   which Quapp and co-workers study valley-ridge inflections, Newton trajectories, and the
//!   gentlest ascent dynamics (Bofill, Quapp, Caballero, Chem. Phys. Lett. 2013, a minimum at
//!   (1.124, -1.486)): three minima, three first-order saddle points, one maximum.
//! - `three_hole` (two variables): the three-hole potential of Park, Sener, Lu, Schulten (J. Chem.
//!   Phys. 2003, 119, 1313) and of Metzner, Schuette, Vanden-Eijnden (J. Chem. Phys. 2006, 125,
//!   084110): two deep minima near `(+-1, 0)`, a shallow one near `(0, 1.5)`, three saddle points
//!   and a maximum between them.
//! - `himmelblau` (two variables): `(x^2 + y - 11)^2 + (x + y^2 - 7)^2` (Himmelblau, Applied
//!   Nonlinear Programming, McGraw-Hill 1972): four minima of value 0, four saddle points, one
//!   maximum.
//! - `rosenbrock` (any number `n >= 2`): `sum_i 100 (x_{i+1} - x_i^2)^2 + (1 - x_i)^2`
//!   (Rosenbrock, Comput. J. 1960, 3, 175); with four to seven variables it has a second minimum
//!   besides `(1, ..., 1)` and a saddle point between the two.
//! - `styblinski_tang` (any number `n >= 1`): `sum_i (x_i^4 - 16 x_i^2 + 5 x_i) / 2` (Styblinski,
//!   Tang, Neural Networks 1990, 3, 467): a sum of one quartic per variable, so its `3^n`
//!   stationary points are known exactly, `C(n, k) 2^(n-k)` of them of index `k`.
//!
//! Every one takes `energy_scale` (`-calc_opt energy_scale=S`, 1 by default: the values are
//! those of the definitions). The stationary points listed here were located by Newton's
//! method from grids of starting points and classified by the eigenvalues of the Hessian.
//!
//! MultiOptPy: no counterpart (new in mopt-rs) (test surfaces).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use crate::error::{MoptError, MoptResult};
use crate::math::ad::{eval_value_and_gradient, eval_value_grad_hessian, Scalar};
use crate::pes::objective::{check_dimension, Objective, ObjectiveValue};

/// A stationary point of a model surface: its index (number of negative curvatures), its
/// value (with `energy_scale` 1), and where it lies.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KnownPoint<const N: usize> {
    /// Negative eigenvalues of the Hessian (0: minimum, 1: transition state, ...).
    pub index: usize,
    /// The value of the surface there.
    pub value: f64,
    /// The variables.
    pub at: [f64; N],
}

const fn point<const N: usize>(index: usize, value: f64, at: [f64; N]) -> KnownPoint<N> {
    KnownPoint { index, value, at }
}

/// The seven stationary points of `wolfe_quapp`.
pub const WOLFE_QUAPP_POINTS: [KnownPoint<2>; 7] = [
    point(0, -6.762_452_573_7, [-1.174_056_05, 1.477_087_06]),
    point(0, -6.368_956_506_6, [1.124_101_75, -1.485_274_27]),
    point(0, -4.137_203_122_9, [-0.821_907_82, -1.366_729_59]),
    point(1, -3.980_303_235_5, [-0.303_210_56, -1.401_337_58]),
    point(1, -1.251_312_365_0, [-1.022_244_48, -0.116_062_27]),
    point(1, -0.636_563_647_2, [0.940_969_47, 0.131_251_72]),
    point(2, 0.013_268_946_8, [0.081_199_31, 0.022_655_73]),
];

/// The seven stationary points of `three_hole`.
pub const THREE_HOLE_POINTS: [KnownPoint<2>; 7] = [
    point(0, -3.994_860_601_9, [1.048_054_99, -0.042_093_67]),
    point(0, -3.994_860_601_9, [-1.048_054_99, -0.042_093_67]),
    point(0, -2.172_153_623_0, [0.0, 1.537_082_00]),
    point(1, -1.646_687_453_5, [-0.617_272_30, 1.102_734_52]),
    point(1, -1.646_687_453_5, [0.617_272_30, 1.102_734_52]),
    point(1, -1.384_586_640_4, [0.0, -0.315_826_55]),
    point(2, -0.715_229_887_0, [0.0, 0.519_186_74]),
];

/// The nine stationary points of `himmelblau`.
pub const HIMMELBLAU_POINTS: [KnownPoint<2>; 9] = [
    point(0, 0.0, [3.0, 2.0]),
    point(0, 0.0, [-2.805_118_09, 3.131_312_52]),
    point(0, 0.0, [3.584_428_34, -1.848_126_52]),
    point(0, 0.0, [-3.779_310_25, -3.283_185_99]),
    point(1, 13.311_926_270_4, [3.385_154_18, 0.073_851_88]),
    point(1, 67.719_150_087_5, [0.086_677_50, 2.884_254_70]),
    point(1, 104.015_162_917_6, [-3.073_025_75, -0.081_353_04]),
    point(1, 178.337_239_201_9, [-0.127_961_35, -1.953_714_98]),
    point(2, 181.616_521_522_6, [-0.270_844_59, -0.923_038_56]),
];

/// The stationary points of `rosenbrock` with four variables (in `[-1.5, 1.5]^4`).
pub const ROSENBROCK_4_POINTS: [KnownPoint<4>; 3] = [
    point(0, 0.0, [1.0, 1.0, 1.0, 1.0]),
    point(
        0,
        3.701_428_610_430_0,
        [
            -0.775_659_226_565_4,
            0.613_093_365_485_0,
            0.382_062_846_338_4,
            0.145_972_018_552_2,
        ],
    ),
    point(
        1,
        3.708_241_996_647_3,
        [
            -0.656_124_635_719_4,
            0.443_120_040_971_6,
            0.204_312_248_228_4,
            0.041_743_494_776_1,
        ],
    ),
];

/// The stationary points of `rosenbrock` with five variables (in `[-1.5, 1.5]^5`).
pub const ROSENBROCK_5_POINTS: [KnownPoint<5>; 3] = [
    point(0, 0.0, [1.0, 1.0, 1.0, 1.0, 1.0]),
    point(
        0,
        3.930_839_434_133_0,
        [
            -0.962_051_020_694_8,
            0.935_739_395_976_7,
            0.880_713_604_194_3,
            0.777_877_675_854_4,
            0.605_093_678_592_7,
        ],
    ),
    point(
        1,
        4.657_371_428_200_6,
        [
            -0.557_468_315_898_8,
            0.324_740_042_833_8,
            0.116_567_321_445_3,
            0.023_354_396_464_3,
            0.000_545_427_834_2,
        ],
    ),
];

/// The three roots of the derivative of one term of `styblinski_tang`, `2x^3 - 16x + 5/2`:
/// the global minimum, the maximum, and the other minimum.
pub const STYBLINSKI_TANG_ROOTS: [f64; 3] = [-2.903_534_027_771, 0.156_731_259_3, 2.746_802_768_5];

/// `wolfe_quapp` at `(x, y)` (energy scale 1).
pub fn wolfe_quapp<T: Scalar>(x: T, y: T) -> T {
    x.powi(4) + y.powi(4) - T::from_f64(2.0) * x * x - T::from_f64(4.0) * y * y
        + x * y
        + T::from_f64(0.3) * x
        + T::from_f64(0.1) * y
}

/// `three_hole` at `(x, y)` (energy scale 1).
pub fn three_hole<T: Scalar>(x: T, y: T) -> T {
    let well = |dx: T, dy: T| (-(dx * dx) - dy * dy).exp();
    let third = T::from_f64(1.0 / 3.0);
    let one = T::one();
    T::from_f64(3.0) * well(x, y - third)
        - T::from_f64(3.0) * well(x, y - T::from_f64(5.0 / 3.0))
        - T::from_f64(5.0) * well(x - one, y)
        - T::from_f64(5.0) * well(x + one, y)
        + T::from_f64(0.2) * x.powi(4)
        + T::from_f64(0.2) * (y - third).powi(4)
}

/// `himmelblau` at `(x, y)` (energy scale 1).
pub fn himmelblau<T: Scalar>(x: T, y: T) -> T {
    let a = x * x + y - T::from_f64(11.0);
    let b = x + y * y - T::from_f64(7.0);
    a * a + b * b
}

/// `rosenbrock` at `x` (energy scale 1).
pub fn rosenbrock<T: Scalar>(x: &[T]) -> T {
    let mut v = T::zero();
    for w in x.windows(2) {
        let a = w[1] - w[0] * w[0];
        let b = T::one() - w[0];
        v += T::from_f64(100.0) * a * a + b * b;
    }
    v
}

/// `styblinski_tang` at `x` (energy scale 1).
pub fn styblinski_tang<T: Scalar>(x: &[T]) -> T {
    let mut v = T::zero();
    for &q in x {
        v += q.powi(4) - T::from_f64(16.0) * q * q + T::from_f64(5.0) * q;
    }
    T::from_f64(0.5) * v
}

/// Which model surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Model {
    /// [`wolfe_quapp`].
    WolfeQuapp,
    /// [`three_hole`].
    ThreeHole,
    /// [`himmelblau`].
    Himmelblau,
    /// [`rosenbrock`].
    Rosenbrock,
    /// [`styblinski_tang`].
    StyblinskiTang,
}

impl Model {
    /// Every model, in the order of the documentation.
    pub const ALL: [Model; 5] = [
        Model::WolfeQuapp,
        Model::ThreeHole,
        Model::Himmelblau,
        Model::Rosenbrock,
        Model::StyblinskiTang,
    ];

    /// The registered name.
    pub fn name(self) -> &'static str {
        match self {
            Model::WolfeQuapp => "wolfe_quapp",
            Model::ThreeHole => "three_hole",
            Model::Himmelblau => "himmelblau",
            Model::Rosenbrock => "rosenbrock",
            Model::StyblinskiTang => "styblinski_tang",
        }
    }

    /// The number of variables it takes (`None`: any number from [`Self::fewest`]).
    pub fn dimension(self) -> Option<usize> {
        match self {
            Model::WolfeQuapp | Model::ThreeHole | Model::Himmelblau => Some(2),
            Model::Rosenbrock | Model::StyblinskiTang => None,
        }
    }

    /// The fewest variables it takes.
    pub fn fewest(self) -> usize {
        match self {
            Model::StyblinskiTang => 1,
            _ => 2,
        }
    }

    /// Where it is defined.
    pub fn citation(self) -> &'static str {
        match self {
            Model::WolfeQuapp => {
                "the Wolfe-Quapp surface (J. M. Bofill, W. Quapp, M. Caballero, Chem. Phys. Lett. \
                 2013)"
            }
            Model::ThreeHole => {
                "the three-hole potential (S. Park, M. K. Sener, D. Lu, K. Schulten, J. Chem. \
                 Phys. 2003, 119, 1313; P. Metzner, C. Schuette, E. Vanden-Eijnden, J. Chem. Phys. \
                 2006, 125, 084110)"
            }
            Model::Himmelblau => {
                "Himmelblau's function (D. M. Himmelblau, Applied Nonlinear Programming, \
                 McGraw-Hill 1972)"
            }
            Model::Rosenbrock => {
                "the Rosenbrock function (H. H. Rosenbrock, Comput. J. 1960, 3, 175)"
            }
            Model::StyblinskiTang => {
                "the Styblinski-Tang function (M. A. Styblinski, T.-S. Tang, Neural Networks \
                 1990, 3, 467)"
            }
        }
    }

    fn value<T: Scalar>(self, x: &[T]) -> T {
        match self {
            Model::WolfeQuapp => wolfe_quapp(x[0], x[1]),
            Model::ThreeHole => three_hole(x[0], x[1]),
            Model::Himmelblau => himmelblau(x[0], x[1]),
            Model::Rosenbrock => rosenbrock(x),
            Model::StyblinskiTang => styblinski_tang(x),
        }
    }
}

/// A model surface as an objective of plain variables.
#[derive(Debug, Clone, Copy)]
pub struct ModelObjective {
    /// Which surface.
    pub model: Model,
    /// Factor applied to the values of the definition.
    pub energy_scale: f64,
}

impl ModelObjective {
    /// `model` with the energy scale 1.
    pub fn new(model: Model) -> Self {
        Self {
            model,
            energy_scale: 1.0,
        }
    }
}

impl Objective for ModelObjective {
    fn name(&self) -> &str {
        self.model.name()
    }

    fn description(&self) -> String {
        format!(
            "{}: {}, built into mopt-rs {}",
            self.model.name(),
            self.model.citation(),
            crate::VERSION
        )
    }

    fn dimension(&self) -> Option<usize> {
        self.model.dimension()
    }

    fn variable_names(&self, n: usize) -> Vec<String> {
        match self.model.dimension() {
            Some(2) => vec!["x".into(), "y".into()],
            _ => (1..=n).map(|k| format!("x{k}")).collect(),
        }
    }

    fn evaluate(&self, x: &Col<f64>, need_hessian: bool) -> MoptResult<ObjectiveValue> {
        check_dimension(self, x.nrows())?;
        let n = x.nrows();
        if n < self.model.fewest() {
            return Err(MoptError::config(format!(
                "objective '{}' takes at least {} variables, got {n}",
                self.model.name(),
                self.model.fewest()
            )));
        }
        let p: Vec<f64> = x.iter().copied().collect();
        let s = self.energy_scale;
        let model = self.model;
        if need_hessian {
            let (e, g, h) = eval_value_grad_hessian(&p, None, |v| model.value(v));
            Ok(ObjectiveValue {
                value: s * e,
                gradient: Col::from_fn(n, |i| s * g[i]),
                hessian: Some(Mat::from_fn(n, n, |i, j| s * h[(i, j)])),
            })
        } else {
            let (e, g) = eval_value_and_gradient(&p, |v| model.value(v));
            Ok(ObjectiveValue {
                value: s * e,
                gradient: Col::from_fn(n, |i| s * g[i]),
                hessian: None,
            })
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::math::linalg::symmetric_eigen;

    /// The gradient at `p` vanishes and the Hessian there has `index` negative eigenvalues.
    fn check(model: Model, p: &[f64], index: usize, value: f64) {
        let o = ModelObjective::new(model);
        let v = o.evaluate(&Col::from_fn(p.len(), |i| p[i]), true).unwrap();
        let largest = v.gradient.iter().fold(0.0_f64, |m, g| m.max(g.abs()));
        assert!(
            largest < 1e-6,
            "{}: gradient {largest:e} at {p:?}",
            model.name()
        );
        assert!(
            (v.value - value).abs() < 1e-8,
            "{}: {} at {p:?}",
            model.name(),
            v.value
        );
        let (eigenvalues, _) = symmetric_eigen(&v.hessian.unwrap()).unwrap();
        let negative = eigenvalues.iter().filter(|e| **e < 0.0).count();
        assert_eq!(
            negative,
            index,
            "{}: {eigenvalues:?} at {p:?}",
            model.name()
        );
    }

    #[test]
    fn the_known_points_are_stationary_with_their_index() {
        for (model, points) in [
            (Model::WolfeQuapp, &WOLFE_QUAPP_POINTS[..]),
            (Model::ThreeHole, &THREE_HOLE_POINTS[..]),
            (Model::Himmelblau, &HIMMELBLAU_POINTS[..]),
        ] {
            for p in points {
                check(model, &p.at, p.index, p.value);
            }
        }
        for p in &ROSENBROCK_4_POINTS {
            check(Model::Rosenbrock, &p.at, p.index, p.value);
        }
        for p in &ROSENBROCK_5_POINTS {
            check(Model::Rosenbrock, &p.at, p.index, p.value);
        }
    }

    #[test]
    fn styblinski_tang_has_every_index_in_its_known_numbers() {
        // Four variables: every choice of a root per variable is a stationary point whose index
        // is the number of variables at the maximum of their term.
        let n = 4;
        let mut counts = [0_usize; 5];
        for code in 0..3_usize.pow(n as u32) {
            let mut c = code;
            let p: Vec<f64> = (0..n)
                .map(|_| {
                    let r = STYBLINSKI_TANG_ROOTS[c % 3];
                    c /= 3;
                    r
                })
                .collect();
            let at_max = p
                .iter()
                .filter(|x| (**x - STYBLINSKI_TANG_ROOTS[1]).abs() < 1e-9)
                .count();
            let value = styblinski_tang(&p);
            check(Model::StyblinskiTang, &p, at_max, value);
            counts[at_max] += 1;
        }
        assert_eq!(counts, [16, 32, 24, 8, 1]);
        // The global minimum: -39.16617 per variable.
        let lowest = styblinski_tang(&[STYBLINSKI_TANG_ROOTS[0]; 3]);
        assert!((lowest / 3.0 + 39.166_165_7).abs() < 1e-6, "{lowest}");
    }

    #[test]
    fn derivatives_match_differences_and_dimensions_are_checked() {
        let h = 1e-6;
        for model in Model::ALL {
            let n = model.dimension().unwrap_or(5);
            let o = ModelObjective {
                model,
                energy_scale: 0.5,
            };
            let x = Col::from_fn(n, |i| 0.3 + 0.17 * i as f64);
            let v = o.evaluate(&x, true).unwrap();
            let hess = v.hessian.clone().unwrap();
            for k in 0..n {
                let mut xp = x.clone();
                xp[k] += h;
                let mut xm = x.clone();
                xm[k] -= h;
                let (p, m) = (
                    o.evaluate(&xp, false).unwrap(),
                    o.evaluate(&xm, false).unwrap(),
                );
                let fd = (p.value - m.value) / (2.0 * h);
                assert!(
                    (fd - v.gradient[k]).abs() < 1e-6 * (1.0 + fd.abs()),
                    "{}",
                    model.name()
                );
                for j in 0..n {
                    let fd = (p.gradient[j] - m.gradient[j]) / (2.0 * h);
                    assert!(
                        (fd - hess[(j, k)]).abs() < 1e-5 * (1.0 + fd.abs()),
                        "{} {j} {k}",
                        model.name()
                    );
                }
            }
        }
        assert!(ModelObjective::new(Model::Himmelblau)
            .evaluate(&Col::zeros(3), false)
            .is_err());
        assert!(ModelObjective::new(Model::Rosenbrock)
            .evaluate(&Col::zeros(1), false)
            .is_err());
        assert!(ModelObjective::new(Model::StyblinskiTang)
            .evaluate(&Col::zeros(0), false)
            .is_err());
    }
}
