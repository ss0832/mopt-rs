//! The Zhu-Nakamura transmission through a crossing of the nonadiabatic-tunnelling type (a
//! peaked crossing: the diabatic energies have slopes of opposite signs along the hopping
//! coordinate, the lower adiabatic energy has a barrier with its top `E_t` below the crossing
//! energy `E_X`, and the upper one a well with its bottom `E_b` above it), at every energy:
//! above the well bottom, between `E_t` and `E_b`, and below the barrier top (nonadiabatic
//! tunnelling).
//!
//! With the coupling `V`, the slopes `F1 > 0 > F2`, `F = sqrt(|F1 F2|)`, and the reduced mass
//! `mu` along the hopping coordinate (atomic units, `hbar = 1`):
//!
//! ```text
//! a^2 = F (F1 - F2) / (16 mu V^3),    b^2 = (E - E_X) (F1 - F2) / (2 F V),
//! ```
//!
//! so that `b^2 = -1` at `E_t` and `b^2 = 1` at `E_b` (for straight diabatic states
//! `E_b, E_t = E_X +- 2 F V / (F1 - F2)`). The probability that matters for a step is the
//! overall transmission `P12` from one side of the crossing to the other along the lower
//! adiabatic energy, in three zones:
//!
//! - `E >= E_b` (`b^2 >= 1`): the probability that one passage changes the adiabatic state,
//!   `p = exp(-(pi / 4a) (2 / (b^2 + sqrt(b^4 - 0.72 + 0.62 a^1.43)))^(1/2))`, and with the
//!   upper well closed `P12 = 4 cos^2 phi / (4 cos^2 phi + p^2 / (1 - p))`. Here the phase
//!   `phi` is averaged, as for the sloped type ([`super::zhu_nakamura`]): the average is
//!   `1 - sqrt(q / (q + 4))` with `q = p^2 / (1 - p)`, which is `2 (1 - p) / (2 - p)`, the
//!   incoherent sum over all the passages through the well. (The phase of a straight model
//!   would put the resonances of a well that a molecule does not have.)
//! - `E_t <= E <= E_b`: `P12 = W^2 / (1 + W^2)` with
//!   `W = (h2 / a^(2/3)) int_0^inf cos(t^3/3 - b^2 t / a^(2/3) - h3 t / (a^(2/3) (h4 + a^(1/3) t))) dt`,
//!   `h2 = 1 + 0.38 (1 + b^2)^(1.2 - 0.4 b^2) / a^2`, `h3 = (a - 3 b^2) / (a + 3) sqrt(1.23 + b^2)`,
//!   `h4 = 0.61 sqrt(2 + b^2)`. The integral is taken along the real axis to beyond its
//!   stationary point and then along a ray at `pi / 6`, where the integrand decays.
//! - `E <= E_t`: with `Be = B(sigma_c / pi) exp(-2 delta)`,
//!   `P12 = Be / ((1 + a / (2 (1 + a)) Be)^2 + Be)`, `B(x) = 2 pi x^(2x) exp(-2x) / (x Gamma(x)^2)`,
//!   `sigma = (pi / (16 a |b|)) sqrt(6 + 10 sqrt(1 - 1/b^4)) / (1 + sqrt(1 - 1/b^4))`,
//!   `sigma_c = sigma (1 - 0.32 10^(-2 / a^2) exp(-delta))`, and `delta = int |K1(R)| dR`, the
//!   action under the barrier of the lower adiabatic energy between its turning points. Here
//!   `delta` is that of the straight model: with `k = (F1 + F2) / (F1 - F2)`,
//!   `s = sqrt(1 - k^2) = 2 F / (F1 - F2)` and `beta = -b^2`,
//!   `delta = (sqrt(s) / (2a)) int sqrt(beta s + k w - sqrt(1 + w^2)) dw` between the turning
//!   points `w = (k beta -+ sqrt(beta^2 - 1)) / s`.
//!
//! The rate correction `zhu_nakamura` of a peaked crossing is the thermal average referred to
//! the crossing point, like the Eckart correction of a saddle point,
//! `kappa = (1 / k_B T) int P12(E) exp(-(E - E_X) / k_B T) dE` from the higher of the two
//! minima (the lower of the two barriers) upwards, so that it is the same from both sides; it
//! exceeds 1 when tunnelling below the crossing dominates.
//!
//! Limits, checked in the tests: far above the crossing `p` is Landau-Zener's
//! `exp(-pi / (4 a b))` (one passage) and the averaged `P12` is `1 - p^2` (the double passage
//! of [`super::crossing`]) to `O(1 - p)`; for a weak coupling the same holds at energies well
//! above the crossing on the energy scale of the straight model (`x = b^2 / a^(2/3)`, the
//! argument of its Airy function, much larger than 1). Within a few units of `x` Landau-Zener
//! does not hold (the Airy region) and the empirical factor `0.72 - 0.62 a^1.43` makes the
//! exponent of `p` smaller than Landau-Zener's by a few per cent (at `x = 2.3`: 2 % for
//! `V = 1e-4` and 5 % for `V = 1e-7` Hartree, measured); its weight in a thermal average
//! falls as `sqrt` of the energy scale over `k_B T`. For a strong coupling (`a^2 -> 0`) below
//! the barrier top `P12` is the transmission of the single barrier,
//! `exp(-2 delta) / (1 + exp(-2 delta))`.
//!
//! The adiabatic form ([`peaked_model`]): with the top `E_t` of the lower adiabatic energy at
//! `R_t`, the bottom `E_b` of the upper one at `R_b`, and `gamma = (E_b - E_t) / (E2 - E1)` at
//! `(R_b + R_t) / 2`, `a^2 = (1 - gamma^2) / (mu (R_b - R_t)^2 (E_b - E_t))`, or for
//! `R_b = R_t`, `a^2 = (E2''(R_b) - E1''(R_t)) / (4 mu (E_b - E_t)^2)`, and
//! `b^2 = (E - (E_b + E_t) / 2) / ((E_b - E_t) / 2)`. For straight diabatic states both give
//! the diabatic `a^2` (the curvature form at every asymmetry), `gamma = s`, `V` half the gap
//! at the midpoint.
//!
//! References: the formulas as given by T. Ishida, S. Nanbu, H. Nakamura, Int. Rev. Phys.
//! Chem. 2017, 36, 229 (the authors' draft, eqs. 2.1, 2.2, 2.7-2.10, 2.26, 2.27, 2.54-2.62)
//! and by D. Shchepanovska, R. J. Shannon, B. F. E. Curchod, D. R. Glowacki, J. Phys. Chem. A
//! 2021, 125, 3473, supporting information (the three zones as implemented in MESMER); the
//! two agree. The theory: C. Zhu, H. Nakamura, J. Chem. Phys. 1994, 101, 10630 (not read).
//! The phase average, the straight-model `delta`, the contour of `W` and the limits are
//! derived here and checked in the tests.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::f64::consts::PI;

use super::zhu_nakamura::at;
use crate::domain::Units;
use crate::kinetics::quadrature::{gauss_legendre, ln_integral};
use crate::math::stable::log_sum_exp;

/// The Zhu-Nakamura parameters of a peaked crossing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PeakedCrossing {
    /// `a^2` (dimensionless).
    pub a2: f64,
    /// `b^2 / (E - E_X)` (per Hartree).
    pub b2_per_hartree: f64,
    /// `k^2 = ((F1 + F2) / (F1 - F2))^2` in `[0, 1)`: how unequal the two slopes are.
    pub asymmetry: f64,
}

/// A thermal average of the transmission of a peaked crossing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PeakedAverage {
    /// `ln kappa`.
    pub ln_kappa: f64,
    /// The share of `kappa` from the energies below the crossing energy.
    pub below_crossing: f64,
}

impl PeakedCrossing {
    /// From the two-state model: the coupling `coupling` (Hartree), `|F1 - F2|`
    /// (`slope_difference`, Hartree/Bohr), `F1 F2` (`slope_product`, negative), and the
    /// reduced mass `mass_amu`. `None` for a sloped crossing (a product that is not negative),
    /// for values that are not finite and positive, and for slopes that cannot be (a
    /// difference below `2 sqrt(|F1 F2|)`).
    pub fn from_diabatic(
        coupling: f64,
        slope_difference: f64,
        slope_product: f64,
        mass_amu: f64,
    ) -> Option<Self> {
        let v = coupling.abs();
        let df = slope_difference.abs();
        let ok = |x: f64| x.is_finite() && x > 0.0;
        if !(ok(v) && ok(df) && ok(-slope_product) && ok(mass_amu)) {
            return None;
        }
        let f = (-slope_product).sqrt();
        let asymmetry = 1.0 - 4.0 * f * f / (df * df);
        if !(0.0..1.0).contains(&asymmetry) {
            return None;
        }
        let mass = mass_amu * Units::AMU_TO_AU_MASS;
        Some(Self {
            a2: f * df / (16.0 * mass * v.powi(3)),
            b2_per_hartree: df / (2.0 * f * v),
            asymmetry,
        })
    }

    /// From the adiabatic energies `lower` and `upper` (Hartree) at the points `arc` of a
    /// path, as [`super::ZhuNakamuraCrossing::from_adiabatic`] (see [`peaked_model`]).
    pub fn from_adiabatic(
        arc: &[f64],
        lower: &[f64],
        upper: &[f64],
        mass_amu: f64,
    ) -> Option<Self> {
        let m = peaked_model(arc, lower, upper)?;
        Self::from_diabatic(
            m.coupling,
            m.slopes[0] - m.slopes[1],
            m.slopes[0] * m.slopes[1],
            mass_amu,
        )
    }

    /// `b^2` at the energy `above` (Hartree) above the crossing (negative below it).
    pub fn b2(&self, above: f64) -> f64 {
        self.b2_per_hartree * above
    }

    /// `-ln p` at `b^2 >= 1`, `p` being the probability that one passage changes the
    /// adiabatic state.
    pub fn minus_ln_hop(&self, b2: f64) -> f64 {
        let a = self.a2.sqrt();
        let inner = b2 + (b2 * b2 - 0.72 + 0.62 * a.powf(1.43)).sqrt();
        PI / (4.0 * a) * (2.0 / inner).sqrt()
    }

    /// `ln P12` at `b^2`: the overall transmission from one side to the other, with the
    /// phase averaged above the well bottom.
    pub fn ln_transmission(&self, b2: f64) -> f64 {
        let a = self.a2.sqrt();
        if b2 >= 1.0 {
            // 2 (1 - p) / (2 - p), 1 - p = -expm1(-x).
            let q = -(-self.minus_ln_hop(b2)).exp_m1();
            return 2f64.ln() + q.ln() - q.ln_1p();
        }
        if b2 > -1.0 {
            let w = w_integral(a, b2);
            return 2.0 * w.abs().ln() - (w * w).ln_1p();
        }
        let delta = straight_delta(a, b2, self.asymmetry);
        let r = (1.0 - 1.0 / (b2 * b2)).max(0.0).sqrt();
        let sigma = PI / (16.0 * a * (-b2).sqrt()) * (6.0 + 10.0 * r).sqrt() / (1.0 + r);
        let sigma_c = sigma
            * (1.0 - 0.32 * (-2.0 * std::f64::consts::LN_10 / self.a2).exp() * (-delta).exp());
        let ln_be = ln_b_function(sigma_c / PI) - 2.0 * delta;
        let be = ln_be.exp();
        let c = a / (2.0 * (1.0 + a));
        ln_be - ((1.0 + c * be).powi(2) + be).ln()
    }

    /// The thermal average of the transmission at `kt` (Hartree) from `lowest` (Hartree,
    /// the lower of the two barriers to the crossing) below the crossing energy upwards.
    pub fn average(&self, kt: f64, lowest: f64) -> PeakedAverage {
        let k = self.b2_per_hartree;
        let (top, bottom) = (-1.0 / k, 1.0 / k);
        let from = -lowest;
        let ln_p = |e: f64| self.ln_transmission(k * e);
        let mut above = Vec::new();
        let mut below = Vec::new();
        // Above the well bottom: E - E_X = start + kT s^2.
        let start = bottom.max(from);
        above.push(
            -start / kt
                + ln_integral(
                    |s| (2.0 * s).ln() - s * s + ln_p(start + kt * s * s),
                    0.0,
                    9.0,
                    200,
                    16,
                ),
        );
        // Between the barrier top and the well bottom, split at the crossing energy.
        let between = |lo: f64, hi: f64| ln_integral(|e| ln_p(e) - e / kt - kt.ln(), lo, hi, 8, 16);
        let low = top.max(from);
        if low < 0.0 {
            below.push(between(low, 0.0));
        }
        if low.max(0.0) < bottom {
            above.push(between(low.max(0.0), bottom));
        }
        // Below the barrier top: E - E_X = top - kT u.
        if from < top {
            let reach = ((top - from) / kt).min(700.0);
            let panels = ((2.0 * reach).ceil() as usize).clamp(16, 2000);
            below.push(-top / kt + ln_integral(|u| ln_p(top - kt * u) + u, 0.0, reach, panels, 16));
        }
        let ln_above = log_sum_exp(&above);
        let ln_below = log_sum_exp(&below);
        let ln_kappa = log_sum_exp(&[ln_above, ln_below]);
        PeakedAverage {
            ln_kappa,
            below_crossing: if ln_kappa.is_finite() {
                (ln_below - ln_kappa).exp()
            } else {
                0.0
            },
        }
    }
}

/// `ln B(x)`, `B(x) = 2 pi x^(2x) exp(-2x) / (x Gamma(x)^2)`, which rises from `2 pi x` at
/// small `x` to 1; for large `x` from Stirling's series (`-1/(6x) + 1/(180 x^3) - ...`),
/// where the terms of the closed form cancel.
fn ln_b_function(x: f64) -> f64 {
    if x > 10.0 {
        let x2 = x * x;
        return -1.0 / (6.0 * x) + 1.0 / (180.0 * x * x2) - 1.0 / (630.0 * x * x2 * x2);
    }
    (2.0 * PI).ln() + 2.0 * x * x.ln() - 2.0 * x - x.ln() - 2.0 * libm::lgamma(x)
}

/// `delta` of the straight model below the barrier top (`b^2 <= -1`), the asymmetry
/// `k^2` (see the module text).
fn straight_delta(a: f64, b2: f64, asymmetry: f64) -> f64 {
    let beta = -b2;
    let s = (1.0 - asymmetry).max(0.0).sqrt();
    let k = asymmetry.max(0.0).sqrt();
    let q = (beta * beta - 1.0).max(0.0).sqrt();
    let (lo, hi) = ((k * beta - q) / s, (k * beta + q) / s);
    if hi.is_nan() || lo.is_nan() || hi <= lo {
        return 0.0;
    }
    let f = |w: f64| (beta * s + k * w - (1.0 + w * w).sqrt()).max(0.0).sqrt();
    // w = m - h cos(theta) on each side of w = 0, where sqrt(1 + w^2) bends: the nodes gather
    // at the turning points (square-root ends) and at the bend.
    let piece = |lo: f64, hi: f64| {
        let (m, h) = (0.5 * (lo + hi), 0.5 * (hi - lo));
        gauss(|t| f(m - h * t.cos()) * h * t.sin(), 0.0, PI, 8)
    };
    let integral = if lo < 0.0 && hi > 0.0 {
        piece(lo, 0.0) + piece(0.0, hi)
    } else {
        piece(lo, hi)
    };
    s.sqrt() / (2.0 * a) * integral
}

/// `int_lo^hi f` by Gauss-Legendre (16 points) on `panels` panels.
fn gauss(f: impl Fn(f64) -> f64, lo: f64, hi: f64, panels: usize) -> f64 {
    let (x, w) = gauss_legendre(16);
    let h = (hi - lo) / panels as f64;
    let mut sum = 0.0;
    for p in 0..panels {
        let mid = lo + (p as f64 + 0.5) * h;
        for (xi, wi) in x.iter().zip(&w) {
            sum += 0.5 * h * wi * f(mid + 0.5 * h * xi);
        }
    }
    sum
}

/// A complex number `(re, im)` for the contour of `W`.
#[derive(Debug, Clone, Copy)]
struct Cx(f64, f64);

impl Cx {
    fn add(self, o: Cx) -> Cx {
        Cx(self.0 + o.0, self.1 + o.1)
    }
    fn mul(self, o: Cx) -> Cx {
        Cx(self.0 * o.0 - self.1 * o.1, self.0 * o.1 + self.1 * o.0)
    }
    fn scale(self, r: f64) -> Cx {
        Cx(self.0 * r, self.1 * r)
    }
    fn div(self, o: Cx) -> Cx {
        let d = o.0 * o.0 + o.1 * o.1;
        Cx(
            (self.0 * o.0 + self.1 * o.1) / d,
            (self.1 * o.0 - self.0 * o.1) / d,
        )
    }
    /// `exp(i z)`.
    fn exp_i(self) -> Cx {
        let m = (-self.1).exp();
        Cx(m * self.0.cos(), m * self.0.sin())
    }
}

/// `W` between the barrier top and the well bottom (`-1 < b^2 < 1`).
fn w_integral(a: f64, b2: f64) -> f64 {
    let a23 = a.powf(2.0 / 3.0);
    let a13 = a.cbrt();
    let h2 = 1.0 + 0.38 * (1.0 + b2).max(0.0).powf(1.2 - 0.4 * b2) / (a * a);
    let h3 = (a - 3.0 * b2) / (a + 3.0) * (1.23 + b2).sqrt();
    let h4 = 0.61 * (2.0 + b2).sqrt();
    let x = b2 / a23;
    let c = h3 / a23;
    // phi(t) = t^3/3 - x t - c t / (h4 + a^(1/3) t).
    let phase = |t: Cx| {
        let cube = t.mul(t).mul(t).scale(1.0 / 3.0);
        let pole = t.scale(c).div(Cx(h4, 0.0).add(t.scale(a13)));
        cube.add(t.scale(-x)).add(pole.scale(-1.0))
    };
    // Beyond t0 the phase rises along every ray at 0..pi/3: t0^2 > x + |c| / h4 bounds
    // phi'(t) from below there.
    let t0 = (x + c.abs() / h4).max(0.0).sqrt() + 1.0;
    let variation = t0 * (t0 * t0 + x.abs() + c.abs() / h4);
    let panels = ((variation.ceil() as usize) + 4).min(1_000_000);
    let (gx, gw) = gauss_legendre(16);
    let mut sum = Cx(0.0, 0.0);
    let h = t0 / panels as f64;
    for p in 0..panels {
        let mid = (p as f64 + 0.5) * h;
        for (xi, wi) in gx.iter().zip(&gw) {
            let t = Cx(mid + 0.5 * h * xi, 0.0);
            sum = sum.add(phase(t).exp_i().scale(0.5 * h * wi));
        }
    }
    let ray = Cx((PI / 6.0).cos(), (PI / 6.0).sin());
    let (reach, ray_panels) = (8.0, 64);
    let h = reach / ray_panels as f64;
    for p in 0..ray_panels {
        let mid = (p as f64 + 0.5) * h;
        for (xi, wi) in gx.iter().zip(&gw) {
            let t = Cx(t0, 0.0).add(ray.scale(mid + 0.5 * h * xi));
            sum = sum.add(phase(t).exp_i().mul(ray).scale(0.5 * h * wi));
        }
    }
    h2 / a23 * sum.0
}

/// The straight two-state model of a peaked crossing from the adiabatic energies.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PeakedModel {
    /// The crossing point, midway between the barrier top and the well bottom (the unit of
    /// `arc`).
    pub position: f64,
    /// The crossing energy `(E_b + E_t) / 2` (Hartree).
    pub energy: f64,
    /// The coupling, half the gap at the crossing point (Hartree).
    pub coupling: f64,
    /// The slopes `F1 > 0 > F2` of the two diabatic states along `arc`.
    pub slopes: [f64; 2],
}

/// The vertex and the curvature of the parabola through three points.
fn vertex(x: [f64; 3], y: [f64; 3]) -> Option<(f64, f64, f64)> {
    let f01 = (y[1] - y[0]) / (x[1] - x[0]);
    let f12 = (y[2] - y[1]) / (x[2] - x[1]);
    let f012 = (f12 - f01) / (x[2] - x[0]);
    if !(f012.is_finite() && f012 != 0.0) {
        return None;
    }
    let xv = 0.5 * (x[0] + x[1]) - f01 / (2.0 * f012);
    let yv = y[0] + f01 * (xv - x[0]) + f012 * (xv - x[0]) * (xv - x[1]);
    Some((xv, yv, 2.0 * f012))
}

/// The straight two-state model of a peaked crossing from the adiabatic energies `lower`
/// and `upper` at the points `arc` (increasing): the lower energy with a maximum and the
/// upper with a minimum inside the path (each refined by a parabola through its three
/// points), the smallest gap between them, and the top below the bottom. `None` otherwise.
/// The asymmetry comes from `gamma`, `a^2` from the positions, or from the curvatures when
/// the two extrema are less than two points apart.
pub fn peaked_model(arc: &[f64], lower: &[f64], upper: &[f64]) -> Option<PeakedModel> {
    let n = arc.len().min(lower.len()).min(upper.len());
    if n < 3 || arc[..n].windows(2).any(|w| w[1] <= w[0]) {
        return None;
    }
    let (arc, lower, upper) = (&arc[..n], &lower[..n], &upper[..n]);
    let t = (0..n).max_by(|&a, &b| lower[a].total_cmp(&lower[b]))?;
    let b = (0..n).min_by(|&a, &b| upper[a].total_cmp(&upper[b]))?;
    let g = (0..n).min_by(|&a, &b| (upper[a] - lower[a]).total_cmp(&(upper[b] - lower[b])))?;
    if t == 0 || t + 1 >= n || b == 0 || b + 1 >= n {
        return None;
    }
    let three =
        |k: usize, v: &[f64]| vertex([arc[k - 1], arc[k], arc[k + 1]], [v[k - 1], v[k], v[k + 1]]);
    let (r_t, e_t, c_t) = three(t, lower)?;
    let (r_b, e_b, c_b) = three(b, upper)?;
    let spacing = |k: usize| 0.5 * (arc[k + 1] - arc[k - 1]);
    let tolerance = 2.0 * spacing(t).max(spacing(b));
    let (first, last) = (r_t.min(r_b), r_t.max(r_b));
    if !(e_b > e_t && c_t < 0.0 && c_b > 0.0)
        || arc[g] < first - tolerance
        || arc[g] > last + tolerance
    {
        return None;
    }
    let mid = 0.5 * (r_t + r_b);
    let gap = at(arc, upper, mid) - at(arc, lower, mid);
    let gamma = ((e_b - e_t) / gap).min(1.0);
    if gamma.is_nan() || gamma <= 0.0 {
        return None;
    }
    let v = 0.5 * gap;
    let k2 = 1.0 - gamma * gamma;
    let a2_mu = if (r_b - r_t).abs() >= tolerance {
        k2 / ((r_b - r_t).powi(2) * (e_b - e_t))
    } else {
        (c_b - c_t) / (4.0 * (e_b - e_t).powi(2))
    };
    // a^2 mu = s dF^2 / (32 V^3), F = s dF / 2, k = (F1 + F2) / dF with the sign from
    // R_b - R_t = -4 k V / (s dF).
    let df = (32.0 * v.powi(3) * a2_mu / gamma).sqrt();
    let k = if r_b > r_t { -k2.sqrt() } else { k2.sqrt() };
    if !(df.is_finite() && df > 0.0) {
        return None;
    }
    Some(PeakedModel {
        position: mid,
        energy: 0.5 * (e_b + e_t),
        coupling: v,
        slopes: [0.5 * (k + 1.0) * df, 0.5 * (k - 1.0) * df],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kinetics::rate::crossing::landau_zener_ln_average;

    fn crossing(f1: f64, f2: f64, v: f64, mu: f64) -> PeakedCrossing {
        PeakedCrossing::from_diabatic(v, f1 - f2, f1 * f2, mu).unwrap()
    }

    #[test]
    fn far_above_the_crossing_one_and_two_passages_tend_to_landau_zener() {
        for a2 in [0.1, 1.0, 10.0] {
            let zn = PeakedCrossing {
                a2,
                b2_per_hartree: 1.0,
                asymmetry: 0.0,
            };
            let mut last = f64::INFINITY;
            for b2 in [1e2, 1e4, 1e6, 1e8] {
                // One passage: -ln p against pi / (4 a b).
                let lz = PI / (4.0 * (a2 * b2).sqrt());
                let one = zn.minus_ln_hop(b2) / lz;
                assert!((one - 1.0).abs() < 2e-3 / b2.sqrt(), "{a2} {b2} {one}");
                // Two passages: the averaged P12 against 1 - P_LZ^2, to O(1 - P_LZ).
                let p_lz = (-lz).exp();
                let two = zn.ln_transmission(b2).exp() / (1.0 - p_lz * p_lz);
                let off = (two - 1.0).abs();
                assert!(off <= 0.5 * (1.0 - p_lz) + 1e-9, "{a2} {b2} {two}");
                assert!(off < last);
                last = off;
            }
            assert!(last < 1.3e-4, "{a2} {last}");
        }
    }

    #[test]
    fn a_weak_coupling_tends_to_landau_zener_well_above_the_crossing() {
        // F1 = -F2 = 0.01 Hartree/Bohr, 10 amu, 0.01 Hartree above the crossing (x = 11.3).
        let (f1, f2, mu, e) = (0.01, -0.01, 10.0, 0.01);
        for v in [1e-4, 1e-5, 1e-6, 1e-7] {
            let zn = crossing(f1, f2, v, mu);
            let b2 = zn.b2(e);
            assert!((b2 / zn.a2.powf(1.0 / 3.0) - 11.34).abs() < 0.01);
            let lz = PI / (4.0 * (zn.a2 * b2).sqrt());
            let one = zn.minus_ln_hop(b2) / lz;
            assert!((one - 1.0).abs() < 3e-3, "{v} {one}");
            let p_lz = (-lz).exp();
            let two = zn.ln_transmission(b2).exp() / (1.0 - p_lz * p_lz);
            assert!((two - 1.0).abs() < 3e-3 + 0.5 * (1.0 - p_lz), "{v} {two}");
        }
        // Near the crossing (x = 2.3, the Airy region) the empirical factor of p lowers the
        // exponent below Landau-Zener's by a few per cent, more as the coupling falls.
        let mut last = 1.0;
        for v in [1e-4, 1e-5, 1e-6, 1e-7] {
            let zn = crossing(f1, f2, v, mu);
            let b2 = zn.b2(0.002);
            let one = zn.minus_ln_hop(b2) / (PI / (4.0 * (zn.a2 * b2).sqrt()));
            assert!(one < last && one > 0.94, "{v} {one}");
            last = one;
        }
    }

    #[test]
    fn the_thermal_average_tends_to_landau_zener_when_hot() {
        // 100 amu, V = 1e-5 Hartree, barriers of 0.05 Hartree: below the crossing the
        // transmission adds tunnelling and the Airy region falls short of Landau-Zener; both
        // fade as k_B T grows.
        let (f1, f2, mu, v) = (0.01, -0.01, 100.0, 1e-5);
        let zn = crossing(f1, f2, v, mu);
        let mut last_off = f64::INFINITY;
        let mut last_below = 1.0;
        for t in [3000.0, 30000.0, 300000.0] {
            let kt = Units::BOLTZMANN_HARTREE_PER_K * t;
            let avg = zn.average(kt, 0.05);
            let lz = landau_zener_ln_average(v, f1 - f2, mu, kt);
            let off = (avg.ln_kappa - lz).exp() - 1.0;
            assert!(off.abs() < last_off, "{t} {off}");
            assert!(avg.below_crossing < last_below && avg.below_crossing > 0.0);
            last_off = off.abs();
            last_below = avg.below_crossing;
        }
        assert!(last_off < 0.01, "{last_off}");
        // At room temperature a light hopping mass tunnels: a large share from below.
        let light = crossing(f1, f2, 1e-4, 1.0);
        let avg = light.average(Units::BOLTZMANN_HARTREE_PER_K * 300.0, 0.05);
        assert!(avg.below_crossing > 0.5, "{}", avg.below_crossing);
    }

    #[test]
    fn the_three_zones_meet() {
        // The formulas of neighbouring zones agree at b^2 = -1 and 1 within their accuracy
        // (measured: up to 29 %; the phase average above the well bottom adds to it).
        for a2 in [0.01, 0.1, 1.0, 10.0, 100.0] {
            let zn = PeakedCrossing {
                a2,
                b2_per_hartree: 1.0,
                asymmetry: 0.0,
            };
            let at_top = zn.ln_transmission(-1.0 - 1e-12) - zn.ln_transmission(-1.0 + 1e-12);
            let at_bottom = zn.ln_transmission(1.0 - 1e-12) - zn.ln_transmission(1.0);
            assert!(at_top.abs() < 0.3, "{a2} {at_top}");
            assert!(at_bottom.abs() < 0.3, "{a2} {at_bottom}");
        }
    }

    #[test]
    fn w_has_its_airy_limit_for_a_weak_coupling() {
        // W -> pi a^(-2/3) Ai(-b^2 a^(-2/3)); Ai(0) = 0.355028053887817.
        let a = 1e3;
        let got = w_integral(a, 0.0) * a.powf(2.0 / 3.0) / PI;
        assert!((got / 0.355028053887817 - 1.0).abs() < 1e-3, "{got}");
    }

    #[test]
    fn sigma_and_delta_are_the_actions_of_the_straight_model() {
        // sigma below the barrier top against (1 / 2a) int_0^1 [sqrt(beta + c) - sqrt(beta - c)]
        // ds, c = sqrt(1 - s^2) (symmetric slopes).
        let a = 0.7;
        for beta in [1.5, 3.0, 10.0, 100.0] {
            let exact = gauss(
                |s| {
                    let c = (1.0 - s * s).sqrt();
                    (beta + c).sqrt() - (beta - c).sqrt()
                },
                0.0,
                1.0,
                40,
            ) / (2.0 * a);
            let r = (1.0 - 1.0 / (beta * beta)).sqrt();
            let closed = PI / (16.0 * a * beta.sqrt()) * (6.0 + 10.0 * r).sqrt() / (1.0 + r);
            assert!(
                (closed / exact - 1.0).abs() < 2e-4,
                "{beta} {closed} {exact}"
            );
        }
        // delta of unequal slopes against the action under the lower adiabatic energy.
        let (f1, f2, v, mu) = (0.03, -0.01, 0.002, 10.0);
        let zn = crossing(f1, f2, v, mu);
        let mass = mu * Units::AMU_TO_AU_MASS;
        let lower = |x: f64| 0.5 * (f1 + f2) * x - ((0.5 * (f1 - f2) * x).powi(2) + v * v).sqrt();
        for e in [-0.004, -0.01, -0.03] {
            let xs: Vec<f64> = (0..=200_000).map(|i| -5.0 + 5e-5 * i as f64).collect();
            let inside: Vec<f64> = xs.iter().copied().filter(|&x| lower(x) > e).collect();
            let (lo, hi) = (inside[0], inside[inside.len() - 1]);
            let direct = gauss(
                |x| (2.0 * mass * (lower(x) - e)).max(0.0).sqrt(),
                lo,
                hi,
                400,
            );
            let model = straight_delta(zn.a2.sqrt(), zn.b2(e), zn.asymmetry);
            assert!((model / direct - 1.0).abs() < 1e-4, "{e} {model} {direct}");
        }
    }

    #[test]
    fn a_strong_coupling_below_the_top_is_the_single_barrier() {
        // B(sigma / pi) -> 1 as a^2 -> 0 (1 - B ~ 1 / (6 sigma / pi)), and the second term of
        // the denominator vanishes with a.
        let b2 = -1.05;
        let mut last = f64::INFINITY;
        for a2 in [1e-4, 1e-5, 1e-6] {
            let zn = PeakedCrossing {
                a2,
                b2_per_hartree: 1.0,
                asymmetry: 0.0,
            };
            let delta = straight_delta(a2.sqrt(), b2, 0.0);
            let ln_single = -2.0 * delta - (-2.0 * delta).exp().ln_1p();
            let off = (zn.ln_transmission(b2) - ln_single).abs();
            assert!(off < last, "{a2} {off}");
            last = off;
        }
        assert!(last < 2e-3, "{last}");
    }

    #[test]
    fn the_adiabatic_form_gives_the_straight_two_state_model() {
        let arc: Vec<f64> = (0..1601).map(|k| -2.0 + 0.0025 * k as f64).collect();
        // Equal and unequal slopes: the curvature form and the form of the positions.
        for (f1, f2, v) in [(0.02, -0.02, 0.002), (0.03, -0.01, 0.002)] {
            let mean = 0.5 * (f1 + f2);
            let half = 0.5 * (f1 - f2);
            let root = |x: f64| ((half * x).powi(2) + v * v).sqrt();
            let lower: Vec<f64> = arc.iter().map(|&x| mean * x - root(x)).collect();
            let upper: Vec<f64> = arc.iter().map(|&x| mean * x + root(x)).collect();
            let m = peaked_model(&arc, &lower, &upper).unwrap();
            assert!(m.position.abs() < 1e-3, "{}", m.position);
            assert!(m.energy.abs() < 1e-6, "{}", m.energy);
            assert!((m.coupling / v - 1.0).abs() < 1e-3);
            let from_path = PeakedCrossing::from_adiabatic(&arc, &lower, &upper, 2.0).unwrap();
            let straight = crossing(f1, f2, v, 2.0);
            assert!(
                (from_path.a2 / straight.a2 - 1.0).abs() < 1e-2,
                "{f1} {from_path:?}"
            );
            assert!(
                (from_path.b2_per_hartree / straight.b2_per_hartree - 1.0).abs() < 1e-3,
                "{f1} {from_path:?}"
            );
            assert!((from_path.asymmetry - straight.asymmetry).abs() < 1e-2);
            assert!((m.slopes[0] / f1 - 1.0).abs() < 1e-2, "{:?}", m.slopes);
            assert!((m.slopes[1] / f2 - 1.0).abs() < 1e-2, "{:?}", m.slopes);
        }
        // A sloped crossing has no model of this type.
        let lower: Vec<f64> = arc
            .iter()
            .map(|&x| -0.02 * x - (1e-4 * x * x + 4e-6).sqrt())
            .collect();
        let upper: Vec<f64> = arc
            .iter()
            .map(|&x| -0.02 * x + (1e-4 * x * x + 4e-6).sqrt())
            .collect();
        assert!(peaked_model(&arc, &lower, &upper).is_none());
        assert!(PeakedCrossing::from_diabatic(1e-3, 0.02, 1e-4, 1.0).is_none());
    }
}
