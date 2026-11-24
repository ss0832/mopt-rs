//! Settings of an intrinsic reaction coordinate calculation.
//!
//! MultiOptPy: `irc.py:IRC.__init__` (`-irc STEP_SIZE MAX_STEP METHOD`, the force
//! thresholds) and the constants of the integrators in `IRC/`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use crate::error::{MoptError, MoptResult};

/// MultiOptPy's largest gradient component at which a path has arrived (Hartree/Bohr).
pub const DEFAULT_MAX_FORCE: f64 = 0.0004;
/// MultiOptPy's RMS gradient at which a path has arrived (Hartree/Bohr).
pub const DEFAULT_RMS_FORCE: f64 = 0.0001;
/// MultiOptPy tests the arrival only after this many points.
pub const DEFAULT_MIN_POINTS: usize = 10;
/// MultiOptPy ends a path after this many reversals of the energy change in a row.
pub const DEFAULT_OSCILLATION_LIMIT: usize = 5;
/// Largest gradient component at which the final approach of a side ends: the tight
/// threshold of the optimizations (`optmain -tcc`), so that an optimization of the end
/// point has nothing left to do.
pub const DEFAULT_APPROACH_MAX_FORCE: f64 = 1.5e-5;
/// RMS gradient at which the final approach ends (`optmain -tcc`).
pub const DEFAULT_APPROACH_RMS_FORCE: f64 = 1.0e-5;
/// Most steps of the final approach of a side.
pub const DEFAULT_APPROACH_STEPS: usize = 100;
/// MultiOptPy leaves the transition state by this share of the step size.
pub const DEFAULT_INITIAL_SHARE: f64 = 0.1;
/// Eigenvalues of the mass-weighted Hessian below minus this count as imaginary modes
/// (MultiOptPy's `-1e-8`).
pub const DEFAULT_IMAGINARY_THRESHOLD: f64 = crate::opt::mode_kill::IMAGINARY_CURVATURE;
/// Largest angle (degrees) between the downhill direction at the first point and the
/// transition vector that is accepted without the hypersphere correction (measured, see
/// `docs/source/irc.md`).
pub const DEFAULT_FIRST_STEP_ANGLE: f64 = 10.0;
/// Smallest accepted ratio of the actual to the predicted energy change of the first step
/// (measured, see `docs/source/irc.md`).
pub const DEFAULT_RHO_MIN: f64 = 0.75;
/// Largest accepted ratio of the actual to the predicted energy change of the first step.
pub const DEFAULT_RHO_MAX: f64 = 1.25;
/// Evaluations at most in one minimization on a hypersphere.
pub const DEFAULT_SPHERE_ITERATIONS: usize = 30;
/// A minimization on a hypersphere ends when the RMS of the gradient within the sphere
/// (mass-weighted) is below this (measured, see `docs/source/irc.md`).
pub const DEFAULT_SPHERE_GRADIENT: f64 = 1e-4;
/// ... or when the next displacement is shorter than this share of the radius.
pub const DEFAULT_SPHERE_STEP_SHARE: f64 = 1e-3;

/// Which sides of the transition state are followed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Sides {
    /// Both (MultiOptPy).
    #[default]
    Both,
    /// Along the transition vector only.
    Forward,
    /// Against it only.
    Backward,
}

impl Sides {
    /// `both`, `forward`, or `backward`.
    pub fn parse(text: &str) -> MoptResult<Self> {
        match text.to_ascii_lowercase().as_str() {
            "both" => Ok(Self::Both),
            "forward" => Ok(Self::Forward),
            "backward" => Ok(Self::Backward),
            other => Err(MoptError::parse(format!(
                "the sides of an IRC are both, forward, or backward; got '{other}'"
            ))),
        }
    }
}

/// The masses of the atoms of a structure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Masses {
    /// Standard atomic weights, as in the vibrational analysis of mopt-rs.
    #[default]
    Standard,
    /// The masses of the most abundant isotopes, as in MultiOptPy's IRC.
    MostAbundantIsotope,
}

/// End of a minimization on a hypersphere (GS2 and the correction of the first step).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SphereSettings {
    /// Evaluations at most.
    pub max_iterations: usize,
    /// RMS of the mass-weighted gradient within the sphere at which the minimization ends.
    pub gradient_tolerance: f64,
    /// It also ends when the next displacement is shorter than this share of the radius.
    pub step_share: f64,
}

impl Default for SphereSettings {
    fn default() -> Self {
        Self {
            max_iterations: DEFAULT_SPHERE_ITERATIONS,
            gradient_tolerance: DEFAULT_SPHERE_GRADIENT,
            step_share: DEFAULT_SPHERE_STEP_SHARE,
        }
    }
}

/// The test of the first point away from the transition state, and its correction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FirstStepSettings {
    /// Whether the first point is tested and, outside the limits, replaced (the default;
    /// off: MultiOptPy's behaviour).
    pub check: bool,
    /// Largest accepted angle (degrees) between the downhill direction at the first point
    /// and the transition vector.
    pub max_angle_deg: f64,
    /// Smallest accepted ratio of the actual to the predicted energy change.
    pub rho_min: f64,
    /// Largest accepted ratio.
    pub rho_max: f64,
    /// The minimization on the hypersphere around the transition state that replaces a
    /// first point outside these limits.
    pub sphere: SphereSettings,
}

impl Default for FirstStepSettings {
    fn default() -> Self {
        Self {
            check: true,
            max_angle_deg: DEFAULT_FIRST_STEP_ANGLE,
            rho_min: DEFAULT_RHO_MIN,
            rho_max: DEFAULT_RHO_MAX,
            sphere: SphereSettings::default(),
        }
    }
}

/// Settings of an IRC calculation.
#[derive(Debug, Clone, PartialEq)]
pub struct IrcSettings {
    /// Step size in mass-weighted coordinates (amu^(1/2) Bohr; the unit of the variables for
    /// plain variables).
    pub step_size: f64,
    /// MultiOptPy's `MAX_STEP` (the integrators count as MultiOptPy's do: LQA, HPC, DVV, and
    /// RK4 evaluate `MAX_STEP - 1` points of a side, Euler and GS2 `MAX_STEP`, ModeKill at
    /// most `MAX_STEP` cycles).
    pub max_steps: usize,
    /// Name of the integrator (`lqa`, `hpc`, `dvv`, `euler`, `rk4`, `gs2`, `modekill`).
    pub method: String,
    /// Options of the integrator (the parts after `:` in `METHOD`), e.g. the modes of
    /// `modekill:1,2` or `v0=0.04` of `dvv`.
    pub method_options: Vec<String>,
    /// Ask for the exact Hessian at every this many points (`-fc`); `None`: at the
    /// transition state only, updated along the path.
    pub exact_hessian_interval: Option<usize>,
    /// Largest gradient component at which a side has arrived.
    pub max_force: f64,
    /// RMS gradient at which a side has arrived.
    pub rms_force: f64,
    /// Largest gradient component and RMS gradient below which the starting structure is
    /// a stationary point (a transition state when it has one imaginary mode); `None`:
    /// `max_force` and `rms_force`. `optmain -irc` sets them to the thresholds of the
    /// optimization that converged where those are wider.
    pub start_forces: Option<(f64, f64)>,
    /// The arrival is tested after this many points only.
    pub min_points: usize,
    /// A side ends after this many reversals of the energy change in a row.
    pub oscillation_limit: usize,
    /// End a side at the first point whose energy is above that of the point before it
    /// (off: MultiOptPy's behaviour, which ends on the oscillation count only).
    pub stop_on_energy_rise: bool,
    /// Take a step again, shorter, when it led to a point of higher energy (LQA and
    /// Euler; MultiOptPy keeps such points). With exact Hessians along the path (`-fc`)
    /// the Hessian of the point the step started from is asked for first.
    pub retry_on_energy_rise: bool,
    /// Once the gradient is below `max_force` and `rms_force`, go on with steps to the
    /// minimum of the quadratic model until the gradient is below `approach_max_force`
    /// and `approach_rms_force`, so that a side ends in its minimum (LQA, Euler, RK4;
    /// MultiOptPy ends at the first point below `max_force` and `rms_force`).
    pub final_approach: bool,
    /// Largest gradient component at which the final approach ends.
    pub approach_max_force: f64,
    /// RMS gradient at which the final approach ends.
    pub approach_rms_force: f64,
    /// Most steps of the final approach.
    pub approach_steps: usize,
    /// Take a step again, shorter, when the gradient at its end points back along it:
    /// the step went past the floor of the valley, and the next one would turn back
    /// (LQA and Euler, with `retry_on_energy_rise`).
    pub retry_on_overshoot: bool,
    /// Share of the energy a step gains that it may lose by cutting the bend of the
    /// valley; the step length is limited accordingly from the curvature of the path and
    /// the Hessian across it (`None`, the default: no such limit; it pays with exact
    /// Hessians along the path, see `docs/source/irc.md`).
    pub bend_share: Option<f64>,
    /// Mass-weighted length of the displacement from the transition state to the first
    /// point; `None`: [`DEFAULT_INITIAL_SHARE`] of the step size.
    pub initial_displacement: Option<f64>,
    /// Eigenvalues below minus this are imaginary modes.
    pub imaginary_threshold: f64,
    /// Sides followed from a transition state.
    pub sides: Sides,
    /// The masses of the atoms of a structure (the commands; a state machine is given
    /// its masses with its [`crate::irc::IrcSpace`]).
    pub masses: Masses,
    /// Test and correction of the first point.
    pub first_step: FirstStepSettings,
    /// The minimizations on hyperspheres of GS2.
    pub sphere: SphereSettings,
}

impl Default for IrcSettings {
    fn default() -> Self {
        Self {
            step_size: 0.1,
            max_steps: 300,
            method: "lqa".to_string(),
            method_options: Vec::new(),
            exact_hessian_interval: None,
            max_force: DEFAULT_MAX_FORCE,
            rms_force: DEFAULT_RMS_FORCE,
            start_forces: None,
            min_points: DEFAULT_MIN_POINTS,
            oscillation_limit: DEFAULT_OSCILLATION_LIMIT,
            stop_on_energy_rise: false,
            retry_on_energy_rise: true,
            final_approach: true,
            approach_max_force: DEFAULT_APPROACH_MAX_FORCE,
            approach_rms_force: DEFAULT_APPROACH_RMS_FORCE,
            approach_steps: DEFAULT_APPROACH_STEPS,
            retry_on_overshoot: true,
            bend_share: None,
            initial_displacement: None,
            imaginary_threshold: DEFAULT_IMAGINARY_THRESHOLD,
            sides: Sides::Both,
            masses: Masses::Standard,
            first_step: FirstStepSettings::default(),
            sphere: SphereSettings::default(),
        }
    }
}

fn number(key: &str, value: &str) -> MoptResult<f64> {
    value
        .parse::<f64>()
        .ok()
        .filter(|v| v.is_finite())
        .ok_or_else(|| MoptError::parse(format!("IRC option {key}: '{value}' is not a number")))
}

fn count(key: &str, value: &str) -> MoptResult<usize> {
    value
        .parse::<usize>()
        .map_err(|_| MoptError::parse(format!("IRC option {key}: '{value}' is not a whole number")))
}

fn switch(key: &str, value: &str) -> MoptResult<bool> {
    match value.to_ascii_lowercase().as_str() {
        "true" | "yes" | "on" | "1" | "check" => Ok(true),
        "false" | "no" | "off" | "0" | "none" => Ok(false),
        _ => Err(MoptError::parse(format!(
            "IRC option {key}: '{value}' is not on or off"
        ))),
    }
}

impl IrcSettings {
    /// The three values of MultiOptPy's `-irc STEP_SIZE MAX_STEP METHOD` (`METHOD` may carry
    /// options after colons, as in `modekill:1,2`).
    pub fn from_multioptpy(step_size: &str, max_steps: &str, method: &str) -> MoptResult<Self> {
        let mut parts = method.split(':');
        let name = parts.next().unwrap_or_default().to_ascii_lowercase();
        let settings = Self {
            step_size: number("STEP_SIZE", step_size)?,
            max_steps: count("MAX_STEP", max_steps)?,
            method: name,
            method_options: parts.map(str::to_string).collect(),
            ..Self::default()
        };
        settings.validated()
    }

    /// The same settings after a check of their ranges.
    pub fn validated(self) -> MoptResult<Self> {
        if !(self.step_size.is_finite() && self.step_size > 0.0) {
            return Err(MoptError::config("the IRC step size must be positive"));
        }
        if self.max_steps == 0 {
            return Err(MoptError::config(
                "the number of IRC steps must be positive",
            ));
        }
        if self.exact_hessian_interval == Some(0) {
            return Err(MoptError::config(
                "the interval of exact Hessians must be positive",
            ));
        }
        if let Some(s) = self.initial_displacement {
            if !(s.is_finite() && s > 0.0) {
                return Err(MoptError::config(
                    "the first displacement of an IRC must be positive",
                ));
            }
        }
        let f = &self.first_step;
        if !(f.max_angle_deg > 0.0 && f.rho_min < f.rho_max) {
            return Err(MoptError::config(
                "the limits of the first-step test must be an angle above zero and rho_min < rho_max",
            ));
        }
        for sphere in [&self.sphere, &f.sphere] {
            let positive = sphere.gradient_tolerance > 0.0 && sphere.step_share > 0.0;
            if sphere.max_iterations == 0 || !positive {
                return Err(MoptError::config(
                    "the settings of the hypersphere minimization must be positive",
                ));
            }
        }
        Ok(self)
    }

    /// The start is at least as stationary as an optimization with the thresholds
    /// `max_force` and `rms_force` requires: a structure such an optimization converged to
    /// passes the test of the starting structure.
    pub fn accepting_converged(mut self, max_force: f64, rms_force: f64) -> Self {
        let (max, rms) = self
            .start_forces
            .unwrap_or((self.max_force, self.rms_force));
        // The optimizers accept a gradient at their thresholds; the test here is strict.
        self.start_forces = Some((
            max.max(max_force * (1.0 + 1e-9)),
            rms.max(rms_force * (1.0 + 1e-9)),
        ));
        self
    }

    /// Mass-weighted length of the displacement from the transition state.
    pub fn first_displacement(&self) -> f64 {
        self.initial_displacement
            .unwrap_or(DEFAULT_INITIAL_SHARE * self.step_size)
    }

    /// Whether the exact Hessian is asked for at point `step` of a side.
    pub fn exact_hessian_at(&self, step: usize) -> bool {
        self.exact_hessian_interval
            .is_some_and(|n| n > 0 && step.is_multiple_of(n))
    }

    /// Apply `KEY=VALUE` options (`-irc_opt`). Keys: `first_step` (`check` / `off`), `angle`
    /// (degrees), `rho_min`, `rho_max`, `initial` (length of the first displacement),
    /// `sides` (`both`, `forward`, `backward`), `masses` (`standard`, `isotope`), `max_force`, `rms_force`, `min_points`,
    /// `oscillations`, `energy_rise` (`retry` / `stop` / `off`), `overshoot` (`retry` /
    /// `keep`), `bend` (a share or `off`), `approach` and its thresholds, `sphere_iterations`, `sphere_gradient`, `sphere_step` (the last three
    /// for GS2 and for the correction of the first step alike).
    pub fn with_options(mut self, tokens: &[String]) -> MoptResult<Self> {
        for token in tokens {
            let (key, value) = token.split_once('=').ok_or_else(|| {
                MoptError::parse(format!("-irc_opt expects KEY=VALUE, got '{token}'"))
            })?;
            match key.to_ascii_lowercase().as_str() {
                "first_step" => self.first_step.check = switch(key, value)?,
                "angle" => self.first_step.max_angle_deg = number(key, value)?,
                "rho_min" => self.first_step.rho_min = number(key, value)?,
                "rho_max" => self.first_step.rho_max = number(key, value)?,
                "initial" => self.initial_displacement = Some(number(key, value)?),
                "sides" => self.sides = Sides::parse(value)?,
                "masses" => {
                    self.masses = match value.to_ascii_lowercase().as_str() {
                        "standard" => Masses::Standard,
                        "isotope" => Masses::MostAbundantIsotope,
                        other => {
                            return Err(MoptError::parse(format!(
                                "IRC option masses is standard or isotope, got '{other}'"
                            )))
                        }
                    }
                }
                "max_force" => self.max_force = number(key, value)?,
                "rms_force" => self.rms_force = number(key, value)?,
                "min_points" => self.min_points = count(key, value)?,
                "oscillations" => self.oscillation_limit = count(key, value)?,
                "energy_rise" => {
                    (self.retry_on_energy_rise, self.stop_on_energy_rise) =
                        match value.to_ascii_lowercase().as_str() {
                            "retry" => (true, false),
                            "stop" => (false, true),
                            "off" | "keep" => (false, false),
                            other => {
                                return Err(MoptError::parse(format!(
                                    "IRC option energy_rise is retry, stop, or off, got '{other}'"
                                )))
                            }
                        }
                }
                "approach" => self.final_approach = switch(key, value)?,
                "approach_max_force" => self.approach_max_force = number(key, value)?,
                "approach_rms_force" => self.approach_rms_force = number(key, value)?,
                "approach_steps" => self.approach_steps = count(key, value)?,
                "bend" => {
                    self.bend_share = match value.to_ascii_lowercase().as_str() {
                        "off" | "none" => None,
                        _ => Some(number(key, value)?),
                    }
                }
                "overshoot" => {
                    self.retry_on_overshoot = match value.to_ascii_lowercase().as_str() {
                        "retry" => true,
                        "keep" | "off" => false,
                        other => {
                            return Err(MoptError::parse(format!(
                                "IRC option overshoot is retry or keep, got '{other}'"
                            )))
                        }
                    }
                }
                "sphere_iterations" => {
                    let n = count(key, value)?;
                    self.sphere.max_iterations = n;
                    self.first_step.sphere.max_iterations = n;
                }
                "sphere_gradient" => {
                    let g = number(key, value)?;
                    self.sphere.gradient_tolerance = g;
                    self.first_step.sphere.gradient_tolerance = g;
                }
                "sphere_step" => {
                    let s = number(key, value)?;
                    self.sphere.step_share = s;
                    self.first_step.sphere.step_share = s;
                }
                _ => {
                    return Err(MoptError::parse(format!(
                        "-irc_opt: unknown key '{key}' (first_step, angle, rho_min, rho_max, \
                         initial, sides, masses, max_force, rms_force, min_points, oscillations, \
                         energy_rise, overshoot, bend, approach, approach_max_force, \
                         approach_rms_force, \
                         approach_steps, sphere_iterations, \
                         sphere_gradient, sphere_step)"
                    )))
                }
            }
        }
        self.validated()
    }
}

/// The `KEY=VALUE` options of an integrator (`dvv:v0=0.04:dt0=0.5`, also separated by
/// commas); bare values are returned in `bare`.
pub(crate) struct MethodOptions {
    pub pairs: Vec<(String, String)>,
    pub bare: Vec<String>,
}

impl MethodOptions {
    pub(crate) fn parse(options: &[String]) -> Self {
        let mut pairs = Vec::new();
        let mut bare = Vec::new();
        for part in options.iter().flat_map(|o| o.split(',')) {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }
            match part.split_once('=') {
                Some((k, v)) => pairs.push((k.trim().to_ascii_lowercase(), v.trim().to_string())),
                None => bare.push(part.to_string()),
            }
        }
        Self { pairs, bare }
    }

    /// The number given for `key`, or `default`.
    pub(crate) fn number(&self, method: &str, key: &str, default: f64) -> MoptResult<f64> {
        match self.pairs.iter().find(|(k, _)| k == key) {
            Some((_, v)) => number(&format!("{method}:{key}"), v),
            None => Ok(default),
        }
    }

    /// The switch given for `key`, or `default`.
    pub(crate) fn switch(&self, method: &str, key: &str, default: bool) -> MoptResult<bool> {
        match self.pairs.iter().find(|(k, _)| k == key) {
            Some((_, v)) => switch(&format!("{method}:{key}"), v),
            None => Ok(default),
        }
    }

    /// Fail on keys other than `known` and, unless `bare_allowed`, on bare values.
    pub(crate) fn only(&self, method: &str, known: &[&str], bare_allowed: bool) -> MoptResult<()> {
        if let Some((k, _)) = self
            .pairs
            .iter()
            .find(|(k, _)| !known.contains(&k.as_str()))
        {
            return Err(MoptError::parse(format!(
                "IRC method {method}: unknown option '{k}' (options: {})",
                if known.is_empty() {
                    "none".to_string()
                } else {
                    known.join(", ")
                }
            )));
        }
        if !bare_allowed {
            if let Some(b) = self.bare.first() {
                return Err(MoptError::parse(format!(
                    "IRC method {method}: option '{b}' is not KEY=VALUE"
                )));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn multioptpy_triple_and_options() {
        let s = IrcSettings::from_multioptpy("0.5", "300", "ModeKill:1,2").unwrap();
        assert_eq!(s.method, "modekill");
        assert_eq!(s.method_options, vec!["1,2".to_string()]);
        assert_eq!(s.max_steps, 300);
        assert!((s.first_displacement() - 0.05).abs() < 1e-15);
        assert!(!s.exact_hessian_at(3));
        let s = s
            .with_options(&[
                "first_step=check".to_string(),
                "angle=5".to_string(),
                "sides=forward".to_string(),
                "initial=0.2".to_string(),
            ])
            .unwrap();
        assert!(s.first_step.check);
        assert_eq!(s.sides, Sides::Forward);
        assert!((s.first_displacement() - 0.2).abs() < 1e-15);
        assert!(IrcSettings::from_multioptpy("0", "300", "lqa").is_err());
        assert!(IrcSettings::default()
            .with_options(&["nothing=1".to_string()])
            .is_err());
        let o = MethodOptions::parse(&["v0=0.05,dt0=1".to_string(), "7".to_string()]);
        assert_eq!(o.number("dvv", "v0", 0.04).unwrap(), 0.05);
        assert_eq!(o.bare, vec!["7".to_string()]);
        assert!(o.only("dvv", &["v0"], true).is_err());
    }
}
