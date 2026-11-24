//! Resolution of `-opt` method strings into step generators, Hessian updaters, and enhancers.
//!
//! Reproduces MultiOptPy's `CalculateMoveVector.initialization`:
//! 1. exact special names (`ranger`, `adam`, ...),
//! 2. quasi-Newton families `<prefix>rsirfo_<updater>` / `rsprfo_<updater>` (the updater part is
//!    required; e.g. plain `rsirfo` is not a quasi-Newton key),
//! 3. first-order optimizers by substring (`fire`, `lbfgs`, ...),
//! 4. conjugate gradients,
//! 5. FIRE as the fallback for unknown names (reported as a warning).
//!
//! Enhancers are enabled by substrings of the same method string (`lars`, `gdiis`, ...).
//! Everything is table-driven; new methods are added with the `register_*` functions.
//!
//! MultiOptPy: `optimizer.py:CalculateMoveVector.initialization`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::sync::Arc;

use super::enhance::adiis::Adiis;
use super::enhance::c2diis::C2Diis;
use super::enhance::component_wise_scaling::ComponentWiseScaling;
use super::enhance::coordinate_locking::CoordinateLocking;
use super::enhance::ediis::Ediis;
use super::enhance::gan_step::GanStep;
use super::enhance::gdiis::Gdiis;
use super::enhance::gediis::Gediis;
use super::enhance::geodesic_step::GeodesicStep;
use super::enhance::gpr_step::GprStep;
use super::enhance::kdiis::Kdiis;
use super::enhance::lars::Lars;
use super::enhance::linesearch::LineSearch;
use super::enhance::lookahead::Lookahead;
use super::enhance::perturbation::Perturbation;
use super::enhance::rl_step::RlStep;
use super::enhance::trim::Trim;
use super::enhance::StepEnhancer;
use super::step::abc_fire::AbcFire;
use super::step::adabelief::AdaBelief;
use super::step::adam::Adam;
use super::step::ccqn::{Ccqn, CcqnSettings, SaddleStep};
use super::step::conjugate_gradient::{CgParams, CgVariant, ConjugateGradient};
use super::step::eve::Eve;
use super::step::fire::Fire;
use super::step::fire2::Fire2;
use super::step::gpmin::GpMin;
use super::step::gradient_descent::{GradientDescent, MassWeightedGradientDescent};
use super::step::hisd::{ReflectedDirections, ReflectedGradient};
use super::step::lbfgs::Lbfgs;
use super::step::mf_rsirfo::MfRsirfo;
use super::step::mode_following::ModeFollowingSettings;
use super::step::newton::Newton;
use super::step::prodigy::Prodigy;
use super::step::radam::Radam;
use super::step::rsirfo::{Rsirfo, RsirfoTrust};
use super::step::rsprfo::{Rsprfo, RsprfoParams};
use super::step::tr_lbfgs::TrLbfgs;
use super::step::StepGenerator;
use super::trust::{OuterTrustRadius, TrustSettings};
use super::update::{HessianUpdater, UpdaterRegistry};
use crate::error::{MoptError, MoptResult};

/// The method of a minimization when none is named (no `-opt`): the restricted-step
/// rational function method with the block FSB update.
pub const DEFAULT_MINIMIZER: &str = "rsirfo_block_fsb";
/// The method of a saddle-point search (`-order 1` and above) when none is named: the
/// same step with the block Bofill update, which does not keep the Hessian positive
/// definite.
pub const DEFAULT_SADDLE_METHOD: &str = "rsirfo_block_bofill";

/// The methods used when none is named, by the order of the stationary point that is
/// looked for. MultiOptPy's default is `FIRELARS` for either, which can be named.
pub fn default_methods(saddle_order: usize) -> Vec<String> {
    let method = if saddle_order == 0 {
        DEFAULT_MINIMIZER
    } else {
        DEFAULT_SADDLE_METHOD
    };
    vec![method.to_string()]
}

/// Options available to factories.
#[derive(Debug, Clone)]
pub struct MethodContext {
    /// Saddle order of the search.
    pub saddle_order: usize,
    /// Trust-radius settings.
    pub trust: TrustSettings,
    /// CCQN settings (reaction-coordinate axis and parameters), required by `ccqn*` methods.
    pub ccqn: Option<Arc<CcqnSettings>>,
}

/// Factory of a step generator from the full method string.
pub type GeneratorFactory =
    Arc<dyn Fn(&str, &MethodContext) -> MoptResult<Box<dyn StepGenerator>> + Send + Sync>;

/// Factory of an enhancer.
pub type EnhancerFactory =
    Arc<dyn Fn(&MethodContext) -> MoptResult<Box<dyn StepEnhancer>> + Send + Sync>;

/// How a generator entry matches a method string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MethodMatch {
    /// Lower-cased method string equals the key.
    Exact(String),
    /// Method string contains `family + updater_key` for a registered updater key.
    QuasiNewton(String),
    /// Method string contains the key.
    Contains(String),
}

struct GeneratorEntry {
    matcher: MethodMatch,
    uses_hessian: bool,
    factory: GeneratorFactory,
    /// Hessian updater used when the method string names none.
    default_updater: Option<&'static str>,
    /// Enhancers switched on by the name itself (MultiOptPy's `specific_cases`).
    forced_enhancers: &'static [&'static str],
}

/// Factory of a generator that needs neither the method string nor the context.
fn simple<G: StepGenerator + Default + 'static>() -> GeneratorFactory {
    Arc::new(|_: &str, _: &MethodContext| Ok(Box::new(G::default()) as Box<dyn StepGenerator>))
}

/// Factory of an enhancer that needs no context.
fn simple_enhancer<E: StepEnhancer + Default + 'static>() -> EnhancerFactory {
    Arc::new(|_: &MethodContext| Ok(Box::new(E::default()) as Box<dyn StepEnhancer>))
}

struct EnhancerEntry {
    key: String,
    excludes: Vec<String>,
    hessian_only: bool,
    factory: EnhancerFactory,
}

/// Resolved method.
pub struct ResolvedMethod {
    /// The method string as given.
    pub method: String,
    /// Base step generator.
    pub generator: Box<dyn StepGenerator>,
    /// Enhancers in application order.
    pub enhancers: Vec<Box<dyn StepEnhancer>>,
    /// Hessian updater (for Hessian-based generators).
    pub updater: Option<Box<dyn HessianUpdater>>,
    /// Non-fatal notes (e.g. fallback to FIRE).
    pub warnings: Vec<String>,
}

impl ResolvedMethod {
    /// Continue at the requested index after a mode-kill displacement.
    pub(super) fn mode_kill_hisd(retained: &[faer::Col<f64>]) -> MoptResult<Self> {
        let mut generator = ReflectedGradient::for_mode_kill(retained.len());
        let updater = if retained.is_empty() {
            None
        } else {
            if !generator.seed_followed_modes(retained) {
                return Err(MoptError::config(
                    "HiSD mode kill has no independent direction to retain",
                ));
            }
            Some(UpdaterRegistry::default().resolve("block_bofill")?)
        };
        Ok(Self {
            method: "hisd".into(),
            generator: Box::new(generator),
            enhancers: Vec::new(),
            updater,
            warnings: Vec::new(),
        })
    }
}

/// Quasi-Newton updater keys in MultiOptPy's mapping order.
const QN_UPDATER_KEYS: [&str; 25] = [
    "bfgs_dd",
    "bfgs",
    "block_bfgs_dd",
    "block_bfgs",
    "fsb_dd",
    "fsb",
    "block_fsb_dd",
    "block_fsb_weighted",
    "block_fsb",
    "block_cfd_fsb_dd",
    "block_cfd_fsb_weighted",
    "block_cfd_fsb",
    "cfd_fsb_dd",
    "cfd_fsb",
    "bofill",
    "block_bofill_weighted",
    "block_bofill",
    "block_cfd_bofill_weighted",
    "block_cfd_bofill",
    "cfd_bofill",
    "pcfd_bofill",
    "msp",
    "sr1",
    "psb",
    "flowchart",
];

/// Table of step generators, enhancers, and updaters.
pub struct MethodRegistry {
    generators: Vec<GeneratorEntry>,
    enhancers: Vec<EnhancerEntry>,
    /// Hessian updaters.
    pub updaters: UpdaterRegistry,
}

impl Default for MethodRegistry {
    fn default() -> Self {
        let mut reg = Self {
            generators: Vec::new(),
            enhancers: Vec::new(),
            updaters: UpdaterRegistry::default(),
        };
        reg.install_defaults();
        reg
    }
}

impl MethodRegistry {
    fn push_generator(
        &mut self,
        matcher: MethodMatch,
        uses_hessian: bool,
        factory: GeneratorFactory,
    ) {
        self.generators.push(GeneratorEntry {
            matcher,
            uses_hessian,
            factory,
            default_updater: None,
            forced_enhancers: &[],
        });
    }

    fn push_enhancer(
        &mut self,
        key: &str,
        excludes: &[&str],
        hessian_only: bool,
        factory: EnhancerFactory,
    ) {
        self.enhancers.push(EnhancerEntry {
            key: key.to_string(),
            excludes: excludes.iter().map(|s| s.to_string()).collect(),
            hessian_only,
            factory,
        });
    }

    fn install_defaults(&mut self) {
        use MethodMatch::{Contains, Exact, QuasiNewton};
        // CCQN (Wu and Wang 2025): `ccqn` uses RS-P-RFO in the saddle region, `ccqn_rsirfo`
        // RS-I-RFO; TS-BFGS unless another updater is named (e.g. `ccqn_block_bofill`). Listed
        // first so that `ccqn_rsirfo_<updater>` is not taken for an RS-I-RFO method.
        self.generators.push(GeneratorEntry {
            matcher: Contains("ccqn".into()),
            uses_hessian: true,
            factory: Arc::new(|m: &str, ctx: &MethodContext| {
                let mut settings = ctx.ccqn.as_deref().cloned().ok_or_else(|| {
                    MoptError::config(format!(
                        "'{m}' needs a CCQN reaction-coordinate axis (reactive bonds, a \
                             product structure, or a direction)"
                    ))
                })?;
                settings.saddle_step = if m.to_ascii_lowercase().contains("rsirfo") {
                    SaddleStep::RsIrfo
                } else {
                    SaddleStep::RsPrfo
                };
                // -tr / -mintr bound the saddle-phase radius.
                if let Some(max) = ctx.trust.max {
                    settings.trust.max = max.value();
                    settings.trust.initial = settings.trust.initial.min(max.value());
                    settings.trust.min = settings.trust.min.min(settings.trust.initial);
                }
                let rsirfo = RsirfoTrust::for_saddle_order(
                    ctx.saddle_order.max(1),
                    ctx.trust.max.map(|q| q.value()),
                    Some(ctx.trust.min.value()),
                );
                Ok(Box::new(Ccqn::new(settings, rsirfo)?) as Box<dyn StepGenerator>)
            }),
            default_updater: Some("ts_bfgs"),
            forced_enhancers: &[],
        });
        // High-index saddle dynamics with the optimizer's Hessian (`-order k`; mopt-rs, not
        // in MultiOptPy): `hisd` follows eigenvectors, `ggad` the deflated flows of Quapp and
        // Bofill. The block Bofill update unless another is named (`hisd_bofill`).
        for (key, directions) in [
            ("hisd", ReflectedDirections::Hisd),
            ("ggad", ReflectedDirections::Ggad),
        ] {
            self.generators.push(GeneratorEntry {
                matcher: Contains(key.into()),
                uses_hessian: true,
                factory: Arc::new(move |_: &str, _: &MethodContext| {
                    Ok(Box::new(ReflectedGradient::new(directions)) as Box<dyn StepGenerator>)
                }),
                default_updater: Some("block_bofill"),
                forced_enhancers: &[],
            });
        }
        // MultiOptPy's `specific_cases`: exact names; `ranger` and `adam` imply Lookahead.
        // (LARS and Lookahead named in the string are found by the enhancer keys.)
        for (name, factory, forced) in [
            ("ranger", simple::<Radam>(), &["lookahead"][..]),
            ("rangerlars", simple::<Radam>(), &["lookahead"]),
            ("adam", simple::<Adam>(), &["lookahead"]),
            ("adamlars", simple::<Adam>(), &[]),
            ("adamlookahead", simple::<Adam>(), &[]),
            ("adamlookaheadlars", simple::<Adam>(), &[]),
        ] {
            self.generators.push(GeneratorEntry {
                matcher: Exact(name.into()),
                uses_hessian: false,
                factory,
                default_updater: None,
                forced_enhancers: forced,
            });
        }
        // Quasi-Newton families in MultiOptPy's mapping order.
        // Mode-following RS-I-RFO: `mw` = mass-weighted overlaps, `s` = static references;
        // the options after `:` are read from the method string.
        for fam in ["mwsmf_rsirfo_", "mwmf_rsirfo_", "smf_rsirfo_", "mf_rsirfo_"] {
            self.push_generator(
                QuasiNewton(fam.into()),
                true,
                Arc::new(|m: &str, ctx: &MethodContext| {
                    let trust = RsirfoTrust::for_saddle_order(
                        ctx.saddle_order,
                        ctx.trust.max.map(|q| q.value()),
                        Some(ctx.trust.min.value()),
                    );
                    let settings = ModeFollowingSettings::from_method(m)?;
                    Ok(Box::new(MfRsirfo::new(trust, ctx.saddle_order, settings))
                        as Box<dyn StepGenerator>)
                }),
            );
        }
        self.push_generator(
            QuasiNewton("crsirfo_".into()),
            true,
            // Constraints are applied by the state machine for every method, so constrained
            // RS-I-RFO is RS-I-RFO.
            Arc::new(|_m: &str, ctx: &MethodContext| {
                Ok(Box::new(Rsirfo::new(RsirfoTrust::for_saddle_order(
                    ctx.saddle_order,
                    ctx.trust.max.map(|q| q.value()),
                    Some(ctx.trust.min.value()),
                ))) as Box<dyn StepGenerator>)
            }),
        );
        self.push_generator(
            QuasiNewton("rsirfo_".into()),
            true,
            Arc::new(|_: &str, ctx: &MethodContext| {
                Ok(Box::new(Rsirfo::new(RsirfoTrust::for_saddle_order(
                    ctx.saddle_order,
                    ctx.trust.max.map(|q| q.value()),
                    Some(ctx.trust.min.value()),
                ))) as Box<dyn StepGenerator>)
            }),
        );
        self.push_generator(
            QuasiNewton("rsprfo_".into()),
            true,
            Arc::new(|_: &str, ctx: &MethodContext| {
                Ok(Box::new(Rsprfo::new(RsprfoParams::for_saddle_order(
                    ctx.saddle_order,
                    ctx.trust.max.map(|q| q.value()),
                    Some(ctx.trust.min.value()),
                ))) as Box<dyn StepGenerator>)
            }),
        );
        // Newton steps (not reachable in MultiOptPy): `mnewton_<updater>` updates its Hessian
        // from exponentially averaged points and gradients.
        let updaters = self.updaters.clone();
        self.push_generator(
            QuasiNewton("mnewton_".into()),
            true,
            Arc::new(move |m: &str, _: &MethodContext| {
                let updater = updaters.resolve(m)?;
                Ok(Box::new(Newton::with_momentum(0.5, 0.5, updater)) as Box<dyn StepGenerator>)
            }),
        );
        self.push_generator(QuasiNewton("newton_".into()), true, simple::<Newton>());
        // First-order methods in the order of MultiOptPy's `optimizer_mapping`.
        for (key, factory) in [
            ("adabelief", simple::<AdaBelief>()),
            ("radam", simple::<Radam>()),
            ("eve", simple::<Eve>()),
            ("prodigy", simple::<Prodigy>()),
            ("abcfire", simple::<AbcFire>()),
            ("fire2", simple::<Fire2>()),
            ("fire", simple::<Fire>()),
            ("mwgradientdescent", simple::<MassWeightedGradientDescent>()),
            ("gradientdescent", simple::<GradientDescent>()),
            ("steepest_descent", simple::<GradientDescent>()),
        ] {
            self.push_generator(Contains(key.into()), false, factory);
        }
        self.push_generator(Contains("gpmin".into()), false, simple::<GpMin>());
        self.push_generator(Contains("tr_lbfgs".into()), false, simple::<TrLbfgs>());
        self.push_generator(Contains("lbfgs".into()), false, simple::<Lbfgs>());
        // Conjugate gradients: the variant is read from the method string (`cg` alone is
        // Polak-Ribiere).
        self.push_generator(
            Contains("cg".into()),
            false,
            Arc::new(|m: &str, _: &MethodContext| {
                let params = CgParams::new(CgVariant::from_method(m));
                Ok(Box::new(ConjugateGradient::new(params)) as Box<dyn StepGenerator>)
            }),
        );

        // Enhancers in MultiOptPy's application order.
        self.push_enhancer("lars", &[], false, simple_enhancer::<Lars>());
        self.push_enhancer("lookahead", &[], false, simple_enhancer::<Lookahead>());
        self.push_enhancer("linesearch", &[], false, simple_enhancer::<LineSearch>());
        // `gediis` contains `ediis`: MultiOptPy then uses GEDIIS only.
        self.push_enhancer("ediis", &["gediis"], false, simple_enhancer::<Ediis>());
        self.push_enhancer("gdiis", &[], false, simple_enhancer::<Gdiis>());
        self.push_enhancer("c2diis", &[], false, simple_enhancer::<C2Diis>());
        self.push_enhancer("adiis", &[], false, simple_enhancer::<Adiis>());
        self.push_enhancer("kdiis", &[], false, simple_enhancer::<Kdiis>());
        self.push_enhancer("gediis", &[], false, simple_enhancer::<Gediis>());
        self.push_enhancer(
            "coordinate_locking",
            &[],
            false,
            simple_enhancer::<CoordinateLocking>(),
        );
        self.push_enhancer(
            "component_wise_scaling",
            &[],
            false,
            simple_enhancer::<ComponentWiseScaling>(),
        );
        self.push_enhancer("gpr_step", &[], false, simple_enhancer::<GprStep>());
        self.push_enhancer("gan_step", &[], false, simple_enhancer::<GanStep>());
        self.push_enhancer("rl_step", &[], false, simple_enhancer::<RlStep>());
        self.push_enhancer(
            "geodesic_step",
            &[],
            false,
            simple_enhancer::<GeodesicStep>(),
        );
        // Not selectable in MultiOptPy (its class is never called): thermal noise at 298.15 K,
        // annealed.
        self.push_enhancer(
            "perturbation",
            &[],
            false,
            simple_enhancer::<Perturbation>(),
        );
        // TRIM applies to Hessian-based methods only; its radius is the optimizer's.
        self.push_enhancer(
            "trim",
            &[],
            true,
            Arc::new(|ctx: &MethodContext| {
                let radius = OuterTrustRadius::new(ctx.trust, ctx.saddle_order, false).radius();
                Ok(Box::new(Trim::new(radius.value())) as Box<dyn StepEnhancer>)
            }),
        );
    }

    /// Register a step generator under an exact (case-insensitive) name, with top priority.
    pub fn register_generator(
        &mut self,
        name: &str,
        uses_hessian: bool,
        factory: GeneratorFactory,
    ) {
        self.generators.insert(
            0,
            GeneratorEntry {
                matcher: MethodMatch::Exact(name.to_ascii_lowercase()),
                uses_hessian,
                factory,
                default_updater: None,
                forced_enhancers: &[],
            },
        );
    }

    /// Register (or replace) an enhancer triggered by `key` appearing in the method string.
    pub fn register_enhancer(&mut self, key: &str, factory: EnhancerFactory) {
        let key = key.to_ascii_lowercase();
        if let Some(e) = self.enhancers.iter_mut().find(|e| e.key == key) {
            e.factory = factory;
        } else {
            self.push_enhancer(&key, &[], false, factory);
        }
    }

    fn matches(&self, matcher: &MethodMatch, lower: &str) -> bool {
        match matcher {
            MethodMatch::Exact(k) => lower == k,
            MethodMatch::Contains(k) => lower.contains(k.as_str()),
            MethodMatch::QuasiNewton(fam) => QN_UPDATER_KEYS
                .iter()
                .any(|u| lower.contains(&format!("{fam}{u}"))),
        }
    }

    /// Resolve one method string.
    pub fn resolve(&self, method: &str, ctx: &MethodContext) -> MoptResult<ResolvedMethod> {
        let lower = method.to_ascii_lowercase();
        let mut warnings = Vec::new();
        let entry = self
            .generators
            .iter()
            .find(|e| self.matches(&e.matcher, &lower));
        let forced = entry.map_or(&[][..], |e| e.forced_enhancers);
        let (generator, uses_hessian, default_updater) = match entry {
            Some(e) => ((e.factory)(method, ctx)?, e.uses_hessian, e.default_updater),
            None => {
                warnings.push(format!(
                    "This method is not implemented: {method}. Thus, Default method (FIRE) is used."
                ));
                (
                    Box::new(Fire::default()) as Box<dyn StepGenerator>,
                    false,
                    None,
                )
            }
        };
        let mut enhancers = Vec::new();
        for e in &self.enhancers {
            let hit = forced.contains(&e.key.as_str())
                || (lower.contains(e.key.as_str())
                    && !e.excludes.iter().any(|x| lower.contains(x.as_str())));
            if hit && (!e.hessian_only || uses_hessian) {
                enhancers.push((e.factory)(ctx)?);
            }
        }
        let updater = match (uses_hessian, default_updater) {
            (false, _) => None,
            (true, Some(d)) => Some(self.updaters.resolve_or(method, d)?),
            (true, None) => Some(self.updaters.resolve(method)?),
        };
        if let Some(u) = updater.as_ref().filter(|u| {
            ctx.saddle_order > 0
                && !u.learns_saddle_curvature()
                && !generator.learns_negative_curvature()
        }) {
            // BFGS, FSB, flowchart and their block forms: the BFGS term keeps the curvature
            // positive.
            warnings.push(format!(
                "The Hessian update '{}' keeps the curvature positive and is meant for \
                 minimizations. For a saddle search use bofill, msp, sr1, psb, or block_bofill.",
                u.name()
            ));
        }
        Ok(ResolvedMethod {
            method: method.to_string(),
            generator,
            enhancers,
            updater,
            warnings,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> MethodContext {
        MethodContext {
            saddle_order: 0,
            trust: TrustSettings::default(),
            ccqn: None,
        }
    }

    #[test]
    fn ccqn_variants_and_updaters() {
        use crate::opt::step::ccqn::CcqnAxis;
        use faer::Col;
        let reg = MethodRegistry::default();
        let mut c = ctx();
        c.saddle_order = 1;
        assert!(reg.resolve("ccqn", &c).is_err(), "an axis is required");
        c.ccqn = Some(Arc::new(CcqnSettings::new(CcqnAxis::Direction(
            Col::from_fn(3, |i| [1.0, 0.0, 0.0][i]),
        ))));
        let name = |m: &str| {
            let r = reg.resolve(m, &c).unwrap();
            (
                r.generator.name().to_string(),
                r.updater.unwrap().name().to_string(),
            )
        };
        assert_eq!(name("ccqn"), ("ccqn".into(), "ts_bfgs".into()));
        assert_eq!(
            name("ccqn_rsirfo"),
            ("ccqn_rsirfo".into(), "ts_bfgs".into())
        );
        assert_eq!(
            name("ccqn_block_bofill"),
            ("ccqn".into(), "block_bofill".into())
        );
        assert_eq!(
            name("ccqn_rsirfo_block_bofill"),
            ("ccqn_rsirfo".into(), "block_bofill".into())
        );
    }

    #[test]
    fn default_firelars_and_quasi_newton() {
        let reg = MethodRegistry::default();
        let m = reg.resolve("FIRELARS", &ctx()).unwrap();
        assert_eq!(m.generator.name(), "fire");
        assert_eq!(m.enhancers.len(), 1);
        assert!(m.updater.is_none());

        let m = reg.resolve("rsirfo_block_fsb", &ctx()).unwrap();
        assert_eq!(m.generator.name(), "rsirfo");
        assert_eq!(m.updater.as_ref().unwrap().name(), "block_fsb");

        let m = reg.resolve("lbfgs", &ctx()).unwrap();
        assert_eq!(m.generator.name(), "lbfgs");
    }

    #[test]
    fn saddle_searches_are_told_when_the_update_is_one_for_minimizations() {
        let reg = MethodRegistry::default();
        let saddle = MethodContext {
            saddle_order: 1,
            ..ctx()
        };
        for method in [
            "rsirfo_fsb",
            "rsirfo_bfgs",
            "rsirfo_block_fsb",
            "rsprfo_block_bfgs",
        ] {
            let m = reg.resolve(method, &saddle).unwrap();
            assert_eq!(m.warnings.len(), 1, "{method}");
            assert!(
                m.warnings[0].contains("meant for minimizations"),
                "{method}"
            );
            assert!(reg.resolve(method, &ctx()).unwrap().warnings.is_empty());
        }
        for method in [
            "rsirfo_bofill",
            "rsirfo_block_bofill",
            "rsirfo_msp",
            "rsprfo_sr1",
        ] {
            assert!(
                reg.resolve(method, &saddle).unwrap().warnings.is_empty(),
                "{method}"
            );
        }
    }

    #[test]
    fn plain_rsirfo_falls_back_to_fire_like_the_reference() {
        let reg = MethodRegistry::default();
        let m = reg.resolve("rsirfo", &ctx()).unwrap();
        assert_eq!(m.generator.name(), "fire");
        assert_eq!(m.warnings.len(), 1);
    }

    #[test]
    fn multioptpy_names_select_the_ported_methods() {
        let reg = MethodRegistry::default();
        let parts = |m: &str| {
            let r = reg.resolve(m, &ctx()).unwrap();
            let enh: Vec<String> = r.enhancers.iter().map(|e| e.name().to_string()).collect();
            (
                r.generator.name().to_string(),
                enh.join("+"),
                r.updater.is_some(),
            )
        };
        let gen = |m: &str| parts(m).0;
        // Exact names of MultiOptPy's `specific_cases`.
        assert_eq!(parts("ranger"), ("radam".into(), "lookahead".into(), false));
        assert_eq!(
            parts("RangerLARS"),
            ("radam".into(), "lars+lookahead".into(), false)
        );
        assert_eq!(parts("adam"), ("adam".into(), "lookahead".into(), false));
        assert_eq!(parts("adamlars"), ("adam".into(), "lars".into(), false));
        assert_eq!(
            parts("adamlookaheadlars"),
            ("adam".into(), "lars+lookahead".into(), false)
        );
        // First-order methods by substring, in MultiOptPy's order of precedence.
        for (m, want) in [
            ("adabelief", "adabelief"),
            ("radam", "radam"),
            ("eve", "eve"),
            ("prodigy", "prodigy"),
            ("abcfire", "abcfire"),
            ("fire2", "fire2"),
            ("FIRE", "fire"),
            ("mwgradientdescent", "mwgradientdescent"),
            ("gradientdescent", "gradientdescent"),
            ("steepest_descent", "gradientdescent"),
            ("gpmin", "gpmin"),
            ("tr_lbfgs", "tr_lbfgs"),
            ("lbfgs", "lbfgs"),
            ("cg", "cg_pr"),
            ("cg_pr", "cg_pr"),
            ("cg_fr", "cg_fr"),
            ("cg_hs", "cg_hs"),
            ("cg_dy", "cg_dy"),
        ] {
            assert_eq!(gen(m), want, "{m}");
        }
        // Quasi-Newton families.
        for (m, want) in [
            ("rsprfo_bofill", "rsprfo"),
            ("crsirfo_bfgs", "rsirfo"),
            ("mf_rsirfo_bofill", "mf_rsirfo"),
            ("smf_rsirfo_fsb", "mf_rsirfo"),
            ("mwmf_rsirfo_block_fsb", "mf_rsirfo"),
            ("mwsmf_rsirfo_msp:1:ema0.5", "mf_rsirfo"),
            ("newton_bfgs", "newton"),
            ("mnewton_fsb", "mnewton"),
        ] {
            let (g, _, updater) = parts(m);
            assert_eq!(g, want, "{m}");
            assert!(updater, "{m}");
        }
        assert!(reg.resolve("mf_rsirfo_bofill:fast", &ctx()).is_err());
        // Enhancers by substring, in MultiOptPy's order of application.
        let all = "rsirfo_bfgs_lars_lookahead_linesearch_ediis_gdiis_c2diis_adiis_kdiis_\
                   coordinate_locking_component_wise_scaling_gpr_step_gan_step_rl_step_\
                   geodesic_step_trim_perturbation";
        assert_eq!(
            parts(all).1,
            "lars+lookahead+linesearch+ediis+gdiis+c2diis+adiis+kdiis+coordinate_locking+\
             component_wise_scaling+gpr_step+gan_step+rl_step+geodesic_step+perturbation+trim"
        );
        // `gediis` replaces `ediis` (but not `gdiis`, which is not a substring of it).
        assert_eq!(parts("lbfgs_gediis").1, "gediis");
        // TRIM only with Hessian-based methods.
        assert_eq!(parts("fire_trim").1, "");
        assert_eq!(parts("rsprfo_bofill_trim").1, "trim");
    }

    #[test]
    fn user_registered_generator_takes_priority() {
        let mut reg = MethodRegistry::default();
        reg.register_generator(
            "my_fire",
            false,
            Arc::new(|_: &str, _: &MethodContext| {
                Ok(Box::new(Fire::default()) as Box<dyn StepGenerator>)
            }),
        );
        assert_eq!(
            reg.resolve("MY_FIRE", &ctx()).unwrap().generator.name(),
            "fire"
        );
    }
}
