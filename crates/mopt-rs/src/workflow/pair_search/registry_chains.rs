//! The chain methods in the registry of two-point methods: the chain relaxation of
//! `nebmain` (NEB and its relatives, with `nebmain`'s own options), the direct MaxFlux method
//! (`nebmain -dmf`), and the binary-image transition state search (`optmain -mf bitss`), each
//! a thin wrapper of the code of its command on the surface of the registry.
//!
//! The chain relaxation has `nebmain`'s defaults, its Hessian-based steps included: every
//! image of structures gets a model Hessian (`-modelhess`, `lindh2007d3` without it; the
//! calculator's with `-fc`), and `-modelhess none` (in `options` or `settings`) turns them
//! off, leaving steps along the forces alone. Chains of plain variables take the second
//! derivatives of the objective, as in `nebmain`.
//!
//! The options of the commands (`nebmain`'s, those of `optmain -mf bitss`) are read by the
//! command layer, which hands its parsers in with the run ([`SearchRun::commands`], a
//! [`CommandSettings`]), so that this layer does not depend on the command line
//! (`docs/source/architecture.md`, layers).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::path::Path;
use std::sync::Arc;

use faer::Col;

use super::registry::{Ends, PairInput, PairMethod, PairResult, PairSettings};
use super::surface::SearchSurface;
use super::SearchRun;
use crate::error::{MoptError, MoptResult};
use crate::model_hessian::ModelHessian;
use crate::path::band::machine::{ChainEnd, ChainRelaxation, ChainSettings, ImageValue};
use crate::path::band::ChainKind;
use crate::path::bitss::{Bitss, BitssAction, BitssSettings};
use crate::path::dmf::{DirectMaxFlux, DmfEnd, DmfSettings};
use crate::path::interp::{FirstPath, PathFrame, PathSettings};
use crate::path::points::{PointRequest, PointValue};
use crate::workflow::chain::{relax_chain, ChainSurface, Flow};
use crate::workflow::chain_input::{first_chain_in, FirstChain};
use crate::workflow::dmf::run_dmf;

/// The chain members.
pub fn standard() -> Vec<Arc<dyn PairMethod>> {
    vec![
        Arc::new(NebMember { dmf: false }),
        Arc::new(NebMember { dmf: true }),
        Arc::new(BitssMember),
    ]
}

/// A surface of the registry as the surface of a chain (every image in the state of the
/// first end).
pub struct ChainOnSearch<'a> {
    surface: &'a dyn SearchSurface,
    model: Option<Box<dyn ModelHessian>>,
}

impl<'a> ChainOnSearch<'a> {
    /// The chain on `surface`, its Hessians from the surface.
    pub fn new(surface: &'a dyn SearchSurface) -> Self {
        Self {
            surface,
            model: None,
        }
    }

    /// The Hessians of the images from `model` (structures in Cartesian coordinates), the
    /// bias potentials of the surface exact on top.
    pub fn with_model_hessian(mut self, model: Box<dyn ModelHessian>) -> Self {
        self.model = Some(model);
        self
    }
}

impl ChainSurface for ChainOnSearch<'_> {
    fn description(&self) -> String {
        self.surface.description()
    }

    fn evaluate(
        &self,
        point: &Col<f64>,
        _index: usize,
        need_hessian: bool,
    ) -> MoptResult<ImageValue> {
        let model = self.model.as_deref().filter(|_| need_hessian);
        let request = PointRequest {
            x: point.clone(),
            gradient: true,
            hessian: need_hessian && model.is_none(),
        };
        let v = self.surface.evaluate(point, 0, &request)?;
        let hessian = match model {
            None => v.hessian,
            Some(m) => Some(
                self.surface
                    .model_hessian(point, 0, &v.gradient, m)?
                    .ok_or_else(|| {
                        MoptError::unsupported(format!(
                            "the model Hessian {} needs structures in Cartesian coordinates",
                            m.name()
                        ))
                    })?,
            ),
        };
        Ok(ImageValue {
            energy: v.biased_energy,
            gradient: v.biased_gradient,
            hessian,
        })
    }

    fn begin_iteration(&self, iteration: usize) {
        self.surface.begin_iteration(iteration);
    }

    fn take_notes(&self) -> Vec<String> {
        self.surface.take_notes()
    }
}

struct NebMember {
    dmf: bool,
}

/// The spacing of the images that replace the points of a first path (Angstrom; the `-nd 0.5`
/// of the step 2 of AutoTS).
pub const FIRST_PATH_SPACING_ANGSTROM: f64 = 0.5;

/// What the chain members take from the settings of `nebmain` (or `nebmain -dmf`).
#[derive(Debug, Clone)]
pub struct ChainCommandSettings {
    /// The chain relaxation.
    pub chain: ChainSettings,
    /// The direct MaxFlux method (`nebmain -dmf`).
    pub dmf: Option<DmfSettings>,
    /// The model Hessian of the images (`-modelhess`; `None`: none).
    pub model_hessian: Option<String>,
    /// Whether the calculator's Hessians are asked for (`-fc`).
    pub exact_hessians: bool,
    /// The first chain between two structures (`nebmain`'s default when not given).
    pub first_path: FirstPath,
    /// Whether the images of the first chain are evenly spaced.
    pub first_equal_spacing: bool,
}

/// The settings of the members of the registry that are those of a command, read from the keys
/// of the member by the command layer, which knows the options of its commands.
pub trait CommandSettings: Sync {
    /// The settings of `nebmain` (`dmf`: of `nebmain -dmf`) from the keys of a chain member:
    /// `settings` (a JSON object of `nebmain`'s settings, as in its settings file: the
    /// `step2_settings` of a workflow), then `images`, `iterations`, `path`, and `options`
    /// (further options of `nebmain` as on its command line) over them; `dir` stands in for
    /// the input that `nebmain` wants (it is not read).
    fn chain(
        &self,
        settings: &PairSettings,
        dir: &Path,
        dmf: bool,
    ) -> MoptResult<ChainCommandSettings>;

    /// The settings of BITSS (`optmain -mf bitss`) from `key=value` tokens.
    fn bitss(&self, tokens: &[String]) -> MoptResult<BitssSettings>;
}

/// The model Hessian of the images of a chain of structures, when its steps use Hessians
/// and the calculator's are not asked for (`-fc`).
pub(super) fn chain_model(
    config: &ChainCommandSettings,
) -> MoptResult<Option<Box<dyn ModelHessian>>> {
    match (
        &config.model_hessian,
        config.exact_hessians,
        &config.chain.rfo,
    ) {
        (Some(name), false, Some(_)) => {
            if crate::pes::o1numhess::is_o1numhess(name) {
                return Err(MoptError::unsupported(format!(
                    "neb: -modelhess {name} (a Hessian from further gradients) is not available \
                     to the chains of the registry of two-point methods; another model or -fc"
                )));
            }
            Ok(Some(crate::model_hessian::resolve(name)?))
        }
        _ => Ok(None),
    }
}

impl PairMethod for NebMember {
    fn name(&self) -> &'static str {
        if self.dmf {
            "dmf"
        } else {
            "neb"
        }
    }
    fn summary(&self) -> &'static str {
        if self.dmf {
            "direct MaxFlux method of nebmain -dmf (Koda, Saito 2024)"
        } else {
            "chain relaxation of nebmain (NEB and its relatives, with nebmain's options)"
        }
    }
    fn ends(&self) -> Ends {
        Ends::Path
    }
    fn keys(&self) -> &'static [(&'static str, &'static str)] {
        &[
            ("images", "images between the ends of the first chain (neb: 10; dmf: 1); with a first path, its points are laid one onto the other and replaced by images 0.5 Angstrom apart along it"),
            ("iterations", "maximum iterations (neb: 30, dmf: 60)"),
            ("path", "the first chain: linear, lst, idpp, sidpp, geodesic, fbenm, cfbenm (nebmain's default)"),
            ("options", "further options of nebmain, as on its command line (-cineb, -neb_opt ...; -modelhess none: steps along the forces alone)"),
            ("settings", "nebmain's settings as a JSON object, as in its settings file (step2_settings); the keys above go over them"),
        ]
    }
    fn run(
        &self,
        input: &PairInput,
        settings: &PairSettings,
        surface: &dyn SearchSurface,
        run: &SearchRun,
    ) -> MoptResult<PairResult> {
        self.run_observed(input, settings, surface, run, None)
    }

    fn run_observed(
        &self,
        input: &PairInput,
        settings: &PairSettings,
        surface: &dyn SearchSurface,
        run: &SearchRun,
        observer: Option<&mut dyn crate::workflow::chain::ChainObserver>,
    ) -> MoptResult<PairResult> {
        let config = run.commands.chain(settings, run.dir, self.dmf)?;
        let first = input.first()?;
        let _ = input.second(self.name())?;
        // Every point laid onto the one before it (rigid motions taken out).
        let mut points = vec![first.clone()];
        for p in &input.points[1..] {
            let before = points[points.len() - 1].clone();
            points.push(input.geometry.laid_onto(p, &before));
        }
        let given_path = points.len() > 2;
        let second = points[points.len() - 1].clone();
        let kind = if input.structures {
            ChainKind::Atoms
        } else {
            ChainKind::Variables
        };
        let images = settings.usize("images", if config.dmf.is_some() { 1 } else { 10 })?;
        let how = if given_path {
            // A first path (a trajectory): its points as they are, replaced by images evenly
            // apart along it, as nebmain does with the frames of a trajectory (-nd).
            FirstChain {
                inserted: 0,
                equal_spacing: false,
                spacing: Some(input.length(FIRST_PATH_SPACING_ANGSTROM)),
                path: crate::path::interp::FirstPath::Linear,
            }
        } else {
            FirstChain {
                inserted: images,
                equal_spacing: config.first_equal_spacing,
                spacing: None,
                path: if input.structures {
                    config.first_path
                } else {
                    FirstPath::Linear
                },
            }
        };
        let frame = PathFrame {
            elements: input.elements.clone().unwrap_or_default(),
            fixed_components: input.geometry.frozen.clone(),
            ..PathFrame::default()
        };
        run.record_settings(self.name(), &(&config, &how))?;
        let (chain, _) = first_chain_in(&points, kind, &how, &frame, &PathSettings::default())?;
        // The images of the first chain on the hard constraints of the points (an interpolated
        // image lies off them), and kept there after every iteration below.
        let held = input.geometry.manifold.is_some() || input.step_map.is_some();
        let chain = if held {
            chain
                .iter()
                .map(|x| super::relax_path::settle(input, x))
                .collect::<MoptResult<Vec<_>>>()?
        } else {
            chain
        };
        let mut adapter = ChainOnSearch::new(surface);
        if input.structures && config.dmf.is_none() {
            if let Some(model) = chain_model(&config)? {
                adapter = adapter.with_model_hessian(model);
            }
        }
        if let (Some(_), true) = (&config.dmf, input.geometry.manifold.is_some()) {
            return Err(MoptError::unsupported(
                "dmf: the direct MaxFlux method does not hold constraints beyond fixed atoms \
                 (-pc, -gfix, -uc); use the chain relaxation (neb)",
            ));
        }
        if let Some(s) = config.dmf {
            let mut m = DirectMaxFlux::new(s, &chain)?;
            let o = run_dmf(&mut m, &adapter, &mut |_, _| Ok(Flow::Continue))?;
            let last = o
                .last
                .ok_or_else(|| MoptError::calc("dmf: the run made no iteration"))?;
            let path: Vec<(Col<f64>, f64)> = last
                .points
                .iter()
                .cloned()
                .zip(last.energies.iter().copied())
                .collect();
            return Ok(PairResult {
                method: "dmf".to_string(),
                converged: matches!(o.end, Some(DmfEnd::Converged | DmfEnd::Settled)),
                end: format!("{:?}", o.end),
                transition_states: vec![(last.highest.clone(), last.e_max)],
                path: Some(path),
                ends: vec![first.clone(), second],
                counts: Default::default(),
                interrupted: false,
            });
        }
        let mut chain_settings = config.chain.clone();
        chain_settings.kind = kind;
        // Fixed variables (the atoms of -fix) keep their values in every image, as in the
        // relaxation of a path after a method (`relax_path`).
        chain_settings.fixed_components = input.geometry.frozen.clone();
        chain_settings.align &= input.structures && input.geometry.aligns();
        if let Some(rfo) = chain_settings.rfo.as_mut() {
            rfo.project_updates = !input.rigid_in_updates;
        }
        let mut m = ChainRelaxation::new(chain_settings, chain)?;
        if let (Some(t), true) = (&input.template, input.structures) {
            m = m.with_structure(t.clone())?;
        }
        let mut keep = super::relax_path::KeepOnConstraints { input };
        let mut observers: Vec<&mut dyn crate::workflow::chain::ChainObserver> = Vec::new();
        if let Some(o) = observer {
            observers.push(o);
        }
        if held {
            observers.push(&mut keep);
        }
        let o = relax_chain(&mut m, &adapter, &mut observers)?;
        let last = o
            .last
            .ok_or_else(|| MoptError::calc("neb: the relaxation made no iteration"))?;
        let path: Vec<(Col<f64>, f64)> = last
            .chain
            .iter()
            .cloned()
            .zip(last.energies.iter().copied())
            .collect();
        let mut maxima: Vec<(Col<f64>, f64)> = last
            .maxima
            .iter()
            .filter_map(|&k| path.get(k).cloned())
            .collect();
        if maxima.is_empty() {
            let n = path.len();
            if let Some(top) = path
                .iter()
                .enumerate()
                .filter(|(k, _)| *k > 0 && *k + 1 < n)
                .max_by(|a, b| a.1 .1.total_cmp(&b.1 .1))
            {
                maxima.push(top.1.clone());
            }
        }
        Ok(PairResult {
            method: self.name().to_string(),
            converged: o.end == Some(ChainEnd::Converged),
            end: format!("{:?}", o.end),
            transition_states: maxima,
            path: Some(path),
            ends: vec![first.clone(), second],
            counts: Default::default(),
            interrupted: false,
        })
    }
}

struct BitssMember;

impl PairMethod for BitssMember {
    fn name(&self) -> &'static str {
        "bitss"
    }
    fn summary(&self) -> &'static str {
        "binary-image transition state search of optmain -mf bitss (Avis, Panter, Kusumaatmaja 2022)"
    }
    fn ends(&self) -> Ends {
        Ends::Two
    }
    fn keys(&self) -> &'static [(&'static str, &'static str)] {
        &[
            ("f", "share by which the separation shrinks per stage (0.5)"),
            ("alpha", "coefficient of the energy term (10)"),
            ("beta", "coefficient of the distance term (0.1)"),
            (
                "interval",
                "iterations after which the coefficients are set again (100)",
            ),
            (
                "points",
                "points between the images for the barrier estimate (3)",
            ),
            ("min_separation", "smallest separation (0.01 Bohr)"),
            ("stages", "stages at most (30)"),
            (
                "step_share",
                "longest step as a share of the separation (0.5; none)",
            ),
            (
                "midpoint_shift",
                "end when the midpoint moved less than this share of a stage",
            ),
            (
                "midpoint_gradient",
                "end on the forces at the midpoint (true)",
            ),
        ]
    }
    fn run(
        &self,
        input: &PairInput,
        settings: &PairSettings,
        surface: &dyn SearchSurface,
        run: &SearchRun,
    ) -> MoptResult<PairResult> {
        let tokens: Vec<String> = settings
            .keys()
            .into_iter()
            .filter_map(|k| settings.text(&k).map(|v| format!("{k}={v}")))
            .collect();
        let s = run.commands.bitss(&tokens)?;
        let first = input.first()?;
        let second = input.geometry.laid_onto(input.second("bitss")?, first);
        run.record_settings(self.name(), &s)?;
        let mut m = Bitss::new(s, first, &second)?;
        if !input.geometry.frozen.is_empty() {
            m = m.with_frozen(&input.geometry.frozen)?;
        }
        let mut paths = super::ImagePaths::default();
        let end = loop {
            let points = match m.ask() {
                BitssAction::Evaluate(p) => p,
                BitssAction::Finished(end) => break end,
            };
            let mut values = Vec::with_capacity(points.len());
            for p in &points {
                let request = PointRequest {
                    x: p.x.clone(),
                    gradient: p.gradient,
                    hessian: p.hessian,
                };
                let v = surface.evaluate(&p.x, 0, &request)?;
                values.push(PointValue {
                    energy: v.biased_energy,
                    gradient: p.gradient.then(|| v.biased_gradient.clone()),
                    hessian: if p.hessian { v.hessian } else { None },
                });
            }
            let report = m.tell(values)?;
            paths.push(&report.images, report.energies);
        };
        let x = m.transition_state_estimate();
        let e = surface
            .evaluate(&x, 0, &PointRequest::energy(x.clone()))?
            .biased_energy;
        let (a, b) = m.images();
        Ok(PairResult {
            method: self.name().to_string(),
            converged: end.converged(),
            end: format!("{end:?}"),
            transition_states: vec![(x, e)],
            path: Some(paths.join()),
            ends: vec![a, b],
            counts: Default::default(),
            interrupted: false,
        })
    }
}
