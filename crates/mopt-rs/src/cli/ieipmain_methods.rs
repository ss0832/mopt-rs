//! The methods of `ieipmain`: their state machines made from the settings and run by the
//! drivers of [`crate::workflow::pair_search`] on a surface (structures or plain variables).
//!
//! MultiOptPy: `ieip.py:iEIP.optimize` (the choice of the method and its call).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::sync::Arc;

use faer::{Col, Mat};

use super::ieipmain_followup::CrossingStates;
use super::ieipmain_settings::{IeipMethod, IeipmainConfig};
use crate::coord::connectivity::BondConnectivity;
use crate::core_infra::log::Logger;
use crate::core_infra::rng::Rng;
use crate::domain::{MolecularSystem, Units};
use crate::error::{MoptError, MoptResult};
use crate::irc::SphereSettings;
use crate::math::linalg::col_norm;
use crate::math::projection::{build_tr_rot_basis, project_out_basis_vec};
use crate::opt::vector::VectorSettings;
use crate::path::dimer::{Dimer, DimerSettings};
use crate::path::gsm::DrivingCoordinate;
use crate::path::ieip::{ElasticImagePair, IeipEnd, IeipSettings, MicroCoordinates, MicroSettings};
use crate::path::manifold::PointManifold;
use crate::path::newton_trajectory::{GrowingNewtonTrajectory, NtDirection, NtSettings};
use crate::path::points::StepMap;
use crate::path::spring_pair::{SpringPair, SpringPairSettings};
use crate::path::string_nodes::PointGeometry;
use crate::path::twopshs::{TwoPointShs, TwoPshsSettings};
use crate::workflow::chain::Flow;
use crate::workflow::explore::addf::{Addf, AddfSettings};
use crate::workflow::pair_search::registry::FragmentTest;
use crate::workflow::pair_search::relax_path::{relax_path, PathRelaxation};
use crate::workflow::pair_search::string_runs::write_path;
use crate::workflow::pair_search::{
    run_addf, run_dimer, run_ieip, run_newton_trajectory, run_two_point_shs, SearchRun,
    SearchSurface,
};
use crate::workflow::pair_search::{PairInput, PairMethodRegistry, PairSettings};

/// Bonds up to this multiple of the sum of the covalent radii hold a structure together in
/// the test for dissociations (the paths of ADDF, the estimates of every method). The
/// covalent bonds of the test molecules stretch to about 1.3 times the sum at their
/// transition states (CH3NC -> CH3CN: C-N 1.86 Angstrom against 1.46), so 1.5 leaves them
/// whole.
pub const DISSOCIATION_SCALE: f64 = 1.5;

/// What the points of a run are.
#[derive(Clone, Default)]
pub struct Geometry {
    /// The first structure (`None` for plain variables, and for structures in the
    /// coordinates of `-coord`).
    pub template: Option<MolecularSystem>,
    /// Whether the surface has no preferred orientation and position (an isolated system
    /// on such a calculator): the structures are laid onto each other and the rigid-body
    /// motions are left out.
    pub rigid: bool,
    /// Whether the surface, the bias potentials, and the constraints are blind to
    /// rigid-body motions (whether or not atoms are fixed).
    pub invariant: bool,
    /// Variables that keep their values (`-fix`).
    pub frozen: Vec<usize>,
    /// The points are coordinates of structures in Bohr and radians (`-coord`).
    pub bohr_lengths: bool,
    /// The test whether a point has fallen apart into fragments (structures).
    pub fragments: Option<FragmentTest>,
    /// The steps of iEIP and DS-AFIR through the coordinates of `-coord`, or back onto the
    /// constraints (`-pc`, `-gfix`, `-uc`).
    pub step_map: Option<StepMap>,
    /// The constraints beyond fixed atoms as a manifold of the points (the methods that are
    /// not made of steps hold them through it).
    pub manifold: Option<Arc<PointManifold>>,
}

impl Geometry {
    /// Lengths given in Angstrom, in the units of the points.
    fn length(&self, angstrom: f64) -> f64 {
        if self.template.is_some() || self.bohr_lengths {
            angstrom * Units::ANGSTROM_TO_BOHR
        } else {
            angstrom
        }
    }

    /// Orthonormal directions left out of scaled coordinates and random directions: the
    /// fixed variables and, on a surface blind to rigid-body motions, the rigid-body motions
    /// that leave the fixed atoms where they are (rotations about one or two fixed atoms),
    /// or all rigid-body motions without fixed atoms.
    fn removed(&self) -> Option<Mat<f64>> {
        let template = self.template.as_ref()?;
        let n = template.dof();
        if !self.frozen.is_empty() {
            let mut columns: Vec<Col<f64>> = self
                .frozen
                .iter()
                .map(|&f| Col::from_fn(n, |i| if i == f { 1.0 } else { 0.0 }))
                .collect();
            if self.invariant {
                // The fixed variables are the unit rows of the constraints they make.
                let kept = crate::math::projection::rigid_motions_keeping(template, &columns);
                columns.extend(kept);
            }
            return Some(Mat::from_fn(n, columns.len(), |i, j| columns[j][i]));
        }
        self.rigid.then(|| build_tr_rot_basis(template, false))
    }

    /// A random direction (seeded) without rigid-body motion and fixed atoms.
    fn random_direction(&self, n: usize, seed: u64) -> MoptResult<Col<f64>> {
        let mut rng = Rng::from_seed(seed);
        let mut v = Col::from_fn(n, |_| rng.standard_normal());
        if let Some(d) = self.removed() {
            v = project_out_basis_vec(&d, &v);
        }
        if col_norm(&v) == 0.0 {
            return Err(MoptError::config("no direction is left to start along"));
        }
        Ok(v)
    }
}

/// How a method went.
#[derive(Debug, Clone)]
pub struct MethodOutcome {
    /// Whether it ended on its criteria.
    pub converged: bool,
    /// Whether a `STOP` file interrupted it.
    pub interrupted: bool,
    /// Why it ended, in words.
    pub end: String,
    /// The estimates of transition states (point, energy if known).
    pub transition_states: Vec<(Col<f64>, Option<f64>)>,
    /// The path of the method with its energies, where it makes one.
    pub path: Option<Vec<(Col<f64>, f64)>>,
}

/// The direction from the first point to the second, or a random one.
fn direction(
    points: &[Col<f64>],
    geometry: &Geometry,
    seed: u64,
    run: &SearchRun,
) -> MoptResult<Col<f64>> {
    match points {
        [a, b, ..] => Ok(b - a),
        [a] => {
            run.log.info(&format!(
                "Starting direction: random (seed {seed}; -ieip_opt seed=); a second structure \
                 gives the direction towards it."
            ));
            geometry.random_direction(a.nrows(), seed)
        }
        [] => Err(MoptError::config("no starting point")),
    }
}

/// The points with the second displaced by `noise` along a random direction (seeded),
/// where it only gives a direction or a start.
fn with_noise(
    points: &[Col<f64>],
    geometry: &Geometry,
    noise: f64,
    seed: u64,
    run: &SearchRun,
) -> MoptResult<Vec<Col<f64>>> {
    let mut out = points.to_vec();
    if let (true, [_, b]) = (noise > 0.0, out.as_mut_slice()) {
        let u = geometry.random_direction(b.nrows(), seed.wrapping_add(7919))?;
        let size = col_norm(&u);
        let displaced = &*b + &u * (noise / size);
        // Back onto the constraints, where the steps of the methods end too.
        *b = match &geometry.step_map {
            Some(map) => map(b, &displaced)?,
            None => displaced,
        };
        run.log.info(&format!(
            "The second structure is displaced by {noise} along a random direction (-ieip_opt \
             symmetry_noise=, seed=), so that a symmetric start can leave its symmetry."
        ));
    }
    Ok(out)
}

fn sphere(config: &IeipmainConfig) -> SphereSettings {
    SphereSettings {
        max_iterations: config.extra.sphere_iterations.max(1),
        gradient_tolerance: config.extra.sphere_gradient,
        ..SphereSettings::default()
    }
}

/// Run the method of `config` from `points` (one or two) on `surface`, and the relaxation
/// of its path afterwards where it is asked for (`-ieip_opt method_relax=lup|neb`; the
/// methods of the registry run it themselves).
pub fn run_method(
    config: &IeipmainConfig,
    surface: &dyn SearchSurface,
    points: &[Col<f64>],
    geometry: &Geometry,
    run: &SearchRun,
) -> MoptResult<MethodOutcome> {
    if config.method.registry_name().is_some() {
        return run_method_only(config, surface, points, geometry, run);
    }
    let settings = PairSettings::new(config.extra.relax_keys());
    let input = pair_input(config, points, geometry, Vec::new());
    let relaxation = PathRelaxation::from_settings(&settings, &input)?;
    if relaxation.is_some()
        && matches!(
            config.method,
            IeipMethod::Dimer | IeipMethod::Addf | IeipMethod::NewtonTrajectory
        )
    {
        return Err(MoptError::unsupported(format!(
            "method_relax: the {} makes no path between two structures to relax",
            config.method.title()
        )));
    }
    if relaxation.is_some() {
        crate::workflow::pair_search::relax_path::check_relaxation(surface)?;
    }
    let mut outcome = run_method_only(config, surface, points, geometry, run)?;
    let (Some(relaxation), Some(path), false) =
        (relaxation, outcome.path.clone(), outcome.interrupted)
    else {
        return Ok(outcome);
    };
    let relaxed = relax_path(&relaxation, &path, &input, surface, run)?;
    write_path(surface, run, "relaxed", &relaxed.path)?;
    outcome.transition_states = relaxed
        .maxima
        .into_iter()
        .map(|(x, e)| (x, Some(e)))
        .collect();
    outcome.path = Some(relaxed.path);
    Ok(outcome)
}

/// The method of `config` alone.
fn run_method_only(
    config: &IeipmainConfig,
    surface: &dyn SearchSurface,
    points: &[Col<f64>],
    geometry: &Geometry,
    run: &SearchRun,
) -> MoptResult<MethodOutcome> {
    let e = &config.extra;
    let first = points
        .first()
        .ok_or_else(|| MoptError::config("no starting point"))?;
    let noisy = match &config.method {
        // DS-AFIR as well: between ends of a symmetry that the saddle does not have (two
        // linear or planar structures), no gradient leads out of that symmetry.
        IeipMethod::Dimer
        | IeipMethod::SpringPair
        | IeipMethod::TwoPshs
        | IeipMethod::NewtonTrajectory
        | IeipMethod::DsAfir => with_noise(points, geometry, e.symmetry_noise, e.seed, run)?,
        _ => points.to_vec(),
    };
    let points = noisy.as_slice();
    match &config.method {
        IeipMethod::Ieip => run_ieip_method(config, surface, points, geometry, run),
        IeipMethod::Dimer => {
            let settings = DimerSettings {
                separation: config.dimer_separation,
                trial_angle: config.dimer_trial_angle,
                max_iterations: config.dimer_iterations.max(1),
                max_rotations: e.rotations,
                rotation_tolerance: e.rotation_angle_deg.to_radians(),
                max_step: e.max_step,
                criteria: e.criteria,
                translation: e.dimer_translation,
                rate_across: e.dimer_rates[0],
                rate_along: e.dimer_rates[1],
                ..DimerSettings::default()
            };
            let axis = direction(points, geometry, e.seed, run)?;
            let mut m = Dimer::new(settings, first, &axis)?.with_frozen(geometry.frozen.clone())?;
            if let (Some(template), true) = (&geometry.template, geometry.rigid) {
                m = m.with_rigid_body(template.clone())?;
            }
            if let Some(map) = &geometry.step_map {
                m = m.with_step_map(map.clone());
            }
            let o = run_dimer(&mut m, surface, run, &mut |_| Ok(Flow::Continue))?;
            Ok(point_outcome(o))
        }
        IeipMethod::SpringPair => {
            let settings = SpringPairSettings {
                spring_length: e.spm[0],
                offset: e.spm[1],
                drift_rate: e.spm[2],
                spring_rate: e.spm[3],
                climb_rate: e.spm[4],
                drift_tolerance: e.spm[5],
                drift_steps: e.spm_drift_steps.max(1),
                max_cycles: config.max_iterations,
                max_step: e.max_step,
                criteria: e.criteria,
                align: geometry.rigid && geometry.frozen.is_empty() && geometry.template.is_some(),
            };
            let toward = direction(points, geometry, e.seed, run)?;
            let mut m = SpringPair::new(settings, first, &toward)?;
            if !geometry.frozen.is_empty() {
                m = m.with_frozen(geometry.frozen.clone())?;
            }
            if let Some(map) = &geometry.step_map {
                m = m.with_step_map(map.clone())?;
            }
            let (o, path) =
                crate::workflow::pair_search::point_runs::spring_pair_path(&mut m, surface, run)?;
            Ok(MethodOutcome {
                path: Some(path),
                ..point_outcome(o)
            })
        }
        IeipMethod::TwoPshs => {
            let settings = TwoPshsSettings {
                step: geometry.length(config.twopshs_step),
                max_spheres: config.twopshs_spheres.max(1),
                sphere: sphere(config),
            };
            let second = points
                .get(1)
                .ok_or_else(|| MoptError::config("2PSHS needs two structures"))?;
            let mut m = TwoPointShs::new(settings, first, second, geometry.removed())?;
            if let Some(manifold) = &geometry.manifold {
                m = m.with_manifold(Arc::clone(manifold));
            }
            let o = run_two_point_shs(&mut m, surface, run, &mut |_| Ok(Flow::Continue))?;
            let path = m
                .path()
                .iter()
                .rev()
                .map(|p| (p.x.clone(), p.energy))
                .collect();
            Ok(MethodOutcome {
                path: Some(path),
                ..point_outcome(o)
            })
        }
        IeipMethod::NewtonTrajectory => {
            let mut minimizer = VectorSettings::default();
            if config.arguments.is_present("opt_method") {
                minimizer.methods = config.arguments.get_list("opt_method").to_vec();
            }
            let settings = NtSettings {
                step_length: config.gnt_step,
                rms_tolerance: e.gnt_tolerance,
                corrector_iterations: config.gnt_corrector,
                max_iterations: config.max_iterations,
                minimizer,
                align: geometry.rigid && geometry.template.is_some(),
                rigid_body: geometry.template.clone().filter(|_| geometry.rigid),
                ..NtSettings::default()
            };
            let source = if config.gnt_bonds.is_empty() {
                let second = points.get(1).ok_or_else(|| {
                    MoptError::config("-gnt needs a second structure or -gnt_vec")
                })?;
                NtDirection::Toward(second.clone())
            } else {
                if geometry.template.is_none() {
                    return Err(MoptError::unsupported(
                        "-gnt_vec names atoms; it needs structures",
                    ));
                }
                NtDirection::Bonds(config.gnt_bonds.clone())
            };
            let mut m = GrowingNewtonTrajectory::new(settings, first, source)?;
            if !geometry.frozen.is_empty() {
                m = m.with_frozen(geometry.frozen.clone())?;
            }
            if let Some(manifold) = &geometry.manifold {
                m = m.with_manifold(Arc::clone(manifold))?;
            }
            let o = run_newton_trajectory(&mut m, surface, run, &mut |_| Ok(Flow::Continue))?;
            Ok(point_outcome(o))
        }
        IeipMethod::Addf => {
            if points.len() > 1 {
                run.log
                    .info("ADDF starts from the first structure; the second is not used.");
            }
            let settings = AddfSettings {
                initial_length: geometry.length(e.addf[0]),
                step_length: geometry.length(config.addf_step),
                max_spheres: config.addf_spheres.max(1),
                paths: config.addf_paths,
                sphere: sphere(config),
                duplicate_angle: e.addf[1],
                gradient_tolerance: e.addf[2],
            };
            let mut m = Addf::new(settings, first, geometry.removed())?;
            if let Some(manifold) = &geometry.manifold {
                m = m.with_manifold(Arc::clone(manifold));
            }
            if let Some(test) = geometry.fragments.clone() {
                // A dissociation channel: more fragments than at the start (bonds up to
                // DISSOCIATION_SCALE times the sum of the covalent radii).
                m = m.with_dissociation_test(Box::new(move |x| test(x)));
            }
            let o = run_addf(&mut m, surface, run, &mut |_| Ok(Flow::Continue))?;
            Ok(MethodOutcome {
                converged: !o.transition_states.is_empty(),
                interrupted: o.interrupted,
                end: format!(
                    "{} transition state estimate(s){}",
                    o.transition_states.len(),
                    match o.end {
                        Some(crate::workflow::explore::addf::AddfEnd::AllPathsEnded) =>
                            "; every path ended",
                        Some(crate::workflow::explore::addf::AddfEnd::SphereLimit) =>
                            "; the spheres are used up",
                        None => "",
                    }
                ),
                transition_states: o
                    .transition_states
                    .into_iter()
                    .map(|(_, x, e)| (x, Some(e)))
                    .collect(),
                path: None,
            })
        }
        IeipMethod::ModelFunction(_) => Err(MoptError::config(
            "-mf runs optmain; not a method of points",
        )),
        IeipMethod::Gsm
        | IeipMethod::SeGsm(_)
        | IeipMethod::Fsm
        | IeipMethod::DsAfir
        | IeipMethod::Lmm => {
            let name = config
                .method
                .registry_name()
                .ok_or_else(|| MoptError::config("not a method of the registry"))?;
            run_registered(name, config, surface, points, geometry, run)
        }
    }
}

/// Fragments of a structure (bonds up to `scale` times the sum of the covalent radii); with
/// `atoms`, only those that hold one of them.
fn fragments_of(s: &MolecularSystem, scale: f64, atoms: Option<&[usize]>) -> usize {
    let conn = BondConnectivity::from_system(s, scale);
    match atoms {
        None => conn.fragments.len(),
        Some(atoms) => conn
            .fragments
            .iter()
            .filter(|f| f.iter().any(|a| atoms.contains(&a.index())))
            .count(),
    }
}

/// Fragments of a structure (bonds up to `scale` times the sum of the covalent radii).
fn fragments(s: &MolecularSystem, scale: f64) -> usize {
    fragments_of(s, scale, None)
}

/// [`fragment_test_with`] in which only fragments that hold one of `atoms` (from 0) count.
pub fn fragment_test_on(template: MolecularSystem, scale: f64, atoms: Vec<usize>) -> FragmentTest {
    let start = fragments_of(&template, scale, Some(&atoms));
    Arc::new(move |x: &Col<f64>| {
        template
            .with_flat_coords_bohr(x)
            .ok()
            .is_some_and(|s| fragments_of(&s, scale, Some(&atoms)) > start)
    })
}

/// The test whether a point (Cartesian, or turned into Cartesian coordinates by
/// `to_cartesian`) has more fragments than `template` (bonds up to [`DISSOCIATION_SCALE`]
/// times the sums of the covalent radii).
pub fn fragment_test(
    template: MolecularSystem,
    to_cartesian: Option<Arc<dyn Fn(&Col<f64>) -> Option<Col<f64>> + Send + Sync>>,
) -> FragmentTest {
    fragment_test_with(template, to_cartesian, DISSOCIATION_SCALE)
}

/// [`fragment_test`] with bonds up to `scale` times the sums of the covalent radii.
pub fn fragment_test_with(
    template: MolecularSystem,
    to_cartesian: Option<Arc<dyn Fn(&Col<f64>) -> Option<Col<f64>> + Send + Sync>>,
    scale: f64,
) -> FragmentTest {
    let start = fragments(&template, scale);
    Arc::new(move |x: &Col<f64>| {
        let cartesian = match &to_cartesian {
            Some(map) => map(x),
            None => Some(x.clone()),
        };
        cartesian
            .and_then(|c| template.with_flat_coords_bohr(&c).ok())
            .is_some_and(|s| fragments(&s, scale) > start)
    })
}

/// What a method of the registry (and the relaxation of a path) is given.
fn pair_input(
    config: &IeipmainConfig,
    points: &[Col<f64>],
    geometry: &Geometry,
    driving: Vec<DrivingCoordinate>,
) -> PairInput {
    PairInput {
        points: points.to_vec(),
        geometry: PointGeometry::of_structure(
            geometry.template.as_ref(),
            geometry.rigid,
            geometry.frozen.clone(),
        )
        .with_manifold(geometry.manifold.clone()),
        structures: geometry.template.is_some(),
        elements: geometry.template.as_ref().map(|t| t.elements.clone()),
        template: geometry.template.clone(),
        driving,
        direction: None,
        fragments: geometry.fragments.clone(),
        bohr_lengths: geometry.bohr_lengths,
        step_map: geometry.step_map.clone(),
        rigid_in_updates: !config.project_updates,
    }
}

/// A method of the registry of two-point methods (`-gsm`, `-se_gsm`, `-fsm`, `-dsafir`).
fn run_registered(
    name: &str,
    config: &IeipmainConfig,
    surface: &dyn SearchSurface,
    points: &[Col<f64>],
    geometry: &Geometry,
    run: &SearchRun,
) -> MoptResult<MethodOutcome> {
    let driving = match &config.method {
        IeipMethod::SeGsm(texts) => texts
            .iter()
            .map(|t| DrivingCoordinate::parse(t))
            .collect::<MoptResult<Vec<_>>>()?,
        _ => Vec::new(),
    };
    let input = pair_input(config, points, geometry, driving);
    let settings = PairSettings::new(config.extra.keys_of(name));
    let r = PairMethodRegistry::default().run(name, &input, &settings, surface, run)?;
    Ok(MethodOutcome {
        converged: r.converged,
        interrupted: r.interrupted,
        end: r.end,
        transition_states: r
            .transition_states
            .into_iter()
            .map(|(x, e)| (x, Some(e)))
            .collect(),
        path: r.path,
    })
}

/// Estimates whose structure falls apart into more fragments than the first structure are
/// dissociations, not transition states between the inputs (the dimer can end on the flat
/// surface of a dissociation, where the forces vanish and the curvature is slightly
/// negative): the log says so and the run does not count as converged.
pub fn check_fragments(
    outcome: &mut MethodOutcome,
    test: &FragmentTest,
    run: &SearchRun,
) -> MoptResult<()> {
    for (k, (x, _)) in outcome.transition_states.iter().enumerate() {
        if test(x) {
            run.log.warn(&format!(
                "Estimate {}: its structure falls apart into more fragments than the first \
                 structure (bonds up to {DISSOCIATION_SCALE} times the sum of the covalent \
                 radii): a dissociation, not a transition state between the inputs.",
                k + 1
            ));
            if outcome.converged {
                outcome.converged = false;
                outcome
                    .end
                    .push_str("; the estimate falls apart into fragments");
            }
        }
    }
    Ok(())
}

fn point_outcome(o: crate::workflow::pair_search::PointOutcome) -> MethodOutcome {
    MethodOutcome {
        converged: o.converged,
        interrupted: o.interrupted,
        end: o.end.unwrap_or_else(|| "stopped".to_string()),
        transition_states: o
            .transition_state
            .into_iter()
            .map(|(x, e)| (x, e.is_finite().then_some(e)))
            .collect(),
        path: None,
    }
}

fn run_ieip_method(
    config: &IeipmainConfig,
    surface: &dyn SearchSurface,
    points: &[Col<f64>],
    geometry: &Geometry,
    run: &SearchRun,
) -> MoptResult<MethodOutcome> {
    let [a, b] = points else {
        return Err(MoptError::config("iEIP needs two structures"));
    };
    let micro = MicroSettings {
        steps: config.microiterations,
        force_tolerance: config.extra.micro_tolerance,
        ..MicroSettings::default()
    };
    let settings = IeipSettings {
        max_iterations: config.max_iterations,
        beta: config.beta,
        micro,
        micro_coordinates: geometry
            .template
            .clone()
            .filter(|_| config.microiterations > 0)
            .map(MicroCoordinates::polar),
        align: geometry.rigid && geometry.frozen.is_empty() && geometry.template.is_some(),
        frozen: geometry.frozen.clone(),
    };
    if settings.align {
        run.log.info("The structures are laid onto each other (centred, the first turned onto the second) at the start and after every step.");
    }
    let mut m = ElasticImagePair::new(settings, a, b)?;
    if let Some(map) = &geometry.step_map {
        m = m.with_step_map(map.clone());
    }
    let o = run_ieip(&mut m, surface, run, &mut |_| Ok(Flow::Continue))?;
    Ok(MethodOutcome {
        converged: o.end == Some(IeipEnd::Converged),
        interrupted: o.interrupted,
        end: match o.end {
            Some(IeipEnd::Converged) => format!("converged after {} iterations", o.iterations),
            Some(IeipEnd::IterationLimit) => {
                "the iterations are used up (not converged)".to_string()
            }
            None => "stopped".to_string(),
        },
        transition_states: vec![(o.transition_state, None)],
        path: Some(o.path),
    })
}

/// The two states of a crossing that the search estimates: DS-AFIR between two structures
/// in two states (another charge, multiplicity, or electronic state) moves each end on the
/// surface of its own state. iEIP and 2PSHS use both states as MultiOptPy does; the other
/// methods run in the state of the first structure, which the log says.
pub fn crossing_states(
    config: &IeipmainConfig,
    structures: &[MolecularSystem],
    surface: &dyn SearchSurface,
    log: &Logger,
) -> Option<CrossingStates> {
    if !surface.two_surfaces() {
        return None;
    }
    let state = |k: usize| {
        let s = structures[k.min(structures.len().saturating_sub(1))].spin_charge;
        (s.charge, s.multiplicity, config.excited_states[k])
    };
    match &config.method {
        IeipMethod::DsAfir => Some(CrossingStates {
            first: state(0),
            second: state(1),
        }),
        IeipMethod::Ieip | IeipMethod::TwoPshs => None,
        other => {
            log.warn(&format!(
                "The two structures are in different states; {} runs on the surface of the \
                 state of the first structure (-dsafir moves each end on its own).",
                other.title()
            ));
            None
        }
    }
}
