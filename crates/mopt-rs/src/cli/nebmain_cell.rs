//! `nebmain -vc`: a chain of the atoms and the cell, every image with a cell of its own.
//!
//! The images are points of the variables of the variable-cell optimization made from the
//! first structure ([`crate::opt::cell::VariableCell`]: quasi-Cartesian coordinates of the
//! atoms and scaled lattice vectors), relaxed as a chain of plain variables
//! ([`ChainKind::Variables`], distances and tangents Euclidean in these variables) on the
//! enthalpy `E + P V` with the bias potentials ([`cell_value`], the surface of `optmain -vc`).
//! This is the chain the mapper relaxes on a variable cell (`run_mapper` with `variable_cell`
//! in the step settings: `cli::mapper_jobs::cell`, the chain member of the registry of
//! two-point methods): the same vector, the same metric, the same values and Hessians of the
//! images (those of [`crate::workflow::irc_cell::CellObjective`]). The structures that are
//! read may have different cells; every atom of a structure is taken to the periodic image
//! nearest to it in the structure before (in fractional coordinates) before the structures
//! become points of the variables.
//!
//! The Hessians of the images, where the steps use them, are those of the enthalpy in the
//! variables (analytic where the calculator gives them, from differences of the gradient
//! otherwise): a model Hessian (`-modelhess`) is one of the Cartesian coordinates of the atoms
//! and has no part for the cell.
//!
//! Constraints (`-fix`, `-pc`, `-gfix`, `-uc`, `-vc_fix`) are those of `optmain -vc`
//! ([`crate::opt::cell_constraints`]): every image holds its own values of the first chain,
//! the gradient and the Hessian of an image without their directions, the images put back onto
//! them after every iteration ([`KeepConstraints`]); where the images are put elsewhere along
//! the chain, they hold the values interpolated at their fractions of its length (as for a
//! chain of structures). Frozen atoms keep their fractional coordinates.
//!
//! References: D. Sheppard, P. Xiao, W. Chemelewski, D. D. Johnson, G. Henkelman, J. Chem.
//! Phys. 2012, 136, 074103 (the cell in the images of a chain); M. Gubler, M. Krummenacher,
//! H. Huber, S. Goedecker, J. Comput. Phys. X 2023, 17, 100131 (the variables).
//!
//! MultiOptPy: no counterpart (new in mopt-rs; MultiOptPy's `nebmain` keeps the cell fixed).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use faer::Col;
use rayon::prelude::*;

use super::nebmain_settings::{NebmainConfig, DEFAULT_SPACING_ANGSTROM};
use super::variable_cell::{cell_states, VariableCellSettings};
use crate::domain::{MolecularSystem, Position, Units};
use crate::error::{MoptError, MoptResult, ResultExt};
use crate::opt::cell::VariableCell;
use crate::opt::cell_constraints::cell_constraints;
use crate::opt::vector_constraints::VectorConstraintSet;
use crate::path::band::machine::{ChainRelaxation, ChainReport, ChainSettings, ImageValue};
use crate::path::band::redistribute::bezier_by_spacing;
use crate::path::band::surrogate::Features;
use crate::path::band::ChainKind;
use crate::path::interp::FirstPath;
use crate::pes::calculator::Calculator;
use crate::pes::constraint::{Constraint, ConstraintKind};
use crate::pes::term::PesTerm;
use crate::workflow::chain::{
    follow_resolved, ChainObserver, ChainSurface, Flow, HeldByImages, KeepConstraints,
};
use crate::workflow::chain_input::{first_chain, FirstChain};
use crate::workflow::irc_cell::{cell_hessian_of, cell_value, CellParts, CellStates};

/// What of `config` a chain of the atoms and the cell cannot do, refused with the reason
/// before anything is calculated.
///
/// MultiOptPy: no counterpart (new in mopt-rs).
pub fn check_options(config: &NebmainConfig, cell: &VariableCellSettings) -> MoptResult<()> {
    if !cell.forces.is_empty() {
        return Err(MoptError::unsupported(
            "-vc_afir with nebmain: the artificial forces on the cell push it in a search; a \
             chain is relaxed on the surface itself (as the IRC of ircmain -vc)",
        ));
    }
    if config.dmf.is_some() {
        return Err(MoptError::unsupported(
            "-dmf with -vc: the direct MaxFlux method fits a path of structures in one cell or \
             of plain variables; with a variable cell use the chain (without -dmf)",
        ));
    }
    let coordinates = config.coordinates.trim().to_ascii_lowercase();
    if !["", "cart", "cartesian"].contains(&coordinates.as_str()) {
        return Err(MoptError::unsupported(format!(
            "-coord {} with -vc: the chain of a variable cell is relaxed in the variables of \
             the atoms and the cell (those of optmain -vc); internal coordinates of the atoms \
             have no part for the cell",
            config.coordinates
        )));
    }
    if let Some(asked) = config.first_path.filter(|path| *path != FirstPath::Linear) {
        return Err(MoptError::unsupported(format!(
            "-neb_opt path={} with -vc: the images of the first chain are on the straight line \
             in the variables of the atoms and the cell (the interpolations of structures are of \
             one cell)",
            asked.name()
        )));
    }
    if config.surrogate.is_some() && config.surrogate_features == Some(Features::InverseDistances) {
        return Err(MoptError::unsupported(
            "-neb_opt gp_features=distances with -vc: the model sees the points of the variables \
             of the atoms and the cell (gp_features=points)",
        ));
    }
    if config.calculator_options.oniom.is_some() {
        return Err(MoptError::unsupported(
            "-oniom with -vc: a variable cell needs the stress of the whole surface; ONIOM has \
             no stress (as optmain -vc)",
        ));
    }
    Ok(())
}

/// The structures `structures` (the same atoms, each in its own periodic cell) as points of
/// the variables of `map`: every atom taken to the periodic image nearest to it in the
/// structure before, in fractional coordinates along the periodic directions.
///
/// MultiOptPy: no counterpart (new in mopt-rs).
pub fn joined_points(
    map: &VariableCell,
    structures: &[MolecularSystem],
) -> MoptResult<Vec<Col<f64>>> {
    let mut points = Vec::with_capacity(structures.len());
    let mut before: Option<Vec<[f64; 3]>> = None;
    for (k, s) in structures.iter().enumerate() {
        let cell = s.active_cell().ok_or_else(|| {
            MoptError::config(format!(
                "-vc: structure {} has no periodic cell (Lattice=\"...\" in every structure)",
                k + 1
            ))
        })?;
        let mut fractions: Vec<[f64; 3]> = s
            .atoms()
            .map(|a| cell.cart_to_frac(s.position(a).value()))
            .collect();
        if let Some(previous) = &before {
            for (f, p) in fractions.iter_mut().zip(previous) {
                for c in (0..3).filter(|c| cell.periodic_axes[*c]) {
                    f[c] -= (f[c] - p[c]).round();
                }
            }
        }
        let mut joined = s.clone();
        for (atom, f) in s.atoms().zip(&fractions) {
            joined.set_position(atom, Position::new(cell.frac_to_cart(*f)));
        }
        points.push(
            map.variables(&joined)
                .context(format!("structure {}", k + 1))?,
        );
        before = Some(fractions);
    }
    Ok(points)
}

/// The constraints of every image of a chain of the atoms and the cell: every image holds
/// its resolved constraints ([`Constraint::resolve`] at its structure in the first chain) as
/// constraints of the variables ([`cell_constraints`]).
#[derive(Clone)]
pub struct CellChainConstraints {
    map: VariableCell,
    /// The constraints as they were given (frozen atoms are resolved anew at every image the
    /// chain makes: their variables are components of the chain that no step moves).
    given: Vec<Constraint>,
    images: Vec<(Vec<Constraint>, VectorConstraintSet)>,
}

impl CellChainConstraints {
    /// `constraints` resolved at every image of `chain` (points of the variables of `map`).
    pub fn new(
        constraints: &[Constraint],
        map: &VariableCell,
        chain: &[Col<f64>],
    ) -> MoptResult<Self> {
        if constraints
            .iter()
            .any(|c| matches!(c.kind, ConstraintKind::EigenVector(_)) && !c.is_resolved())
        {
            return Err(MoptError::unsupported(
                "-pc eigvec: the direction of a Hessian eigenvector is resolved at the first \
                 Hessian of an optimization; the images of a chain have none to resolve it at",
            ));
        }
        let mut images = Vec::with_capacity(chain.len());
        for (k, point) in chain.iter().enumerate() {
            let sys = map.system(point)?;
            let resolved = constraints
                .iter()
                .map(|c| c.resolve(&sys, None))
                .collect::<MoptResult<Vec<_>>>()
                .context(format!("the constraints of image {k}"))?;
            let set = cell_constraints(map, &resolved, &sys)
                .context(format!("the constraints of image {k}"))?;
            images.push((resolved, set));
        }
        Ok(Self {
            map: map.clone(),
            given: constraints.to_vec(),
            images,
        })
    }

    /// The set of image `image`.
    pub fn set(&self, image: usize) -> Option<&VectorConstraintSet> {
        self.images.get(image).map(|(_, set)| set)
    }

    /// Scalar constraints of an image.
    pub fn per_image(&self) -> usize {
        self.images.first().map_or(0, |(_, set)| set.len())
    }
}

impl HeldByImages for CellChainConstraints {
    fn images(&self) -> usize {
        self.images.len()
    }

    fn drop_images(&mut self, dropped: &[usize]) {
        for &image in dropped.iter().rev() {
            if image < self.images.len() {
                self.images.remove(image);
            }
        }
    }

    fn follow(&mut self, before: &[Col<f64>], after: &[Col<f64>]) -> MoptResult<()> {
        if before.len() != self.images.len() {
            return Err(MoptError::config(format!(
                "constraints of {} images for a chain of {}",
                self.images.len(),
                before.len()
            )));
        }
        let resolved: Vec<&[Constraint]> = self.images.iter().map(|(r, _)| r.as_slice()).collect();
        let followed = follow_resolved(&resolved, before, after, ChainKind::Variables)?;
        let mut images = Vec::with_capacity(after.len());
        for (k, (point, mut resolved)) in after.iter().zip(followed).enumerate() {
            let sys = self.map.system(point)?;
            // Frozen atoms keep the fractional coordinates the chain has for them (Cartesian
            // positions interpolated between images of two cells would not).
            for (held, given) in resolved.iter_mut().zip(&self.given) {
                if matches!(given.kind, ConstraintKind::Frozen(_)) {
                    *held = given.resolve(&sys, None)?;
                }
            }
            let set = cell_constraints(&self.map, &resolved, &sys)
                .context(format!("the constraints of image {k}"))?;
            images.push((resolved, set));
        }
        self.images = images;
        Ok(())
    }

    fn restore(&self, chain: &[Col<f64>]) -> MoptResult<Vec<Col<f64>>> {
        if chain.len() != self.images.len() {
            return Err(MoptError::config(format!(
                "constraints of {} images for a chain of {}",
                self.images.len(),
                chain.len()
            )));
        }
        chain
            .iter()
            .zip(&self.images)
            .enumerate()
            .map(|(k, (point, (_, set)))| set.restore(point).context(format!("image {k}")))
            .collect()
    }
}

/// The enthalpy of the atoms and the cell as the surface of a chain of points of the
/// variables of `map` (the values of [`crate::workflow::irc_cell::CellObjective`]), with the
/// constraints of the images taken out of their gradients and Hessians.
pub struct CellChainSurface<'a> {
    calc: &'a dyn Calculator,
    terms: Vec<(String, Arc<dyn PesTerm>)>,
    map: VariableCell,
    start: MolecularSystem,
    excited_state: usize,
    states: Option<CellStates>,
    constraints: Option<Arc<Mutex<CellChainConstraints>>>,
    bare: Mutex<Vec<(usize, f64)>>,
    iteration: AtomicUsize,
}

impl<'a> CellChainSurface<'a> {
    /// The enthalpy of `calc` with `terms` in the variables `map` of structures like `start`
    /// (in the electronic state `excited_state`, or on the two states of `states`).
    pub fn new(
        calc: &'a dyn Calculator,
        terms: Vec<(String, Arc<dyn PesTerm>)>,
        map: VariableCell,
        start: MolecularSystem,
        excited_state: usize,
        states: Option<CellStates>,
    ) -> Self {
        Self {
            calc,
            terms,
            map,
            start,
            excited_state,
            states,
            constraints: None,
            bare: Mutex::new(Vec::new()),
            iteration: AtomicUsize::new(0),
        }
    }

    /// The constraints of the images (shared with the observer that restores them).
    #[must_use]
    pub fn with_constraints(mut self, constraints: Arc<Mutex<CellChainConstraints>>) -> Self {
        self.constraints = Some(constraints);
        self
    }

    /// The variables.
    pub fn map(&self) -> &VariableCell {
        &self.map
    }

    fn parts(&self) -> CellParts<'_> {
        CellParts {
            calc: self.calc,
            terms: &self.terms,
            map: &self.map,
            start: &self.start,
            excited_state: self.excited_state,
            states: self.states.as_ref(),
            log: None,
        }
    }

    /// The value of image `index` at `point` and its enthalpy without the bias potentials.
    fn value_of(
        &self,
        point: &Col<f64>,
        index: usize,
        need_hessian: bool,
    ) -> MoptResult<(ImageValue, f64)> {
        let step = self.iteration.load(Ordering::Relaxed);
        let of = format!("image {index}");
        let parts = self.parts();
        let v = cell_value(&parts, point, step, &of)?;
        if let (Some(states), Some(values)) = (&self.states, &v.states) {
            states.commit(&v.system, values);
        }
        let mut hessian = if need_hessian {
            Some(cell_hessian_of(&parts, point, step, &of)?)
        } else {
            None
        };
        let mut gradient = v.slope;
        if let Some(shared) = &self.constraints {
            let set = shared
                .lock()
                .map_err(|_| MoptError::calc("the constraints of the chain are not available"))?
                .set(index)
                .cloned();
            if let Some(set) = set {
                let frame = set
                    .frame(point)
                    .context(format!("the constraints of image {index}"))?;
                gradient = frame.project(&gradient);
                hessian = hessian.map(|h| frame.step_hessian(&h));
            }
        }
        Ok((
            ImageValue {
                energy: v.biased,
                gradient,
                hessian,
            },
            v.enthalpy,
        ))
    }

    fn keep_bare(&self, index: usize, enthalpy: f64) {
        if self.terms.is_empty() {
            return;
        }
        if let Ok(mut all) = self.bare.lock() {
            all.push((index, enthalpy));
        }
    }
}

impl ChainSurface for CellChainSurface<'_> {
    fn description(&self) -> String {
        format!(
            "the enthalpy of the atoms and the cell ({} variables) of {}",
            self.map.dimension(),
            self.calc.description()
        )
    }

    fn begin_iteration(&self, iteration: usize) {
        self.iteration.store(iteration, Ordering::Relaxed);
    }

    fn take_bare_energies(&self) -> Vec<(usize, f64)> {
        self.bare
            .lock()
            .map(|mut bare| std::mem::take(&mut *bare))
            .unwrap_or_default()
    }

    fn evaluate(
        &self,
        point: &Col<f64>,
        index: usize,
        need_hessian: bool,
    ) -> MoptResult<ImageValue> {
        let (value, enthalpy) = self.value_of(point, index, need_hessian)?;
        self.keep_bare(index, enthalpy);
        Ok(value)
    }

    fn evaluate_all(
        &self,
        images: &[(usize, &Col<f64>)],
        need_hessian: bool,
    ) -> Vec<MoptResult<ImageValue>> {
        let one = |(index, point): &(usize, &Col<f64>)| self.value_of(point, *index, need_hessian);
        // The images are calculations that do not depend on each other unless the calculator
        // asks for its structures in order or two states learn from point to point.
        let independent = !self.calc.evaluates_in_order() && self.states.is_none();
        let found: Vec<MoptResult<(ImageValue, f64)>> = if images.len() > 1 && independent {
            images.par_iter().map(one).collect()
        } else {
            images.iter().map(one).collect()
        };
        found
            .into_iter()
            .zip(images)
            .map(|(result, (index, _))| {
                result.map(|(value, enthalpy)| {
                    self.keep_bare(*index, enthalpy);
                    value
                })
            })
            .collect()
    }
}

/// The chain of a `nebmain -vc` run as it is set up before the relaxation: the variables, the
/// first chain, the settings of the relaxation, the surface, and the observer of the
/// constraints, with what is said about them for the log. Shared by `nebmain` and the Python
/// `relax_band`.
pub struct CellChain<'a> {
    /// The settings of the relaxation.
    pub settings: ChainSettings,
    /// The first chain (on the constraints already).
    pub chain: Vec<Col<f64>>,
    /// The surface of the images.
    pub surface: CellChainSurface<'a>,
    /// The observer that puts the images back onto their constraints (`None` without).
    pub keep: Option<KeepConstraints<CellChainConstraints>>,
    /// What is said about the chain, in order, for the log.
    pub notes: Vec<String>,
}

/// See [`CellChain`]: the chain of `structures` (`-p`, `-ad`, `-nd`, which is 0.5 Angstrom
/// of the variables unless given, `-ndb`) on the enthalpy of `calc` with the bias potentials
/// `terms`, holding `constraints`; `hessian_directory` for the Hessians of the images.
///
/// MultiOptPy: no counterpart (new in mopt-rs).
#[allow(clippy::too_many_arguments)]
pub fn cell_chain<'a>(
    config: &NebmainConfig,
    cell: &VariableCellSettings,
    structures: &[MolecularSystem],
    calc: &'a dyn Calculator,
    terms: Vec<(String, Arc<dyn PesTerm>)>,
    constraints: &[Constraint],
    hessian_directory: Option<PathBuf>,
) -> MoptResult<CellChain<'a>> {
    check_options(config, cell)?;
    let first = structures
        .first()
        .ok_or_else(|| MoptError::config("a chain needs at least two structures"))?;
    let map = cell.coordinates(first)?;
    let points = joined_points(&map, structures)?;
    let spacing = match config.spacing {
        None if config.bezier_spacing.is_some() => None,
        None => Some(DEFAULT_SPACING_ANGSTROM),
        Some(given) => (given > 0.0).then_some(given),
    };
    let how = FirstChain {
        inserted: config.inserted,
        equal_spacing: config.first_equal_spacing,
        spacing: spacing.map(|angstrom| angstrom * Units::ANGSTROM_TO_BOHR),
        path: FirstPath::Linear,
    };
    let mut chain = first_chain(&points, ChainKind::Variables, &how)?;
    let mut notes = Vec::new();
    if let Some(distance) = config.bezier_spacing {
        let before = chain.len();
        chain = bezier_by_spacing(
            &chain,
            distance * Units::ANGSTROM_TO_BOHR,
            ChainKind::Variables,
        );
        notes.push(format!(
            "The first chain is put on the Bezier curve of its {before} images (-ndb \
             {distance}): {} images.",
            chain.len()
        ));
        if chain.len() < 3 {
            return Err(MoptError::config(format!(
                "-ndb {distance}: the first chain has {} images; a relaxation needs at least \
                 three (a smaller distance)",
                chain.len()
            )));
        }
    }
    notes.push(format!(
        "Variable cell (-vc): {} structures read; first chain of {} images on the straight line \
         in the {} variables of the atoms and the cell (quasi-Cartesian coordinates of the atoms \
         and scaled lattice vectors, weight {}){}; external pressure {} GPa: the profile is the \
         enthalpy E + P V.",
        structures.len(),
        chain.len(),
        map.dimension(),
        crate::core_infra::log::python_float(cell.weight),
        spacing.map_or_else(String::new, |d| format!(
            ", {d} apart (-nd, Bohr of the variables per Angstrom)"
        )),
        crate::core_infra::log::python_float(cell.pressure_gpa)
    ));
    let mut settings = config.chain.clone();
    settings.kind = ChainKind::Variables;
    settings.align = false;
    settings.periodic_shifts = None;
    if config.per_atom {
        if map.dimension() % 3 != 0 {
            return Err(MoptError::unsupported(
                "-qsm (projection atom by atom) with -vc on a slab or a wire: the lattice \
                 variables of the cell are not blocks of three",
            ));
        }
        settings.block = Some(crate::path::band::BLOCK_ATOM);
        notes.push(
            "Projection block by block (-qsm, projection=atom): every atom, and every lattice \
             vector, is a block of three variables."
                .to_string(),
        );
    }
    if let Some(rfo) = settings.rfo.as_mut() {
        rfo.directory = hessian_directory;
        if !config.exact_hessians {
            notes.push(
                "Hessians of the images: of the enthalpy in the variables of the atoms and the \
                 cell (a model Hessian of -modelhess is one of the atoms alone)."
                    .to_string(),
            );
        }
    }
    let n = first.n_atoms();
    let mut fixed = Vec::new();
    for constraint in constraints {
        if let ConstraintKind::Frozen(atoms) = &constraint.kind {
            for atom in atoms {
                if atom.index() >= n {
                    return Err(MoptError::config(format!(
                        "-fix: atom {} is beyond the {n} atoms of the structures",
                        atom.index() + 1
                    )));
                }
                fixed.extend((0..3).map(|k| 3 * atom.index() + k));
            }
        }
    }
    fixed.sort_unstable();
    fixed.dedup();
    settings.fixed_components = fixed;
    let states = cell_states(&config.model_function, config.excited_state)?;
    let mut surface = CellChainSurface::new(
        calc,
        terms,
        map.clone(),
        first.clone(),
        config.excited_state,
        states,
    );
    let mut keep = None;
    if !constraints.is_empty() {
        let shared = Arc::new(Mutex::new(CellChainConstraints::new(
            constraints,
            &map,
            &chain,
        )?));
        let per_image = shared
            .lock()
            .map_err(|_| MoptError::calc("the constraints of the chain are not available"))?
            .per_image();
        notes.push(format!(
            "Constraints: {per_image} held in the variables of the atoms and the cell at every \
             image (the values of the image in the first chain)."
        ));
        surface = surface.with_constraints(Arc::clone(&shared));
        let mut observer = KeepConstraints::new(shared);
        chain = observer.restore_first(&chain)?;
        keep = Some(observer);
    }
    Ok(CellChain {
        settings,
        chain,
        surface,
        keep,
        notes,
    })
}

/// The cells of the images after every iteration: `cell_volume.csv` (Angstrom^3, one line per
/// iteration, one number per image) and their lattice parameters in the log.
pub struct CellTables {
    map: VariableCell,
    directory: PathBuf,
    log: crate::core_infra::log::Logger,
}

impl CellTables {
    /// The table in `directory` for points of `map` (an earlier one there starts anew).
    pub fn new(
        map: &VariableCell,
        directory: &Path,
        log: crate::core_infra::log::Logger,
    ) -> MoptResult<Self> {
        let path = directory.join(VOLUME_TABLE);
        if path.exists() {
            std::fs::remove_file(&path)?;
        }
        Ok(Self {
            map: map.clone(),
            directory: directory.to_path_buf(),
            log,
        })
    }
}

/// The file of the volumes of the images.
pub const VOLUME_TABLE: &str = "cell_volume.csv";

/// Lengths (Angstrom) and angles (degrees) of the lattice vectors of `sys`, and its volume
/// (Angstrom^3).
pub fn lattice_parameters(sys: &MolecularSystem) -> Option<([f64; 3], [f64; 3], f64)> {
    let cell = sys.active_cell()?;
    let v: [[f64; 3]; 3] = std::array::from_fn(|k| cell.lattice_vector(k).value());
    let dot = |a: &[f64; 3], b: &[f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let length: [f64; 3] = std::array::from_fn(|k| dot(&v[k], &v[k]).sqrt());
    let angle = |i: usize, j: usize| {
        (dot(&v[i], &v[j]) / (length[i] * length[j]))
            .clamp(-1.0, 1.0)
            .acos()
            .to_degrees()
    };
    let b3 = Units::BOHR_TO_ANGSTROM.powi(3);
    Some((
        length.map(|l| l * Units::BOHR_TO_ANGSTROM),
        [angle(1, 2), angle(0, 2), angle(0, 1)],
        cell.volume_bohr3() * b3,
    ))
}

impl ChainObserver for CellTables {
    fn after_iteration(
        &mut self,
        report: &ChainReport,
        _machine: &mut ChainRelaxation,
    ) -> MoptResult<Flow> {
        use std::fmt::Write as _;
        use std::io::Write as _;
        let mut volumes = Vec::with_capacity(report.chain.len());
        let mut table = String::from(
            "  image   a (Ang)    b (Ang)    c (Ang)   alpha     beta      gamma     volume (Ang^3)\n",
        );
        for (image, point) in report.chain.iter().enumerate() {
            let sys = self.map.system(point)?;
            let (l, a, volume) = lattice_parameters(&sys)
                .ok_or_else(|| MoptError::calc("an image of the variable cell has no cell"))?;
            volumes.push(volume);
            let _ = writeln!(
                table,
                "  {image:5} {:10.5} {:10.5} {:10.5} {:9.4} {:9.4} {:9.4} {volume:14.5}",
                l[0], l[1], l[2], a[0], a[1], a[2]
            );
        }
        self.log.info(table.trim_end_matches('\n'));
        let line: Vec<String> = volumes
            .iter()
            .map(|v| crate::core_infra::log::python_float(*v))
            .collect();
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.directory.join(VOLUME_TABLE))?;
        writeln!(file, "{}", line.join(","))?;
        Ok(Flow::Continue)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn structures_in_other_cells_are_joined_to_the_nearest_images() {
        let a = crate::io::xyz::parse_xyz_str(
            "2\nLattice=\"4.0 0.0 0.0 0.0 4.0 0.0 0.0 0.0 4.0\"\nAr 0.1 0.0 0.0\nAr 2.0 2.0 2.0\n",
            crate::domain::SpinCharge::default(),
            None,
        )
        .unwrap();
        // The first atom across the boundary, in a longer cell.
        let b = crate::io::xyz::parse_xyz_str(
            "2\nLattice=\"4.4 0.0 0.0 0.0 4.0 0.0 0.0 0.0 4.0\"\nAr 4.3 0.0 0.0\nAr 2.2 2.0 2.0\n",
            crate::domain::SpinCharge::default(),
            None,
        )
        .unwrap();
        let cell = VariableCellSettings {
            pressure_gpa: 0.0,
            weight: 1.0,
            forces: Vec::new(),
        };
        let map = cell.coordinates(&a).unwrap();
        let points = joined_points(&map, &[a.clone(), b]).unwrap();
        // The quasi-Cartesian x of the first atom: 4.3/4.4 - 1 of the first cell, near 0.1.
        let q = points[1][0] * Units::BOHR_TO_ANGSTROM;
        assert!((q - (4.3 / 4.4 - 1.0) * 4.0).abs() < 1e-9, "{q}");
        // The longer cell is in the lattice variables.
        let back = map.system(&points[1]).unwrap();
        let (l, _, _) = lattice_parameters(&back).unwrap();
        assert!((l[0] - 4.4).abs() < 1e-9, "{l:?}");
    }
}
