//! What `orientsearch` writes: the log lines of the run and of every sample, the minima
//! (`<stem>_minimum_<rank>.xyz`, `<stem>_minima.xyz`) with the energies of their parts alone
//! ([`super::orientsearch_energies`]), and the tables `orientsearch_minima.csv` and
//! `orientsearch_samples.csv`.
//!
//! MultiOptPy: no counterpart (new in mopt-rs); MultiOptPy leaves the runs of its samples as
//!  they are.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use super::optmain::JobSummary;
use super::orientsearch::OrientsearchConfig;
use super::orientsearch_energies::PartEnergies;
use crate::core_infra::log::Logger;
use crate::domain::{MolecularSystem, Units};
use crate::error::MoptResult;
use crate::io::xyz::format_xyz_system;
use crate::pes::counting::EvaluationCounts;
use crate::workflow::explore::identity::BondGraph;
use crate::workflow::explore::orient::{
    ComplexParts, OrientationSearch, SampleOrigin, SampleRecord,
};
use crate::workflow::explore::pool::boltzmann_weights;

/// One sample as it ran.
#[derive(Debug, Clone)]
pub struct SampleRun {
    /// Its number.
    pub index: usize,
    /// Where its start came from.
    pub origin: SampleOrigin,
    /// `<stem>_Sample_<k>.xyz`.
    pub start: PathBuf,
    /// Its `optmain` run directory.
    pub directory: PathBuf,
    /// The evaluations it made.
    pub evaluations: EvaluationCounts,
    /// Whether it converged.
    pub converged: bool,
    /// Iterations of its optimization.
    pub iterations: Option<usize>,
    /// Its energy and the energy with bias potentials (Hartree).
    pub energies: Option<(f64, f64)>,
    /// Imaginary frequencies of its end (`-freq`).
    pub imaginary: Option<usize>,
    /// Its optimized structure.
    pub optimized: Option<PathBuf>,
}

impl SampleRun {
    /// A sample before its result is known.
    pub fn new(
        index: usize,
        origin: SampleOrigin,
        start: PathBuf,
        directory: PathBuf,
        evaluations: EvaluationCounts,
    ) -> Self {
        Self {
            index,
            origin,
            start,
            directory,
            evaluations,
            converged: false,
            iterations: None,
            energies: None,
            imaginary: None,
            optimized: None,
        }
    }

    /// The end of its optimization.
    pub fn finished(&mut self, summary: &JobSummary, converged: bool, optimized: PathBuf) {
        self.converged = converged;
        self.iterations = Some(summary.iterations);
        self.energies = Some((summary.energy.value(), summary.biased_energy.value()));
        self.imaginary = summary.imaginary_modes;
        self.optimized = Some(optimized);
    }
}

fn origin_text(origin: &SampleOrigin, scheme: &str, search: &OrientationSearch) -> String {
    match origin {
        SampleOrigin::Random { redraws: 0 } => format!("random ({scheme})"),
        SampleOrigin::Random { redraws } => format!("random ({scheme}, {redraws} redrawn)"),
        SampleOrigin::Perturbed { minimum } => {
            let first = search
                .pool()
                .minima()
                .get(*minimum)
                .map_or(0, |m| m.first_trial);
            format!("perturbed minimum of sample {first}")
        }
    }
}

/// The lines at the start of a run.
pub fn log_header(
    config: &OrientsearchConfig,
    template: &MolecularSystem,
    parts: &ComplexParts,
    search: &OrientationSearch,
    dir: &Path,
    log: &Logger,
) {
    let s = search.settings();
    log.info(&format!(
        "mopt-rs {} orientsearch: {}",
        crate::VERSION,
        config.input.display()
    ));
    log.info(&format!("Run directory: {}", dir.display()));
    log.info(&format!(
        "Number of Parts: {} ({})",
        parts.len(),
        match parts.source {
            crate::workflow::explore::orient::PartSource::Lists => "-part",
            crate::workflow::explore::orient::PartSource::Molecules =>
                "the molecules of the bond graph; -part gives others",
        }
    ));
    for line in parts.describe(template) {
        log.info(&line);
    }
    let p = &s.placement;
    let placement = if !super::orientsearch::uses_distance(p.scheme) {
        format!(
            "contact ratio {:.2} to {:.2} of the sums of van der Waals radii",
            p.contact[0], p.contact[1]
        )
    } else {
        format!(
            "centres {:.3} to {:.3} Angstrom apart (-dist), clash ratio {:.2}",
            p.distance.value() * Units::BOHR_TO_ANGSTROM,
            1.1 * p.distance.value() * Units::BOHR_TO_ANGSTROM,
            p.clash
        )
    };
    log.info(&format!(
        "Sampling: {} samples, scheme {} ({placement}), seed {}{}",
        s.samples,
        p.scheme.name(),
        s.seed,
        if config.seed_given {
            ""
        } else {
            " (drawn; in settings.json)"
        }
    ));
    if config.distance_given && !super::orientsearch::uses_distance(p.scheme) {
        log.info("-dist is the distance of the multioptpy and uniform schemes; the ring and contact schemes place the parts at van der Waals contact");
    }
    if let Some(slab) = search.sampler().slab() {
        log.info(&format!(
            "Periodic cell: the parts that stay form a slab; the others come down onto its free side (lattice axis {}, empty layer {:.1} Angstrom)",
            ["a", "b", "c"][slab.axis],
            slab.gap * Units::BOHR_TO_ANGSTROM
        ));
    } else if template.has_pbc() {
        log.info("Periodic cell: the parts are put at random free points of the cell");
    }
    if s.refine > 0.0 {
        log.info(&format!(
            "Refinement: {:.0}% of the samples after the first minimum start from a known minimum chosen by '{}', one part turned by up to {:.0} degrees and shifted by up to {:.2} Angstrom",
            100.0 * s.refine,
            s.priority,
            s.perturbation.max_angle.to_degrees(),
            s.perturbation.max_shift * Units::BOHR_TO_ANGSTROM
        ));
    }
    let m = &s.matcher;
    log.info(&format!(
        "Same minimum: same bonds (scale {:.2}), RMSD up to {:.3} Angstrom after matching equivalent atoms and identical molecules, energies within {:.1e} Hartree{}",
        m.bond_scale,
        m.rmsd_tolerance.value() * Units::BOHR_TO_ANGSTROM,
        m.energy_tolerance.value(),
        if m.mirror_images { ", mirror images included" } else { "" }
    ));
    let criteria = if config.parsed.get_bool("tight_convergence_criteria", false) {
        "the tight criteria (-tcc)"
    } else if config.parsed.get_bool("loose_convergence_criteria", false) {
        "the loose criteria (-lcc)"
    } else {
        "the default criteria of optmain"
    };
    log.info(&format!(
        "Every sample is optimized by optmain ({}) with {criteria} in <stem>_Sample_<k>_OPT/; create {}/STOP to end the search after the sample in progress",
        config.optmain.calculator,
        dir.display()
    ));
}

/// The progress line of one sample.
pub fn log_sample(
    search: &OrientationSearch,
    record: &SampleRecord,
    run: &SampleRun,
    log: &Logger,
) {
    let scheme = search.settings().placement.scheme.name();
    let mut line = format!(
        "Sample {}/{} ({}): ",
        record.index + 1,
        search.settings().samples,
        origin_text(&record.origin, scheme, search)
    );
    match (record.energy, run.iterations) {
        (Some(e), Some(it)) => {
            let _ = write!(
                line,
                "{} after {it} iterations, E = {:.10} Hartree",
                if record.converged {
                    "converged"
                } else {
                    "not converged"
                },
                e.value()
            );
        }
        _ => line.push_str("failed"),
    }
    if let Some(ins) = &record.insertion {
        let rank = search
            .pool()
            .ranking()
            .iter()
            .position(|&k| k == ins.index)
            .map_or(0, |r| r + 1);
        if ins.new {
            let _ = write!(line, "; new minimum (now rank {rank})");
        } else {
            let _ = write!(
                line,
                "; the minimum of sample {} (rank {rank}, RMSD {:.3} Angstrom){}",
                search.pool().minima()[ins.index].first_trial,
                ins.rmsd
                    .map_or(0.0, |r| r.value() * Units::BOHR_TO_ANGSTROM),
                if ins.replaced { ", lower: kept" } else { "" }
            );
        }
    } else if record.energy.is_some() {
        line.push_str("; not counted as a minimum");
    }
    let _ = write!(
        line,
        "; {} distinct; {} gradients, {} Hessians",
        search.pool().len(),
        run.evaluations.gradients,
        run.evaluations.hessians
    );
    log.info(&line);
}

/// One distinct minimum as reported.
#[derive(Debug, Clone)]
pub struct MinimumReport {
    /// Rank by energy (from 1).
    pub rank: usize,
    /// `<stem>_minimum_<rank>.xyz`.
    pub file: PathBuf,
    /// Energy on the surface the samples were minimized on, with the bias potentials (Hartree):
    /// the energy the minima are ranked by and weighted with.
    pub energy: f64,
    /// Energy without the bias potentials (Hartree).
    pub bare_energy: f64,
    /// Energy above the lowest minimum (kcal/mol, of [`Self::energy`]).
    pub relative_kcal_mol: f64,
    /// Boltzmann weight at `-temp` (minima only, `-order 0`).
    pub weight: Option<f64>,
    /// Bare energy minus the parts alone at their geometries in it (kcal/mol).
    pub interaction_kcal_mol: Option<f64>,
    /// Bare energy minus the parts optimized alone (kcal/mol).
    pub binding_kcal_mol: Option<f64>,
    /// The sample that found it first.
    pub first_sample: usize,
    /// The sample whose structure it is.
    pub sample: usize,
    /// How many samples ended in it.
    pub hits: usize,
    /// How its first sample started.
    pub origin: String,
    /// Imaginary frequencies (`-freq`).
    pub imaginary: Option<usize>,
    /// The closest contact of two parts: the smallest ratio of a distance to the sum of the
    /// van der Waals radii of the two atoms.
    pub contact: f64,
    /// Whether its bonds differ from those of the input.
    pub bonds_changed: bool,
    /// The structure.
    pub system: MolecularSystem,
}

/// Minima whose parts are nowhere closer than this ratio of the sums of their van der Waals
/// radii are marked "parts apart" in the log: the parts did not find each other (hydrogen
/// bonds are at 0.7 to 0.8 of the sums, stacked rings at about 1.0; a choice, not a
/// measurement).
pub const APART_RATIO: f64 = 1.5;

/// A field of a CSV table, quoted when it holds a comma, a quote, or a line break.
fn field(text: &str) -> String {
    if text.contains([',', '"', '\n']) {
        format!("\"{}\"", text.replace('"', "\"\""))
    } else {
        text.to_string()
    }
}

fn opt<T: std::fmt::Display>(v: Option<T>) -> String {
    v.map_or_else(String::new, |x| x.to_string())
}

/// Write the minima and the tables; return the minima, lowest first.
#[allow(clippy::too_many_arguments)]
pub fn write_results(
    config: &OrientsearchConfig,
    template: &MolecularSystem,
    search: &OrientationSearch,
    runs: &[SampleRun],
    energies: &PartEnergies,
    stem: &str,
    dir: &Path,
    log: &Logger,
) -> MoptResult<Vec<MinimumReport>> {
    let pool = search.pool();
    let order = pool.ranking();
    let minimum_order = config.optmain.settings.saddle_order == 0;
    let imaginary_of = |k: usize| {
        runs.iter()
            .find(|s| s.index == pool.minima()[k].trial)
            .and_then(|s| s.imaginary)
    };
    // Boltzmann weights of the minima; structures that `-freq` found to have imaginary
    // frequencies are saddle points and get none.
    let weights: Option<Vec<Option<f64>>> = if minimum_order {
        let counted: Vec<usize> = (0..pool.len())
            .filter(|&k| imaginary_of(k).is_none_or(|n| n == 0))
            .collect();
        let energies_counted: Vec<f64> = counted
            .iter()
            .map(|&k| pool.minima()[k].energy.value())
            .collect();
        let w = boltzmann_weights(&energies_counted, config.temperature)?;
        let mut all = vec![None; pool.len()];
        for (j, &k) in counted.iter().enumerate() {
            all[k] = Some(w[j]);
        }
        Some(all)
    } else {
        None
    };
    let reference = BondGraph::of(template, pool.matcher.bond_scale);
    let lowest = order
        .first()
        .map_or(0.0, |&k| pool.minima()[k].energy.value());
    let scheme = search.settings().placement.scheme.name();
    let kcal = Units::HARTREE_TO_KCAL_MOL;
    let mut reports = Vec::with_capacity(order.len());
    let mut all = String::new();
    for (r, &k) in order.iter().enumerate() {
        let m = &pool.minima()[k];
        let rank = r + 1;
        let energy = m.energy.value();
        let bare = search.bare_energy(k).value();
        let origin = runs
            .iter()
            .find(|s| s.index == m.first_trial)
            .map_or_else(String::new, |s| origin_text(&s.origin, scheme, search));
        let report = MinimumReport {
            rank,
            file: dir.join(format!("{stem}_minimum_{rank}.xyz")),
            energy,
            bare_energy: bare,
            relative_kcal_mol: (energy - lowest) * kcal,
            weight: weights.as_ref().and_then(|w| w[k]),
            // The parts alone are calculated without the bias potentials: so is the complex.
            interaction_kcal_mol: energies
                .vertical
                .get(k)
                .copied()
                .flatten()
                .map(|e| (bare - e) * kcal),
            binding_kcal_mol: energies.relaxed.map(|e| (bare - e) * kcal),
            first_sample: m.first_trial,
            sample: m.trial,
            hits: m.hits,
            origin,
            imaginary: imaginary_of(k),
            contact: search.sampler().closest_contact(&m.system),
            bonds_changed: !reference.same_as(&m.graph),
            system: m.system.clone(),
        };
        let mut sys = m.system.clone();
        sys.comment = format!(
            "Minimum {rank}: E = {energy:.10} Hartree, {:.4} kcal/mol above the lowest, sample {}, \
             bare E = {bare:.10} Hartree",
            report.relative_kcal_mol, m.trial
        );
        let text = format_xyz_system(&sys, None);
        std::fs::write(&report.file, &text)?;
        all.push_str(&text);
        reports.push(report);
    }
    std::fs::write(dir.join(format!("{stem}_minima.xyz")), all)?;

    let mut csv = String::from(
        "rank,energy_hartree,relative_kcal_mol,boltzmann_weight,interaction_kcal_mol,binding_kcal_mol,hits,first_sample,sample,origin,imaginary_modes,contact_ratio,bonds,file,bare_energy_hartree\n",
    );
    for m in &reports {
        let _ = writeln!(
            csv,
            "{},{:.10},{:.6},{},{},{},{},{},{},{},{},{:.3},{},{},{:.10}",
            m.rank,
            m.energy,
            m.relative_kcal_mol,
            opt(m.weight.map(|w| format!("{w:.6}"))),
            opt(m.interaction_kcal_mol.map(|e| format!("{e:.6}"))),
            opt(m.binding_kcal_mol.map(|e| format!("{e:.6}"))),
            m.hits,
            m.first_sample,
            m.sample,
            field(&m.origin),
            opt(m.imaginary),
            m.contact,
            if m.bonds_changed { "changed" } else { "same" },
            field(
                &m.file
                    .file_name()
                    .map(|f| f.to_string_lossy().into_owned())
                    .unwrap_or_default()
            ),
            m.bare_energy
        );
    }
    std::fs::write(dir.join("orientsearch_minima.csv"), csv)?;

    let mut samples = String::from(
        "sample,origin,converged,iterations,energy_hartree,biased_energy_hartree,minimum_rank,new_minimum,gradients,hessians,start,directory\n",
    );
    for run in runs {
        let record = search.records().iter().find(|r| r.index == run.index);
        let rank = record
            .and_then(|r| r.insertion)
            .and_then(|ins| order.iter().position(|&k| k == ins.index))
            .map(|p| p + 1);
        let name = |p: &Path| {
            p.file_name()
                .map(|f| f.to_string_lossy().into_owned())
                .unwrap_or_default()
        };
        let _ = writeln!(
            samples,
            "{},{},{},{},{},{},{},{},{},{},{},{}",
            run.index,
            field(&origin_text(&run.origin, scheme, search)),
            run.converged,
            opt(run.iterations),
            opt(run.energies.map(|e| format!("{:.10}", e.0))),
            opt(run.energies.map(|e| format!("{:.10}", e.1))),
            opt(rank),
            record.and_then(|r| r.insertion).is_some_and(|i| i.new),
            run.evaluations.gradients,
            run.evaluations.hessians,
            field(&name(&run.start)),
            field(&name(&run.directory))
        );
    }
    std::fs::write(dir.join("orientsearch_samples.csv"), samples)?;

    log.info(&format!(
        "Distinct minima ({} of {} samples converged{}):",
        search.records().iter().filter(|r| r.converged).count(),
        search.records().len(),
        if minimum_order {
            format!("; Boltzmann weights at {} K", config.temperature)
        } else {
            "; saddle points: no Boltzmann weights".to_string()
        }
    ));
    log.info(&format!(
        "{:>4} {:>18} {:>18} {:>10} {:>8} {:>12} {:>5}  {}",
        "rank",
        "E (Hartree)",
        "bare E (Hartree)",
        "dE (kcal)",
        "weight",
        "E_int (kcal)",
        "hits",
        "first found by"
    ));
    for m in &reports {
        log.info(&format!(
            "{:>4} {:>18.10} {:>18.10} {:>10.4} {:>8} {:>12} {:>5}  sample {} ({}){}{}{}",
            m.rank,
            m.energy,
            m.bare_energy,
            m.relative_kcal_mol,
            m.weight
                .map_or_else(|| "-".to_string(), |w| format!("{w:.4}")),
            m.interaction_kcal_mol
                .map_or_else(|| "-".to_string(), |e| format!("{e:.4}")),
            m.hits,
            m.first_sample,
            m.origin,
            m.imaginary
                .filter(|n| *n > 0)
                .map_or_else(String::new, |n| format!(", saddle point ({n} imaginary)")),
            if m.bonds_changed {
                ", bonds changed"
            } else {
                ""
            },
            if m.contact > APART_RATIO {
                ", parts apart"
            } else {
                ""
            }
        ));
    }
    if let Some(first) = reports.first() {
        if let Some(b) = first.binding_kcal_mol {
            log.info(&format!(
                "Binding energy of the lowest minimum against the parts optimized alone: {b:.4} kcal/mol"
            ));
        }
        log.info(&format!("Lowest minimum: {}", first.file.display()));
    }
    log.info(&format!(
        "Files: {stem}_minimum_<rank>.xyz, {stem}_minima.xyz, orientsearch_minima.csv, orientsearch_samples.csv"
    ));
    Ok(reports)
}
