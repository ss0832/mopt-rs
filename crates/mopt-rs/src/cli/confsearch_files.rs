//! The files and the log of `confsearch`: MultiOptPy's files with their names
//! (`<stem>_EQ<k>.xyz`, `EQ_energy.dat`, `visit_counts.dat`, `switch_conformer.log`,
//! `no_update_count.log`, `search_atom_pairs.log`, `tabu_search_settings.log`,
//! `tabu_selection_log.log`, `REJECTED_CONFORMERS/`, `reference_bond_table.dat`,
//! `reference_bond_connectivity.log`, `EQ_summary.log`) and those of mopt-rs: the ensemble
//! sorted by energy with the energies (`conformers.xyz`; with the rotamers,
//! `rotamers.xyz`), its table (`conformers.csv`: energy, relative energy, Boltzmann weight,
//! RMSD to the lowest, rotational constants, how each was found), and the populations with
//! the conformational entropy (`ensemble.txt`).
//!
//! MultiOptPy: the file writing of `Entrypoints/conformation_search.py:run_confsearch`
//!  (`save_xyz_file`, `save_rejected_conformer`, `save_bond_table_file`, the summary).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::fmt::Write as _;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use crate::coord::graph::MolecularGraph;
use crate::core_infra::log::{python_float, Logger};
use crate::domain::{MolecularSystem, Quantity, Units};
use crate::error::MoptResult;
use crate::io::xyz::{format_xyz_system, write_xyz_file};
use crate::workflow::confsearch::{CandidateFate, SearchEvent, SearchObserver, SearchResult};
use crate::workflow::explore::{EqRecord, HeatedBoltzmann};

/// Folder of the minima rejected for their bonds.
pub const REJECTED_FOLDER: &str = "REJECTED_CONFORMERS";

/// What the conformers are ranked by, as `ensemble.txt` says it.
pub const RANKING: &str = "the energy with the bias potentials of the command (energy_hartree, \
                           the surface the conformers are minima of); bare_energy_hartree is \
                           the energy without them";

fn append(path: &Path, text: &str) -> MoptResult<()> {
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    f.write_all(text.as_bytes())?;
    Ok(())
}

fn pair_text(p: &[usize; 2]) -> String {
    format!("({}, {})", p[0] + 1, p[1] + 1)
}

/// The observer that writes the files of a run.
pub struct ConfsearchFiles {
    dir: PathBuf,
    stem: String,
    log: Logger,
    tabu: Option<HeatedBoltzmann>,
    rejected_dir: Option<PathBuf>,
    energies: Vec<f64>,
    visits: Vec<usize>,
    rejected: Vec<f64>,
    bond_scale: f64,
}

impl ConfsearchFiles {
    /// Files of a run in `dir` for the input `stem`; `tabu` is the selection of `-tabu` (for
    /// its logs), `conformers` whether minima with other bonds are rejected, `bond_scale` the
    /// scale of the covalent radii of the bonds.
    pub fn new(
        dir: &Path,
        stem: &str,
        log: Logger,
        tabu: Option<HeatedBoltzmann>,
        conformers: bool,
        bond_scale: f64,
    ) -> MoptResult<Self> {
        // A restart writes every file anew.
        for name in [
            "EQ_energy.dat",
            "visit_counts.dat",
            "switch_conformer.log",
            "no_update_count.log",
            "search_atom_pairs.log",
            "tabu_selection_log.log",
        ] {
            let _ = std::fs::remove_file(dir.join(name));
        }
        let rejected_dir = conformers.then(|| dir.join(REJECTED_FOLDER));
        if let Some(r) = &rejected_dir {
            let _ = std::fs::remove_dir_all(r);
            std::fs::create_dir_all(r)?;
        }
        if let Some(t) = &tabu {
            std::fs::write(
                dir.join("tabu_search_settings.log"),
                format!(
                    "Tabu Search / Metadynamics-like Frequency Penalty Settings\n{}\nAlpha \
                     (penalty coefficient): {}\nFormula: P_i^select ~ P_i^Boltzmann x exp(-alpha \
                     x N_i)\n  where N_i is the number of times conformer i has been selected.\n",
                    "=".repeat(60),
                    python_float(t.alpha)
                ),
            )?;
            log.info(&format!(
                "Tabu Search / Metadynamics-like frequency penalty is enabled (alpha = {}).",
                python_float(t.alpha)
            ));
        }
        Ok(Self {
            dir: dir.to_path_buf(),
            stem: stem.to_string(),
            log,
            tabu,
            rejected_dir,
            energies: Vec::new(),
            visits: Vec::new(),
            rejected: Vec::new(),
            bond_scale,
        })
    }

    /// The bond table of the optimized input (`-pbc`, the conformer mode).
    pub fn reference(&self, graph: &MolecularGraph) -> MoptResult<()> {
        let Some(rejected_dir) = &self.rejected_dir else {
            return Ok(());
        };
        let bonds = graph.bonds();
        let mut table = String::from(
            "# Reference Bond Connectivity Table (1-indexed)\n# Obtained from initial optimized \
             structure (EQ0)\n",
        );
        let mut human = format!(
            "Reference Bond Connectivity Table\n{}\nObtained from: Initial OPTIMIZED structure \
             (EQ0)\nNote: This is the structure after optimization of the input XYZ file.\n\nBond \
             List (1-indexed):\n",
            "=".repeat(50)
        );
        for [a, b] in &bonds {
            let _ = writeln!(table, "{} {}", a + 1, b + 1);
            let _ = writeln!(human, "  {} - {}", a + 1, b + 1);
        }
        let _ = writeln!(human, "\nTotal number of bonds: {}", bonds.len());
        std::fs::write(self.dir.join("reference_bond_table.dat"), table)?;
        std::fs::write(self.dir.join("reference_bond_connectivity.log"), human)?;
        std::fs::write(
            rejected_dir.join("README.txt"),
            "Rejected Conformers due to Bond Connectivity Change\n\
             =======================================================\n\n\
             This folder contains conformers that were excluded from the\n\
             main conformer search results because their bond connectivity\n\
             differs from the initial OPTIMIZED structure (EQ0).\n\n\
             Each rejected conformer has:\n  - XYZ file: *_REJECTEDn.xyz\n  - Info file: \
             *_REJECTEDn_info.txt (contains bond change details)\n",
        )?;
        self.log.info(&format!(
            "Bond connectivity preservation is enabled: {} bonds of the optimized input (EQ0).",
            bonds.len()
        ));
        Ok(())
    }

    fn eq_path(&self, k: usize) -> PathBuf {
        self.dir.join(format!("{}_EQ{k}.xyz", self.stem))
    }

    fn new_conformer(
        &mut self,
        k: usize,
        energy: f64,
        system: Option<&MolecularSystem>,
    ) -> MoptResult<()> {
        if let Some(sys) = system {
            let mut s = sys.clone();
            s.comment = format!("Sample_EQ{k}");
            write_xyz_file(self.eq_path(k), &s, None)?;
            if k == 0 {
                self.reference(&MolecularGraph::from_system(sys, self.bond_scale))?;
            }
        }
        if self.energies.len() <= k {
            self.energies.resize(k + 1, f64::NAN);
            self.visits.resize(k + 1, 0);
        }
        self.energies[k] = energy;
        let text: String = self
            .energies
            .iter()
            .map(|e| format!("{}\n", python_float(*e)))
            .collect();
        std::fs::write(self.dir.join("EQ_energy.dat"), text)?;
        self.write_visits()?;
        if k == 0 {
            self.log.info("initial conformer.");
        } else {
            self.log.info("Find new conformer.");
        }
        self.log.info(&format!("Energy:  {}", python_float(energy)));
        Ok(())
    }

    fn write_visits(&self) -> MoptResult<()> {
        if self.tabu.is_some() {
            let text: String = self.visits.iter().map(|v| format!("{v}\n")).collect();
            std::fs::write(self.dir.join("visit_counts.dat"), text)?;
        }
        Ok(())
    }

    fn rejected_conformer(
        &mut self,
        origin: &str,
        system: &MolecularSystem,
        energy: f64,
        changes: &crate::coord::graph::BondChanges,
    ) -> MoptResult<()> {
        let n = self.rejected.len();
        self.rejected.push(energy);
        if !changes.formed.is_empty() {
            self.log.info(&format!(
                "New bonds formed: {}",
                changes
                    .formed
                    .iter()
                    .map(pair_text)
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        if !changes.broken.is_empty() {
            self.log.info(&format!(
                "Bonds broken: {}",
                changes
                    .broken
                    .iter()
                    .map(pair_text)
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        self.log
            .info("Bond connectivity is NOT preserved. This conformer will be excluded.");
        let Some(dir) = &self.rejected_dir else {
            return Ok(());
        };
        let mut s = system.clone();
        s.comment = format!("Rejected ({origin}) Energy {energy:.8}");
        let name = format!("{}_REJECTED{n}", self.stem);
        write_xyz_file(dir.join(format!("{name}.xyz")), &s, None)?;
        let mut info = format!(
            "Rejected Conformer Information\n{}\nOrigin: {origin}\nEnergy: {}\nXYZ file: \
             {name}.xyz\n\nBond Connectivity Changes:\n",
            "=".repeat(50),
            python_float(energy)
        );
        if !changes.formed.is_empty() {
            info.push_str("  New bonds formed:\n");
            for [a, b] in &changes.formed {
                let _ = writeln!(info, "    {} - {}", a + 1, b + 1);
            }
        }
        if !changes.broken.is_empty() {
            info.push_str("  Bonds broken:\n");
            for [a, b] in &changes.broken {
                let _ = writeln!(info, "    {} - {}", a + 1, b + 1);
            }
        }
        std::fs::write(dir.join(format!("{name}_info.txt")), info)?;
        append(
            &dir.join("rejected_energy.dat"),
            &format!("{}\n", python_float(energy)),
        )?;
        std::fs::write(dir.join("rejected_count.dat"), format!("{}", n + 1))?;
        append(
            &dir.join("rejected_summary.log"),
            &format!("REJECTED{n}: {origin}, Energy {energy:.8}\n"),
        )?;
        self.log.info(&format!(
            "Rejected conformer saved to: {}",
            dir.join(format!("{name}.xyz")).display()
        ));
        Ok(())
    }
}

impl SearchObserver for ConfsearchFiles {
    fn on_event(&mut self, event: &SearchEvent<'_>) -> MoptResult<()> {
        match event {
            SearchEvent::Note(text) => self.log.info(text),
            SearchEvent::Sample { trial, from, what } => {
                self.log.info(&format!("Sampling conformation:  {trial}"));
                self.log.info(&format!("  from EQ{from}: {what}"));
            }
            SearchEvent::Candidate {
                origin,
                fate,
                system,
            } => match fate {
                CandidateFate::New { index, energy } => {
                    self.new_conformer(*index, *energy, *system)?
                }
                CandidateFate::Duplicate { of } => self.log.info(&format!(
                    "{origin}: the same structure as EQ{of}. Skip this conformer."
                )),
                CandidateFate::Rotamer { of } => self.log.info(&format!(
                    "{origin}: a rotamer of EQ{of} (the same conformer with permuted atoms)."
                )),
                CandidateFate::Rejected {
                    system,
                    energy,
                    changes,
                } => self.rejected_conformer(origin, system, *energy, changes)?,
                CandidateFate::NotConverged => {
                    self.log
                        .info(&format!("{origin}: Optimization is failed..."));
                }
                CandidateFate::Dissociated => {
                    self.log.info(&format!("{origin}: DC is detected..."))
                }
                CandidateFate::Failed(why) => {
                    self.log
                        .warn(&format!("{origin}: the calculation failed: {why}"));
                }
                CandidateFate::OffConstraints(why) => {
                    self.log.info(&format!("{origin}: discarded: {why}"));
                }
                CandidateFate::OutsideWindow(de) => self.log.debug(&format!(
                    "{origin}: {:.2} kcal/mol above the lowest, outside the window",
                    de * Units::HARTREE_TO_KCAL_MOL
                )),
            },
            SearchEvent::Switch {
                trial, to, visits, ..
            } => {
                if self.visits.len() <= *to {
                    self.visits.resize(to + 1, 0);
                }
                self.visits[*to] = *visits;
                self.log.info(&format!("Switch conformer: EQ{to}"));
                let line = match &self.tabu {
                    Some(_) => {
                        format!("Trial {trial}: Switch conformer: EQ{to} (visit_count={visits})\n")
                    }
                    None => format!("Trial {trial}: Switch conformer: EQ{to}\n"),
                };
                append(&self.dir.join("switch_conformer.log"), &line)?;
                self.write_visits()?;
                if let Some(t) = &mut self.tabu {
                    if trial % 10 == 0 {
                        let pool: Vec<EqRecord> = self
                            .energies
                            .iter()
                            .zip(&self.visits)
                            .map(|(e, v)| EqRecord {
                                energy: Quantity::new(*e),
                                visits: *v,
                            })
                            .collect();
                        crate::workflow::explore::SamplingPriority::on_trial(t, *trial);
                        let p = t.probabilities(&pool);
                        append(
                            &self.dir.join("tabu_selection_log.log"),
                            &format!(
                                "\nTrial {trial}: Selection probabilities (with frequency penalty)\n  \
                                 Visit counts: {:?}\n  Probabilities: {:?}\n  Selected: EQ{to}\n",
                                self.visits,
                                p.iter().map(|x| format!("{x:.4}")).collect::<Vec<_>>()
                            ),
                        )?;
                    }
                }
            }
            SearchEvent::Counter { trial, count } => {
                append(
                    &self.dir.join("no_update_count.log"),
                    &format!("{trial} {count}\n"),
                )?;
            }
            SearchEvent::Pairs(pairs) => {
                let text: String = pairs
                    .iter()
                    .map(|(p, s)| {
                        format!(
                            "[{}, {}] {}\n",
                            p[0],
                            p[1],
                            if *s > 0.0 { "p" } else { "m" }
                        )
                    })
                    .collect();
                std::fs::write(self.dir.join("search_atom_pairs.log"), text)?;
            }
        }
        Ok(())
    }
}

/// The files at the end of a search: `EQ_summary.log`, `conformers.xyz`, `rotamers.xyz`,
/// `conformers.csv`, and `ensemble.txt`; `imaginary` has the imaginary frequencies of the
/// members when `-freq` analysed them.
pub fn write_summary(
    dir: &Path,
    result: &SearchResult,
    temperature: f64,
    imaginary: &[Option<usize>],
    log: &Logger,
) -> MoptResult<()> {
    let e = &result.ensemble;
    let members = e.members();
    let (Some(low), Some(high)) = (
        e.lowest(),
        (0..members.len()).max_by(|&a, &b| members[a].energy.total_cmp(&members[b].energy)),
    ) else {
        return Ok(());
    };
    let mut summary = format!("Summary\nReason of Termination: {}\n", result.reason);
    let _ = writeln!(
        summary,
        "conformer of lowest energy: {}",
        python_float(members[low].energy)
    );
    let _ = writeln!(summary, "structure of lowest energy: EQ{low}");
    let _ = writeln!(
        summary,
        "conformer of highest energy: {}",
        python_float(members[high].energy)
    );
    let _ = writeln!(summary, "structure of highest energy: EQ{high}");
    if !result.rejected.is_empty() {
        let _ = writeln!(
            summary,
            "\nBond Connectivity Preservation Results:\n  Reference structure: EQ0 (initial \
             optimized structure)\n  Conformers rejected: {}\n  Rejected conformers saved in: \
             {REJECTED_FOLDER}/",
            result.rejected.len()
        );
    }
    std::fs::write(dir.join("EQ_summary.log"), &summary)?;
    log.info(&format!(
        "conformer of lowest energy:  {}",
        python_float(members[low].energy)
    ));
    log.info(&format!("structure of lowest energy:  EQ{low}"));
    log.info(&format!(
        "conformer of highest energy:  {}",
        python_float(members[high].energy)
    ));
    log.info(&format!("structure of highest energy:  EQ{high}"));
    let thermo = e.thermo(temperature)?;
    let order = e.by_energy();
    let e0 = members[low].energy;
    let mut xyz = String::new();
    let mut rotamers = String::new();
    let mut csv = String::from(
        "rank,eq,energy_hartree,relative_kcal_mol,boltzmann_weight,degeneracy,\
         rmsd_to_lowest_angstrom,rotational_constant_a_ghz,rotational_constant_b_ghz,\
         rotational_constant_c_ghz,rotamers,same_bonds_as_eq0,imaginary_frequencies,\
         bare_energy_hartree,origin\n",
    );
    for (rank, &i) in order.iter().enumerate() {
        let m = &members[i];
        let rel = (m.energy - e0) * Units::HARTREE_TO_KCAL_MOL;
        let mut s = m.system.clone();
        s.comment = format!(
            "EQ{i} energy={:.12} relative={rel:.4} kcal/mol weight={:.6} bare_energy={:.12}",
            m.energy,
            thermo.populations[i],
            result.bare_energy(i)
        );
        xyz.push_str(&format_xyz_system(&s, None));
        rotamers.push_str(&format_xyz_system(&s, None));
        for (k, r) in m.rotamers.iter().enumerate() {
            let mut r = r.clone();
            r.comment = format!("EQ{i} rotamer {} energy={:.12}", k + 1, m.energy);
            rotamers.push_str(&format_xyz_system(&r, None));
        }
        let rmsd = e.rmsd_between(i, low)? * Units::BOHR_TO_ANGSTROM;
        let _ = writeln!(
            csv,
            "{},{i},{:.10},{rel:.4},{:.6},{},{rmsd:.4},{:.5},{:.5},{:.5},{},{},{},{:.10},\"{}\"",
            rank + 1,
            m.energy,
            thermo.populations[i],
            m.degeneracy(e.settings().mirror),
            m.rotational[0],
            m.rotational[1],
            m.rotational[2],
            m.rotamers.len(),
            m.graph.is_isomorphic(&result.reference),
            imaginary
                .get(i)
                .copied()
                .flatten()
                .map_or(String::new(), |n| n.to_string()),
            result.bare_energy(i),
            m.origin.replace('"', "'")
        );
    }
    std::fs::write(dir.join("conformers.xyz"), xyz)?;
    std::fs::write(dir.join("rotamers.xyz"), rotamers)?;
    std::fs::write(dir.join("conformers.csv"), csv)?;
    let kcal = Units::HARTREE_TO_KCAL_MOL;
    let mut text = format!(
        "Conformer ensemble at {temperature:.2} K\n\nconformers: {}\nrotamers: {}\nrejected for \
         their bonds: {}\nranked by: {RANKING}\n\n",
        members.len(),
        members.iter().map(|m| m.rotamers.len()).sum::<usize>(),
        result.rejected.len()
    );
    let _ = writeln!(
        text,
        "{:>5} {:>6} {:>12} {:>10} {:>4}",
        "rank", "EQ", "dE kcal/mol", "weight", "g"
    );
    for (rank, &i) in order.iter().enumerate() {
        let m = &members[i];
        let _ = writeln!(
            text,
            "{:>5} {:>6} {:>12.4} {:>10.6} {:>4}",
            rank + 1,
            format!("EQ{i}"),
            (m.energy - e0) * kcal,
            thermo.populations[i],
            m.degeneracy(e.settings().mirror)
        );
    }
    let _ = writeln!(
        text,
        "\nensemble average energy (kcal/mol): {:.4}\nensemble entropy (J/mol K, cal/mol K): \
         {:.3} {:.3}\nensemble free energy -kT ln Z (kcal/mol): {:.4}\npopulation of the lowest \
         conformer: {:.2} %",
        thermo.average_energy * kcal,
        thermo.entropy,
        thermo.entropy / Units::CALORIE_J,
        thermo.free_energy * kcal,
        100.0 * thermo.populations[low]
    );
    std::fs::write(dir.join("ensemble.txt"), &text)?;
    for line in text.lines() {
        log.info(line);
    }
    log.info(&format!(
        "Conformer ensemble file path: {}",
        dir.join("conformers.xyz").display()
    ));
    Ok(())
}
