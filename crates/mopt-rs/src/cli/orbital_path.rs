//! `freqmain PATH.xyz -nro` and `-roet`: analyses of the molecular orbitals along the
//! frames of a path, written into the run directory.
//!
//! `-nro`: at every frame the natural reaction orbitals for the displacement along the
//! path (`nro.csv`, `nro_pairs.txt`, `nro_orbitals_frame<k>_<spin>.csv` for the frame where
//! the electrons respond most, `nro_sum_of_squares.png` / `.svg`). `-roet`: the energies of
//! the orbitals followed along the path, their derivatives, and the orbitals that change
//! most (`orbital_energies.csv`, `1st_derivative_orbital_energy_plot.csv`,
//! `2nd_derivative_orbital_energy_plot.csv`, `reactive_orbitals.txt`,
//! `orbital_energies.png` / `.svg`). Either can be asked for alone.
//!
//! MultiOptPy: `WFAnalyzer/MO_analysis.py:NROAnalysis` (`run`, `save_results`), which no
//! option of MultiOptPy reaches.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::fmt::Write as _;
use std::path::Path;

use crate::analysis::orbitals::roet::{derivatives, reactive_orbitals};
use crate::core_infra::log::Logger;
use crate::core_infra::viz::{PlotFigure, PlotSeries, PlotStyle, Rgb};
use crate::domain::MolecularSystem;
use crate::error::{MoptError, MoptResult};
use crate::pes::calculator::Calculator;
use crate::workflow::orbital_analysis::{
    arc_lengths, natural_reaction_orbitals_along, orbital_energies_along, FollowedEnergies,
    ReactionOrbitalSet,
};

/// What `-nro` and `-roet` ask for.
#[derive(Debug, Clone, PartialEq)]
pub struct OrbitalPathSettings {
    /// `-nro`: natural reaction orbitals along the path.
    pub nro: bool,
    /// `-roet`: orbital energies along the path and the reactive orbitals.
    pub roet: bool,
    /// `-orbital_step`: displacement for the differences of the orbitals (amu^(1/2) Bohr).
    pub step: f64,
    /// `-diabatize N M ...`: the excited states that are made diabatic along the path
    /// ([`super::diabatic_path`]); empty: none.
    pub diabatic: Vec<usize>,
}

impl OrbitalPathSettings {
    /// Whether any analysis along the frames is asked for.
    pub fn any(&self) -> bool {
        self.nro || self.roet || !self.diabatic.is_empty()
    }
}

/// What the analyses found, for callers that go on with it.
#[derive(Debug, Clone)]
pub struct OrbitalPath {
    /// Positions of the frames along the path (amu^(1/2) Bohr).
    pub arc_lengths: Vec<f64>,
    /// `-nro`: for each frame, the natural reaction orbitals of each spin.
    pub reaction_orbitals: Option<Vec<Vec<ReactionOrbitalSet>>>,
    /// `-roet`: the followed orbital energies of each spin.
    pub energies: Option<Vec<FollowedEnergies>>,
    /// `-diabatize`: the states as diabatic states.
    pub diabatic: Option<super::diabatic_path::DiabaticPath>,
}

const COLORS: [Rgb; 4] = [Rgb::BLUE, Rgb::RED, Rgb::EMERALD, Rgb::AMBER];

fn file_name(spin: &str) -> String {
    spin.replace(' ', "_")
}

/// The leading parts of column `pair` of `weights` (an orbital of a pair in the orbitals of
/// the calculation): `index (weight)` for those of more than 5 %, largest first.
fn composition(weights: &faer::Mat<f64>, pair: usize, indices: &[usize]) -> String {
    let mut parts: Vec<(usize, f64)> = indices
        .iter()
        .enumerate()
        .map(|(row, &orbital)| (orbital + 1, weights[(row, pair)].powi(2)))
        .filter(|(_, w)| *w > 0.05)
        .collect();
    parts.sort_by(|a, b| b.1.total_cmp(&a.1));
    if parts.is_empty() {
        return "-".to_string();
    }
    parts
        .iter()
        .map(|(orbital, w)| format!("{orbital} ({:.0} %)", 100.0 * w))
        .collect::<Vec<_>>()
        .join(", ")
}

fn write_nro(
    s: &[f64],
    all: &[Vec<ReactionOrbitalSet>],
    dir: &Path,
    log: &Logger,
) -> MoptResult<()> {
    log.info("### Natural reaction orbitals along the path ###");
    log.info(
        "Reference: S. Ebisawa, M. Hasebe, T. Tsutsumi, T. Tsuneda, T. Taketsugu, Phys. Chem. \
         Chem. Phys. 2022, 24, 3532 (natural reaction orbitals)",
    );
    let mut table = String::from(
        "frame,arc_length,spin,sum_of_squares,lambda_1,share_1,lambda_2,share_2,lambda_3,\
         share_3,weakest_overlap\n",
    );
    let mut pairs = String::from(
        "Natural reaction orbitals: pairs of an occupied and an empty orbital that mix under \
         the displacement along the path.\nSingular values in amu^(-1/2) Bohr^(-1); orbitals \
         are numbered from 1 in the order of their energies.\n",
    );
    for (k, sets) in all.iter().enumerate() {
        for set in sets {
            let nro = &set.orbitals;
            let _ = write!(
                table,
                "{k},{:.8},{},{:.10e}",
                s[k],
                file_name(set.spin),
                nro.sum_of_squares
            );
            for p in 0..3 {
                match nro.singular_values.get(p) {
                    Some(value) => {
                        let _ = write!(table, ",{value:.10e},{:.6}", nro.shares[p]);
                    }
                    None => table.push_str(",,"),
                }
            }
            let _ = writeln!(table, ",{:.8}", set.weakest_overlap);
            let _ = writeln!(
                pairs,
                "\nframe {k} ({}), sum of squares {:.6e}, weakest overlap of an occupied \
                 orbital with its follower {:.6}",
                set.spin, nro.sum_of_squares, set.weakest_overlap
            );
            for p in 0..nro.singular_values.len().min(3) {
                let _ = writeln!(
                    pairs,
                    "  pair {}: singular value {:.6e} ({:.1} %)\n    occupied: {}\n    empty:    {}",
                    p + 1,
                    nro.singular_values[p],
                    100.0 * nro.shares[p],
                    composition(&nro.occupied_weights, p, &nro.occupied_indices),
                    composition(&nro.virtual_weights, p, &nro.virtual_indices)
                );
            }
            if set.weakest_overlap < 0.99 {
                log.warn(&format!(
                    "frame {k} ({}): an occupied orbital has an overlap of only {:.4} with what \
                     follows it at the displaced structures; a smaller -orbital_step may be \
                     needed",
                    set.spin, set.weakest_overlap
                ));
            }
        }
    }
    std::fs::write(dir.join("nro.csv"), table)?;
    std::fs::write(dir.join("nro_pairs.txt"), pairs)?;

    // The sum of the squared singular values along the path, and where it peaks.
    let spins: Vec<&'static str> = all
        .first()
        .map(|sets| sets.iter().map(|set| set.spin).collect())
        .unwrap_or_default();
    let mut figure = PlotFigure::new(
        "Natural reaction orbitals",
        "arc length (amu^0.5 Bohr)",
        "sum of squared singular values",
    );
    for (channel, spin) in spins.iter().enumerate() {
        let sums: Vec<f64> = all
            .iter()
            .map(|sets| sets[channel].orbitals.sum_of_squares)
            .collect();
        let peak = (0..sums.len())
            .max_by(|&a, &b| sums[a].total_cmp(&sums[b]))
            .unwrap_or(0);
        log.info(&format!(
            "{spin}: the sum of the squared singular values is largest at frame {peak} \
             ({:.6e} amu^-1 Bohr^-2, arc length {:.4})",
            sums[peak], s[peak]
        ));
        let nro = &all[peak][channel].orbitals;
        let n_ao = nro.occupied.nrows();
        let mut text = String::from("atomic_orbital");
        for p in 0..nro.singular_values.len() {
            let _ = write!(text, ",occupied_{},empty_{}", p + 1, p + 1);
        }
        text.push('\n');
        for row in 0..n_ao {
            let _ = write!(text, "{}", row + 1);
            for p in 0..nro.singular_values.len() {
                let _ = write!(
                    text,
                    ",{:.10e},{:.10e}",
                    nro.occupied[(row, p)],
                    nro.virtuals[(row, p)]
                );
            }
            text.push('\n');
        }
        std::fs::write(
            dir.join(format!("nro_orbitals_frame{peak}_{}.csv", file_name(spin))),
            text,
        )?;
        figure.add_series(PlotSeries {
            label: (*spin).to_string(),
            xs: s.to_vec(),
            ys: sums,
            color: COLORS[channel % COLORS.len()],
            style: PlotStyle::LinePoints,
        });
    }
    figure.save(dir.join("nro_sum_of_squares.png"))?;
    figure.save(dir.join("nro_sum_of_squares.svg"))?;
    Ok(())
}

fn write_roet(s: &[f64], all: &[FollowedEnergies], dir: &Path, log: &Logger) -> MoptResult<()> {
    log.info("### Orbital energies along the path ###");
    log.info(
        "Reference: T. Tsuneda, R. K. Singh, P. K. Chattaraj, Phys. Chem. Chem. Phys. 2018, 20, \
         14211 (reactive orbital energy diagrams)",
    );
    let frames = s.len();
    let mut energies = String::from("frame,arc_length,spin");
    let mut first = String::from("frame,arc_length,spin");
    let mut second = String::from("frame,arc_length,spin");
    let widest = all
        .iter()
        .map(|set| set.occupations.len())
        .max()
        .unwrap_or(0);
    for i in 1..=widest {
        let _ = write!(energies, ",orbital_{i}");
        let _ = write!(first, ",orbital_{i}");
        let _ = write!(second, ",orbital_{i}");
    }
    for text in [&mut energies, &mut first, &mut second] {
        text.push('\n');
    }
    let mut report = String::from(
        "Orbital energies followed along the path (Hartree), first and last frame, and their \
         relative change 2 (last - first) / |last + first|.\n",
    );
    let mut figure = PlotFigure::new(
        "Orbital energies along the path",
        "arc length (amu^0.5 Bohr)",
        "orbital energy (Hartree)",
    );
    for (channel, set) in all.iter().enumerate() {
        let n = set.occupations.len();
        // Derivatives of every orbital along the path.
        let mut slopes = vec![vec![0.0; n]; frames];
        let mut curvatures = vec![vec![0.0; n]; frames];
        for i in 0..n {
            let values: Vec<f64> = set.energies.iter().map(|row| row[i]).collect();
            let (d1, d2) = derivatives(s, &values)?;
            for k in 0..frames {
                slopes[k][i] = d1[k];
                curvatures[k][i] = d2[k];
            }
        }
        for k in 0..frames {
            let head = format!("{k},{:.8},{}", s[k], file_name(set.spin));
            for (text, row) in [
                (&mut energies, &set.energies[k]),
                (&mut first, &slopes[k]),
                (&mut second, &curvatures[k]),
            ] {
                text.push_str(&head);
                for value in row {
                    let _ = write!(text, ",{value:.10e}");
                }
                text.push('\n');
            }
        }
        let ends = reactive_orbitals(
            &set.energies[0],
            &set.energies[frames - 1],
            &set.occupations,
        )?;
        let weakest = set
            .weakest_overlaps
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let _ = writeln!(
            report,
            "\n{} (weakest overlap of an orbital with its follower from frame to frame: \
             {weakest:.6})\n{:>8} {:>10} {:>14} {:>14} {:>12}",
            set.spin, "orbital", "occupation", "first", "last", "change"
        );
        for i in 0..n {
            let _ = writeln!(
                report,
                "{:>8} {:>10.4} {:>14.8} {:>14.8} {:>12.6}",
                i + 1,
                set.occupations[i],
                set.energies[0][i],
                set.energies[frames - 1][i],
                ends.changes[i]
            );
        }
        let named = |index: Option<usize>| {
            index.map_or_else(
                || "none".to_string(),
                |i| format!("orbital {} (change {:.6})", i + 1, ends.changes[i]),
            )
        };
        let occupied = format!(
            "occupied reactive orbital (the most stabilized occupied one): {}",
            named(ends.occupied)
        );
        let unoccupied = format!(
            "unoccupied reactive orbital (the empty one that changes most): {}",
            named(ends.unoccupied)
        );
        for line in [&occupied, &unoccupied] {
            let _ = writeln!(report, "{line}");
            log.info(&format!("{}: {line}", set.spin));
        }
        if weakest < 0.9 {
            log.warn(&format!(
                "{}: an orbital has an overlap of only {weakest:.4} with its follower between \
                 two frames; the frames are too far apart to follow the orbitals safely",
                set.spin
            ));
        }
        // The figure: the reactive orbitals and the frontier, not every orbital.
        let highest = set.occupations.iter().copied().fold(0.0, f64::max);
        let filled: Vec<usize> = (0..n)
            .filter(|&i| highest > 0.0 && set.occupations[i] > 0.5 * highest)
            .collect();
        let homo = filled
            .iter()
            .copied()
            .max_by(|&a, &b| set.energies[0][a].total_cmp(&set.energies[0][b]));
        let lumo = (0..n)
            .filter(|i| !filled.contains(i))
            .min_by(|&a, &b| set.energies[0][a].total_cmp(&set.energies[0][b]));
        let mut shown: Vec<usize> = [ends.occupied, ends.unoccupied, homo, lumo]
            .into_iter()
            .flatten()
            .collect();
        shown.sort_unstable();
        shown.dedup();
        for (slot, &i) in shown.iter().enumerate() {
            figure.add_series(PlotSeries {
                label: format!("{} orbital {}", set.spin, i + 1),
                xs: s.to_vec(),
                ys: set.energies.iter().map(|row| row[i]).collect(),
                color: COLORS[(channel * 2 + slot) % COLORS.len()],
                style: PlotStyle::LinePoints,
            });
        }
    }
    std::fs::write(dir.join("orbital_energies.csv"), energies)?;
    std::fs::write(dir.join("1st_derivative_orbital_energy_plot.csv"), first)?;
    std::fs::write(dir.join("2nd_derivative_orbital_energy_plot.csv"), second)?;
    std::fs::write(dir.join("reactive_orbitals.txt"), report)?;
    figure.save(dir.join("orbital_energies.png"))?;
    figure.save(dir.join("orbital_energies.svg"))?;
    Ok(())
}

/// Run the analyses asked for along `frames` with the orbitals of `calc`.
pub fn run(
    settings: &OrbitalPathSettings,
    calc: &dyn Calculator,
    frames: &[MolecularSystem],
    temperature: f64,
    dir: &Path,
    log: &Logger,
) -> MoptResult<OrbitalPath> {
    if frames.len() < 2 {
        return Err(MoptError::config(
            "-nro, -roet and -diabatize need a file with the frames of a path (two at least)",
        ));
    }
    let s = arc_lengths(frames)?;
    if s.windows(2).any(|w| w[1] <= w[0]) {
        return Err(MoptError::config(
            "two consecutive frames of the path are the same structure",
        ));
    }
    let reaction_orbitals = if settings.nro {
        let all = natural_reaction_orbitals_along(calc, frames, settings.step)?;
        write_nro(&s, &all, dir, log)?;
        Some(all)
    } else {
        None
    };
    let energies = if settings.roet {
        let all = orbital_energies_along(calc, frames)?;
        write_roet(&s, &all, dir, log)?;
        Some(all)
    } else {
        None
    };
    let diabatic = if settings.diabatic.is_empty() {
        None
    } else {
        Some(super::diabatic_path::run(
            calc,
            frames,
            &settings.diabatic,
            temperature,
            dir,
            log,
        )?)
    };
    Ok(OrbitalPath {
        arc_lengths: s,
        reaction_orbitals,
        energies,
        diabatic,
    })
}
