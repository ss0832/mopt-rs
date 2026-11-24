//! `freqmain PATH.xyz -diabatize N M ...`: excited states along the frames of a path as
//! diabatic states, written into the run directory.
//!
//! `diabatic_states.csv` (for every frame the adiabatic energies, the diabatic potential
//! matrix, which adiabatic state every diabatic state mostly is, and how much of the states
//! the next frame keeps), `diabatic_crossings.txt` (where two diabatic states cross: the
//! coupling, the slopes, the Landau-Zener probability of a passage, and the estimates from
//! the gaps alone), `diabatic_crossings.csv` (the same crossings with the Zhu-Nakamura model
//! of the gaps, for the annotations of the kinetics, [`crate::workflow::diabatic_crossings`]),
//! `diabatic_states.png` / `.svg`.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::fmt::Write as _;
use std::path::Path;

use crate::analysis::diabatic::{
    crossings, from_gap, from_overlaps, landau_zener_stay, DiabaticCrossing, DiabaticStates,
};
use crate::core_infra::log::Logger;
use crate::core_infra::viz::{PlotFigure, PlotSeries, PlotStyle, Rgb};
use crate::domain::{MolecularSystem, Units};
use crate::error::MoptResult;
use crate::kinetics::rate::{adiabatic_model, peaked_model, AdiabaticModel, PeakedModel};
use crate::pes::calculator::Calculator;
use crate::workflow::diabatic_crossings::{self, slopes_of, CrossingRow};
use crate::workflow::state_path::{states_along, StateSet};

/// Share of the states kept from one frame to the next below which the step is pointed
/// out.
const KEPT_ENOUGH: f64 = 0.9;

const COLORS: [Rgb; 4] = [Rgb::BLUE, Rgb::RED, Rgb::EMERALD, Rgb::AMBER];

/// What `-diabatize` found, for callers that go on with it.
#[derive(Debug, Clone)]
pub struct DiabaticPath {
    /// The adiabatic states along the frames.
    pub set: StateSet,
    /// The diabatic states made of them (state `i` is the state `set.numbers[0][i]` of
    /// the first frame).
    pub states: DiabaticStates,
    /// Where two diabatic states cross.
    pub crossings: Vec<DiabaticCrossing>,
    /// For every two states that are neighbours in energy (`i` and `i + 1` of the set), the
    /// crossing of the two-state model fitted to their smallest gap, if the gap has a
    /// minimum inside the path.
    pub from_gaps: Vec<Option<DiabaticCrossing>>,
    /// For the same two states, Zhu and Nakamura's two-state model from the adiabatic
    /// energies, if the gap has a minimum inside the path and the crossing is sloped.
    pub zhu_nakamura: Vec<Option<AdiabaticModel>>,
    /// For the same two states, the model of a peaked crossing from the adiabatic energies
    /// (the top of the lower one and the bottom of the upper one inside the path), where no
    /// sloped model fits.
    pub zhu_nakamura_peaked: Vec<Option<PeakedModel>>,
}

fn kcal(hartree: f64) -> f64 {
    hartree * Units::HARTREE_TO_KCAL_MOL
}

/// The excited states `states` (numbers at the first frame) along `frames` as diabatic
/// states; the files go to `dir`. `temperature` (K) is that of the Landau-Zener
/// probabilities in the report.
pub fn run(
    calc: &dyn Calculator,
    frames: &[MolecularSystem],
    states: &[usize],
    temperature: f64,
    dir: &Path,
    log: &Logger,
) -> MoptResult<DiabaticPath> {
    let named = |numbers: &[usize]| {
        numbers
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    };
    log.info(&format!(
        "### Diabatic states of the excited states {} along the path ###",
        named(states)
    ));
    log.info(
        "Reference: S. P. Neville, I. Seidu, M. S. Schuurman, arXiv:1912.09553 (propagative \
         block-diagonalisation diabatisation); T. Pacher, L. S. Cederbaum, H. Koppel, Adv. \
         Chem. Phys. 1993, 84, 293",
    );
    let set = states_along(calc, frames, states)?;
    let found = from_overlaps(&set.energies, &set.overlaps)?;
    let crossed = crossings(&set.arc_lengths, &found);
    let n = found.states();
    let level = |i: usize| set.energies.iter().map(|e| e[i]).collect::<Vec<f64>>();
    let fitted: Vec<Option<DiabaticCrossing>> = (0..n - 1)
        .map(|i| from_gap(&set.arc_lengths, &level(i), &level(i + 1)))
        .collect();
    let s = &set.arc_lengths;
    let zero = set.energies[0][0];
    // The diabatic states are named by the numbers of the states of the first frame.
    let names = &set.numbers[0];

    let mut table = String::from("frame,arc_length");
    for i in 0..n {
        let _ = write!(table, ",state_{}", i + 1);
    }
    for i in 0..n {
        let _ = write!(table, ",e_{}", i + 1);
    }
    for i in 0..n {
        for j in i..n {
            let _ = write!(table, ",w_{}_{}", names[i], names[j]);
        }
    }
    for name in names {
        let _ = write!(table, ",mostly_{name},share_{name}");
    }
    table.push_str(",kept_to_next\n");
    for (k, position) in s.iter().enumerate() {
        let _ = write!(table, "{k},{position:.8}");
        for number in &set.numbers[k] {
            let _ = write!(table, ",{number}");
        }
        for energy in &set.energies[k] {
            let _ = write!(table, ",{energy:.10}");
        }
        for i in 0..n {
            for j in i..n {
                let _ = write!(table, ",{:.10e}", found.potentials[k][(i, j)]);
            }
        }
        for i in 0..n {
            let most = (0..n)
                .max_by(|&a, &b| found.weights[k][(i, a)].total_cmp(&found.weights[k][(i, b)]))
                .unwrap_or(i);
            let _ = write!(
                table,
                ",{},{:.6}",
                set.numbers[k][most],
                found.weights[k][(i, most)]
            );
        }
        match found.kept.get(k) {
            Some(kept) => {
                let _ = writeln!(table, ",{kept:.6}");
            }
            None => table.push_str(",\n"),
        }
    }
    std::fs::write(dir.join("diabatic_states.csv"), table)?;

    let mut report = format!(
        "Diabatic states of the excited states {} (numbers at the first frame) along the \
         path.\nThe diabatic states are the states of the first frame, carried from frame to \
         frame by the overlaps of the states (propagative block-diagonalisation \
         diabatisation).\nEnergies in Hartree; positions in amu^(1/2) Bohr; kcal/mol relative \
         to the lowest of the states at the first frame.\n",
        named(names)
    );
    let weakest = found
        .kept
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.total_cmp(b.1));
    if let Some((step, kept)) = weakest {
        let _ = writeln!(
            report,
            "\nThe states of a frame are kept by the states of the next to {kept:.4} at least \
             (the step from frame {step} to {}).",
            step + 1
        );
        if *kept < KEPT_ENOUGH {
            let line = format!(
                "Between the frames {step} and {} only {kept:.3} of one of the states is found \
                 again in the states that are followed: another state mixes in, or the frames \
                 are too far apart. The diabatic states are uncertain from there on.",
                step + 1
            );
            log.warn(&line);
            let _ = writeln!(report, "Warning: {line}");
        }
    }
    let kinetic = 0.5 * Units::BOLTZMANN_HARTREE_PER_K * temperature;
    if crossed.is_empty() {
        let line = "The diabatic states do not cross between the frames.";
        log.info(line);
        let _ = writeln!(report, "\n{line}");
    }
    for (count, crossing) in crossed.iter().enumerate() {
        let (i, j) = crossing.states;
        let stay = landau_zener_stay(crossing.coupling, crossing.slope_difference, kinetic);
        let line = format!(
            "Crossing {} of the states {} and {}: at {:.4} amu^0.5 Bohr, {:.3} kcal/mol; \
             coupling {:.6e} Hartree ({:.4} kcal/mol, {:.1} cm^-1); slope difference {:.6e} \
             Hartree/(amu^0.5 Bohr)",
            count + 1,
            names[i],
            names[j],
            crossing.position,
            kcal(crossing.energy - zero),
            crossing.coupling,
            kcal(crossing.coupling),
            crossing.coupling * Units::HARTREE_TO_WAVENUMBER,
            crossing.slope_difference
        );
        log.info(&line);
        let _ = writeln!(report, "\n{line}");
        let _ = writeln!(
            report,
            "  energy {:.10} Hartree\n  Landau-Zener for these two states alone, one passage \
             with the kinetic energy kT/2 along the path at {temperature:.2} K ({kinetic:.6e} \
             Hartree):\n    stays on its diabatic state (changes the adiabatic state) with \
             probability {stay:.6}\n    follows the adiabatic state with probability {:.6}",
            crossing.energy,
            1.0 - stay
        );
    }
    for (i, model) in fitted.iter().enumerate() {
        let pair = format!(
            "the states {} and {} by energy",
            set.numbers[0][i],
            set.numbers[0][i + 1]
        );
        match model {
            Some(model) => {
                let line = format!(
                    "From the gap alone, {pair} (two straight states with a constant coupling \
                     fitted to the smallest gap): at {:.4} amu^0.5 Bohr, coupling {:.6e} \
                     Hartree ({:.4} kcal/mol), slope difference {:.6e} Hartree/(amu^0.5 Bohr)",
                    model.position,
                    model.coupling,
                    kcal(model.coupling),
                    model.slope_difference
                );
                log.info(&line);
                let _ = writeln!(report, "\n{line}");
            }
            None => {
                let _ = writeln!(
                    report,
                    "\nThe gap of {pair} has no minimum inside the path: no estimate from the \
                     gap alone."
                );
            }
        }
    }
    let zhu_nakamura: Vec<Option<AdiabaticModel>> = (0..n - 1)
        .map(|i| adiabatic_model(s, &level(i), &level(i + 1)))
        .collect();
    let zhu_nakamura_peaked: Vec<Option<PeakedModel>> = (0..n - 1)
        .map(|i| match zhu_nakamura[i] {
            Some(_) => None,
            None => peaked_model(s, &level(i), &level(i + 1)),
        })
        .collect();
    for (i, model) in zhu_nakamura.iter().enumerate() {
        let pair = format!(
            "the states {} and {} by energy",
            set.numbers[0][i],
            set.numbers[0][i + 1]
        );
        let line = match model {
            Some(m) => format!(
                "Zhu-Nakamura model of {pair} from the adiabatic energies (a sloped crossing): \
                 at {:.4} amu^0.5 Bohr, {:.3} kcal/mol; coupling {:.6e} Hartree; slopes of the \
                 same sign, sqrt(F1 F2) {:.6e} and |F1 - F2| {:.6e} Hartree/(amu^0.5 Bohr)",
                m.position,
                kcal(m.energy - zero),
                m.coupling,
                m.mean_slope.abs(),
                m.slope_difference
            ),
            None => match zhu_nakamura_peaked[i] {
                Some(m) => format!(
                    "Zhu-Nakamura model of {pair} from the adiabatic energies (a peaked \
                     crossing, the nonadiabatic-tunnelling type): at {:.4} amu^0.5 Bohr, \
                     {:.3} kcal/mol; coupling {:.6e} Hartree; slopes of opposite signs, \
                     {:.6e} and {:.6e} Hartree/(amu^0.5 Bohr)",
                    m.position,
                    kcal(m.energy - zero),
                    m.coupling,
                    m.slopes[0],
                    m.slopes[1]
                ),
                None => format!(
                    "No Zhu-Nakamura model for {pair}: the smallest gap is at an end of the \
                     path, or neither the sloped form (the lower energy reaching the crossing \
                     energy on one side and the upper on the other) nor the peaked one (a top \
                     of the lower energy and a bottom of the upper one around the smallest \
                     gap) fits inside the path."
                ),
            },
        };
        log.info(&line);
        let _ = writeln!(report, "\n{line}");
    }
    std::fs::write(dir.join("diabatic_crossings.txt"), report)?;
    let mut rows: Vec<CrossingRow> = crossed
        .iter()
        .map(|c| CrossingRow {
            kind: "crossing".into(),
            states: [names[c.states.0], names[c.states.1]],
            position: c.position,
            energy: c.energy,
            coupling: c.coupling,
            slopes: c.slopes,
        })
        .collect();
    for (i, model) in fitted.iter().enumerate() {
        if let Some(c) = model {
            rows.push(CrossingRow {
                kind: "gap".into(),
                states: [set.numbers[0][i], set.numbers[0][i + 1]],
                position: c.position,
                energy: c.energy,
                coupling: c.coupling,
                slopes: c.slopes,
            });
        }
    }
    for i in 0..n - 1 {
        let states = [set.numbers[0][i], set.numbers[0][i + 1]];
        if let Some(m) = zhu_nakamura[i] {
            rows.push(CrossingRow {
                kind: "zhu_nakamura".into(),
                states,
                position: m.position,
                energy: m.energy,
                coupling: m.coupling,
                slopes: slopes_of(m.mean_slope, m.slope_difference),
            });
        } else if let Some(m) = zhu_nakamura_peaked[i] {
            rows.push(CrossingRow {
                kind: "zhu_nakamura".into(),
                states,
                position: m.position,
                energy: m.energy,
                coupling: m.coupling,
                slopes: m.slopes,
            });
        }
    }
    diabatic_crossings::write(&dir.join("diabatic_crossings.csv"), &rows)?;

    let mut figure = PlotFigure::new(
        "Adiabatic and diabatic states",
        "arc length (amu^0.5 Bohr)",
        "energy (kcal/mol)",
    );
    let relative = |values: Vec<f64>| values.iter().map(|e| kcal(e - zero)).collect::<Vec<f64>>();
    for i in 0..n {
        figure.add_series(PlotSeries {
            label: format!("adiabatic {}", i + 1),
            xs: s.clone(),
            ys: relative(level(i)),
            color: Rgb::SLATE,
            style: PlotStyle::Scatter,
        });
    }
    for (i, name) in names.iter().enumerate() {
        figure.add_series(PlotSeries {
            label: format!("diabatic {name}"),
            xs: s.clone(),
            ys: relative(found.element(i, i)),
            color: COLORS[i % COLORS.len()],
            style: PlotStyle::LinePoints,
        });
    }
    figure.save(dir.join("diabatic_states.png"))?;
    figure.save(dir.join("diabatic_states.svg"))?;
    Ok(DiabaticPath {
        set,
        states: found,
        crossings: crossed,
        from_gaps: fitted,
        zhu_nakamura,
        zhu_nakamura_peaked,
    })
}
