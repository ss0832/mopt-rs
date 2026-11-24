//! The parts of `run_kinetics` beyond the canonical model: the annotations of the models and
//! corrections that need more than energies (`residence_shape`, `lennard_jones`,
//! `uncertainty_kcal_mol`, the friction of `kramers` and `grote_hynes`, the transmission
//! coefficient of `reactive_flux`, the path of `small_curvature`, the measured `outcomes` of a
//! saddle point), the semi-Markov model of a trajectory assigned to the minima of the network
//! (with its Markov state model), the refinement tasks, and the files of these models
//! (`phenomenological.csv`, `memory.csv`, `msm.csv`, `refinement.csv`).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use crate::core_infra::json::JsonValue;
use crate::core_infra::log::Logger;
use crate::domain::{SpinCharge, Units};
use crate::error::{MoptError, MoptResult};
use crate::io::xyz::read_xyz_trajectory_file;
use crate::kinetics::generator::Generator;
use crate::kinetics::msm::{
    core_set, count_matrix, dwell_statistics, implied_timescales, rate_matrix,
    reversible_transition_matrix,
};
use crate::kinetics::network::{
    Bottleneck, Friction, KineticNetwork, KineticState, TunnellingPath,
};
use crate::kinetics::refine::RefinementTask;
use crate::kinetics::semi_markov::semi_markov_memory;
use crate::kinetics::solve::{Evolution, SemiMarkov};
use crate::kinetics::RateSet;
use crate::workflow::kinetics_dynamics::MinimumClassifier;

/// The node keys of the annotations read here.
pub const NODE_KEYS: &[&str] = &[
    "xyz",
    "residence_shape",
    "lennard_jones",
    "uncertainty_kcal_mol",
];
/// The edge keys of the annotations read here.
pub const EDGE_KEYS: &[&str] = &[
    "xyz",
    "friction_s-1",
    "friction_memory_s",
    "friction_kernel_file",
    "reactive_flux_kappa",
    "reactive_flux_error",
    "sct_profile_file",
    "uncertainty_kcal_mol",
    "outcomes",
    "outcome_errors",
];

/// A trajectory of molecular dynamics for the semi-Markov model.
#[derive(Debug, Clone, PartialEq)]
pub struct MdTrajectory {
    /// The XYZ trajectory.
    pub path: PathBuf,
    /// The time between two frames (s).
    pub time_step: f64,
    /// The lag of the Markov state model (frames).
    pub lag: usize,
}

/// The settings of the analyses beyond the canonical model.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct KineticsExtras {
    /// The seed of `kmc` (the seed of `--uncertainty` when that is given).
    pub seed: Option<u64>,
    /// `--refine [NODE ...]`: rank the refinement tasks for the populations of these nodes at
    /// the last output time (empty: the most populated state).
    pub refine: Option<Vec<usize>>,
    /// The standard deviation of the energies of the refinement tasks (Hartree) without
    /// `--uncertainty`.
    pub refine_sigma: Option<f64>,
    /// `--md-trajectory`: the semi-Markov model of a trajectory.
    pub md_trajectory: Option<MdTrajectory>,
}

/// The Markov state model of a trajectory.
#[derive(Debug, Clone, PartialEq)]
pub struct MsmReport {
    /// The trajectory.
    pub source: String,
    /// The frames assigned to a state.
    pub frames: usize,
    /// The lag (s).
    pub lag_time: f64,
    /// The implied time scales (s), slowest first.
    pub timescales: Vec<f64>,
    /// The rate matrix `ln(T) / lag` (`[to][from]`, 1/s).
    pub rates: Vec<Vec<f64>>,
    /// Its most negative off-diagonal entry over the largest rate (0 for a generator).
    pub most_negative: f64,
    /// The visits, mean residence times (s), and squared coefficients of variation.
    pub visits: Vec<usize>,
    /// Mean residence times (s).
    pub mean: Vec<f64>,
    /// `cv^2` of the residence times.
    pub cv2: Vec<f64>,
}

/// What the analyses beyond the canonical model gave.
#[derive(Debug, Clone, Default)]
pub struct ExtraResults {
    /// The refinement tasks for every observable (its label).
    pub refinement: Vec<(String, Vec<RefinementTask>)>,
    /// The Markov state model of `--md-trajectory`.
    pub msm: Option<MsmReport>,
}

fn structure(v: &JsonValue, what: &str) -> MoptResult<crate::domain::MolecularSystem> {
    let text = v
        .as_str()
        .ok_or_else(|| MoptError::parse(format!("{what} is XYZ text")))?;
    crate::io::xyz::parse_xyz_str(text, SpinCharge::default(), None).map_err(|e| e.context(what))
}

fn number(v: &JsonValue, what: &str) -> MoptResult<f64> {
    v.as_f64()
        .ok_or_else(|| MoptError::parse(format!("{what} is a number")))
}

/// Read the node key `key` of the annotations into `s`; whether it is one of [`NODE_KEYS`].
pub fn apply_node_key(
    s: &mut KineticState,
    key: &str,
    v: &JsonValue,
    what: &str,
) -> MoptResult<bool> {
    match key {
        "xyz" => s.system = Some(structure(v, what)?),
        "residence_shape" => {
            let k = number(v, what)?;
            if !(k > 0.0 && k.is_finite()) {
                return Err(MoptError::parse(format!("{what} must be positive")));
            }
            s.extra.residence_shape = Some(k);
        }
        "lennard_jones" => {
            let pair = v
                .as_array()
                .and_then(|a| match a {
                    [x, y] => Some((x.as_f64()?, y.as_f64()?)),
                    _ => None,
                })
                .ok_or_else(|| {
                    MoptError::parse(format!("{what} is [SIGMA_ANGSTROM, EPSILON_K]"))
                })?;
            s.extra.lennard_jones = Some(pair);
        }
        "uncertainty_kcal_mol" => {
            s.extra.uncertainty = Some(number(v, what)?.abs() * Units::KCAL_MOL_TO_HARTREE)
        }
        _ => return Ok(false),
    }
    Ok(true)
}

/// Read the edge key `key` of the annotations into `b` (files relative to `base`); whether it
/// is one of [`EDGE_KEYS`] but `outcomes` and `outcome_errors`, which [`apply_outcomes`] reads.
pub fn apply_edge_key(
    b: &mut Bottleneck,
    key: &str,
    v: &JsonValue,
    what: &str,
    base: &Path,
) -> MoptResult<bool> {
    let path = || -> MoptResult<PathBuf> {
        v.as_str()
            .map(|p| base.join(p))
            .ok_or_else(|| MoptError::parse(format!("{what} is a path")))
    };
    fn friction(b: &mut Bottleneck) -> &mut Friction {
        let source = format!("{} (annotations)", b.label);
        b.friction.get_or_insert_with(|| Friction {
            source,
            ..Friction::default()
        })
    }
    match key {
        "xyz" => b.system = Some(structure(v, what)?),
        "friction_s-1" => friction(b).gamma = Some(number(v, what)?),
        "friction_memory_s" => friction(b).memory = Some(number(v, what)?),
        "friction_kernel_file" => {
            let p = path()?;
            let kernel = read_friction_kernel(&p)?;
            let f = friction(b);
            f.kernel = Some(kernel);
            f.source = p.display().to_string();
        }
        "reactive_flux_kappa" => {
            let e = b.reactive_flux.map_or(0.0, |r| r.1);
            b.reactive_flux = Some((number(v, what)?, e));
        }
        "reactive_flux_error" => {
            let k = b.reactive_flux.map_or(f64::NAN, |r| r.0);
            b.reactive_flux = Some((k, number(v, what)?));
        }
        "sct_profile_file" => b.tunnelling_path = Some(read_tunnelling_path(&path()?)?),
        "uncertainty_kcal_mol" => {
            b.uncertainty = Some(number(v, what)?.abs() * Units::KCAL_MOL_TO_HARTREE)
        }
        _ => return Ok(false),
    }
    Ok(true)
}

/// The columns `names` of a CSV file of numbers.
fn csv_columns(path: &Path, names: &[&str]) -> MoptResult<Vec<Vec<f64>>> {
    let text =
        std::fs::read_to_string(path).map_err(|e| MoptError::from(e).context(path.display()))?;
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let header: Vec<&str> = lines
        .next()
        .ok_or_else(|| MoptError::parse(format!("{}: empty", path.display())))?
        .split(',')
        .map(str::trim)
        .collect();
    let index: Vec<usize> = names
        .iter()
        .map(|n| {
            header.iter().position(|h| h == n).ok_or_else(|| {
                MoptError::parse(format!(
                    "{}: no column {n} (the columns {} are needed)",
                    path.display(),
                    names.join(", ")
                ))
            })
        })
        .collect::<MoptResult<_>>()?;
    let mut out = vec![Vec::new(); names.len()];
    for line in lines {
        let cells: Vec<&str> = line.split(',').map(str::trim).collect();
        for (col, &i) in out.iter_mut().zip(&index) {
            col.push(
                cells
                    .get(i)
                    .and_then(|x| x.parse::<f64>().ok())
                    .ok_or_else(|| {
                        MoptError::parse(format!("{}: bad line '{line}'", path.display()))
                    })?,
            );
        }
    }
    Ok(out)
}

/// A friction kernel `time_s,zeta_s-2` (as [`write_friction_kernel`] writes it).
pub fn read_friction_kernel(path: &Path) -> MoptResult<(Vec<f64>, Vec<f64>)> {
    let mut c = csv_columns(path, &["time_s", "zeta_s-2"])?;
    let z = c.pop().unwrap_or_default();
    let t = c.pop().unwrap_or_default();
    Ok((t, z))
}

/// The text of a friction kernel file.
pub fn write_friction_kernel(times: &[f64], values: &[f64]) -> String {
    let mut s = String::from("time_s,zeta_s-2\n");
    for (t, z) in times.iter().zip(values) {
        let _ = writeln!(s, "{t:.10e},{z:.10e}");
    }
    s
}

/// The adiabatic barrier and the effective masses of `path_profile.csv` of `freqmain -path`.
pub fn read_tunnelling_path(path: &Path) -> MoptResult<TunnellingPath> {
    let c = csv_columns(
        path,
        &[
            "arc_length_sqrt_amu_bohr",
            "adiabatic_energy_hartree",
            "sct_mass_ratio",
        ],
    )?;
    Ok(TunnellingPath {
        arc_length: c[0].clone(),
        adiabatic: c[1].clone(),
        mass_ratio: c[2].clone(),
        source: path.display().to_string(),
    })
}

/// Replace the bottleneck `k` (with the steps of its N-furcation, if any) by steps from the
/// state `outcomes.from` to every state of `outcomes.to` with those shares (normalized), and
/// their standard errors from `errors`.
pub fn apply_outcomes(
    net: &mut KineticNetwork,
    k: usize,
    outcomes: &JsonValue,
    errors: Option<&JsonValue>,
    name: &str,
) -> MoptResult<()> {
    let what = format!("annotations: edges.{name}.outcomes");
    let m = outcomes.as_object().ok_or_else(|| {
        MoptError::parse(format!(
            "{what} is {{\"from\": STATE, \"to\": {{STATE: SHARE}}}}"
        ))
    })?;
    let state = |label: &str| -> MoptResult<usize> {
        net.states
            .iter()
            .position(|s| s.label == label || s.id.to_string() == label)
            .ok_or_else(|| MoptError::config(format!("{what}: no state '{label}'")))
    };
    let label_of = |v: &JsonValue| -> Option<String> {
        v.as_str()
            .map(str::to_string)
            .or_else(|| v.as_usize().map(|x| x.to_string()))
    };
    let from = m
        .get("from")
        .and_then(label_of)
        .ok_or_else(|| MoptError::parse(format!("{what}.from is a state")))?;
    let from = state(&from)?;
    let to = m
        .get("to")
        .and_then(JsonValue::as_object)
        .ok_or_else(|| MoptError::parse(format!("{what}.to is {{STATE: SHARE}}")))?;
    let mut shares = Vec::new();
    for (label, v) in to {
        let x = number(v, &format!("{what}.to.{label}"))?;
        if !(x >= 0.0 && x.is_finite()) {
            return Err(MoptError::parse(format!(
                "{what}.to.{label} must not be negative"
            )));
        }
        let err = errors
            .and_then(JsonValue::as_object)
            .and_then(|e| e.get(label))
            .and_then(JsonValue::as_f64);
        shares.push((state(label)?, x, err));
    }
    let total: f64 = shares.iter().map(|s| s.1).sum();
    if total.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
        return Err(MoptError::parse(format!("{what}: the shares add to zero")));
    }
    if shares.iter().any(|s| s.0 == from) {
        return Err(MoptError::parse(format!(
            "{what}: an outcome is the state the flux comes from"
        )));
    }
    let base = net.bottlenecks[k].clone();
    net.bottlenecks
        .retain(|b| !(b.id == base.id && b.label == base.label));
    for (to, x, err) in shares {
        if x == 0.0 {
            continue;
        }
        let mut b = base.clone();
        b.states = [from, to];
        b.share = (x / total < 1.0).then_some(x / total);
        b.share_error = err.map(|e| e / total);
        net.bottlenecks.push(b);
    }
    Ok(())
}

/// The semi-Markov model of the trajectory `md` assigned to the minima of `net` that are
/// states of `rates`: the mean residence times, their spreads (the shapes of gamma
/// distributions of the same mean and variance), the jumps, and the Markov state model at the
/// lag of `md`.
pub fn md_semi_markov(
    net: &KineticNetwork,
    rates: &RateSet,
    md: &MdTrajectory,
    log: &Logger,
) -> MoptResult<(SemiMarkov, MsmReport)> {
    let traj = read_xyz_trajectory_file(&md.path, SpinCharge::default(), None)
        .map_err(|e| e.context(md.path.display()))?;
    let Some(first) = traj.frames.first() else {
        return Err(MoptError::config(format!(
            "{}: no frames",
            md.path.display()
        )));
    };
    let minima: Vec<_> = rates
        .states
        .iter()
        .map(|&k| {
            net.states[k].system.clone().ok_or_else(|| {
                MoptError::config(format!(
                    "--md-trajectory assigns the frames to the minima by their structures; {} has \
                     none",
                    net.states[k].label
                ))
            })
        })
        .collect::<MoptResult<_>>()?;
    let classifier = MinimumClassifier::new(&first.system, &minima);
    let mut assigned = Vec::with_capacity(traj.frames.len());
    for f in &traj.frames {
        assigned.push(classifier.classify(&f.system.coords_flat_bohr())?);
    }
    let d = core_set(&assigned);
    let n = rates.states.len();
    let stats = dwell_statistics(std::slice::from_ref(&d), n, md.time_step)?;
    let k = stats.markov_limit();
    let generator = Generator::from_matrix(&k)?;
    let generator = Generator {
        labels: rates.labels.clone(),
        ..generator
    }
    .with_detected_balance(1e-9);
    let cv2: Vec<f64> = (0..n)
        .map(|j| {
            if stats.visits[j] > 1 {
                stats.cv2(j)
            } else {
                1.0
            }
        })
        .collect();
    let shapes: Vec<f64> = cv2
        .iter()
        .map(|c| if *c > 1e-6 { 1.0 / c } else { 1e6 })
        .collect();
    let k0 = generator.rates().matrix();
    let k1 = semi_markov_memory(&k0, &cv2);
    let m = faer::Mat::<f64>::identity(n, n) - &k1;
    let memory = {
        let lu = m.partial_piv_lu();
        let inv =
            faer::linalg::solvers::Solve::solve(&lu, faer::Mat::<f64>::identity(n, n).as_ref());
        let finite = inv.col_iter().all(|c| c.iter().all(|v| v.is_finite()));
        finite.then(|| &k0 * &inv)
    };
    let c = count_matrix(std::slice::from_ref(&d), n, md.lag)?;
    let (t, pi) = reversible_transition_matrix(&c)?;
    let lag_time = md.lag as f64 * md.time_step;
    let timescales = implied_timescales(&t, &pi, lag_time)?;
    let (msm_rates, most_negative) = rate_matrix(&t, &pi, lag_time)?;
    let unvisited: Vec<&str> = (0..n)
        .filter(|&j| stats.visits[j] == 0)
        .map(|j| rates.labels[j].as_str())
        .collect();
    let mut notes = vec![format!(
        "semi_markov: residence times and jumps from {} ({} frames of {:.3e} s, {} assigned to a \
         state); the Markov state model at the lag {:.3e} s",
        md.path.display(),
        traj.frames.len(),
        md.time_step,
        d.len(),
        lag_time
    )];
    if !unvisited.is_empty() {
        notes.push(format!(
            "semi_markov: no completed visit of {} in the trajectory: no exits from there",
            unvisited.join(", ")
        ));
    }
    for n in &notes {
        log.info(n);
    }
    let report = MsmReport {
        source: md.path.display().to_string(),
        frames: d.len(),
        lag_time,
        timescales,
        rates: msm_rates,
        most_negative,
        visits: stats.visits.clone(),
        mean: stats.mean.clone(),
        cv2,
    };
    Ok((
        SemiMarkov {
            generator,
            shapes,
            memory,
            notes,
        },
        report,
    ))
}

/// Write the files of the models beyond the canonical one.
pub fn write_outputs(
    evolution: &Evolution,
    labels: &[String],
    extra: &ExtraResults,
    dir: &Path,
) -> MoptResult<()> {
    let write = |name: &str, content: &str| -> MoptResult<()> {
        let path = dir.join(name);
        std::fs::write(&path, content).map_err(|e| MoptError::from(e).context(path.display()))
    };
    if let Evolution::Resolved(r) = evolution {
        if let Some(p) = &r.phenomenological {
            let mut csv = String::from(
                "from,to,k_contraction_a_s-1,k_contraction_b_s-1,k_eigen_s-1,k_high_pressure_s-1,\
                 k_over_k_inf\n",
            );
            let n = p.species.len();
            for j in 0..n {
                for i in 0..n {
                    if i == j {
                        continue;
                    }
                    let inf = p.high_pressure[(i, j)];
                    let a = p.contraction_a[(i, j)];
                    if a == 0.0 && inf == 0.0 {
                        continue;
                    }
                    let _ = writeln!(
                        csv,
                        "{},{},{a:.10e},{:.10e},{},{inf:.10e},{}",
                        labels[j],
                        labels[i],
                        p.contraction_b[(i, j)],
                        p.eigen
                            .as_ref()
                            .map_or(String::new(), |e| format!("{:.10e}", e[(i, j)])),
                        if inf > 0.0 {
                            format!("{:.10e}", a / inf)
                        } else {
                            String::new()
                        }
                    );
                }
            }
            let _ = writeln!(
                csv,
                "# gap {:.6e}; slowest eigenvalues {}{}",
                p.gap,
                p.eigenvalues
                    .iter()
                    .map(|v| format!("{v:.6e}"))
                    .collect::<Vec<_>>()
                    .join(" "),
                p.eigen_note
                    .as_ref()
                    .map_or(String::new(), |n| format!("; {n}"))
            );
            write("phenomenological.csv", &csv)?;
        }
    }
    if let Evolution::SemiMarkov(s) = evolution {
        if let Some(m) = &s.memory {
            let k0 = s.generator.rates();
            let mut csv = String::from("from,to,k_markov_s-1,k_memory_s-1,shape\n");
            for j in 0..k0.n {
                for i in 0..k0.n {
                    if i != j && (k0.get(i, j) > 0.0 || m[(i, j)] != 0.0) {
                        let _ = writeln!(
                            csv,
                            "{},{},{:.10e},{:.10e},{}",
                            labels[j],
                            labels[i],
                            k0.get(i, j),
                            m[(i, j)],
                            s.shapes[j]
                        );
                    }
                }
            }
            write("memory.csv", &csv)?;
        }
    }
    if let Some(m) = &extra.msm {
        let mut csv = String::from("state,visits,mean_residence_s,cv2\n");
        for (j, l) in labels.iter().enumerate() {
            let _ = writeln!(
                csv,
                "{l},{},{:.10e},{:.6e}",
                m.visits[j], m.mean[j], m.cv2[j]
            );
        }
        let _ = writeln!(
            csv,
            "# lag {:.6e} s; implied time scales (s) {}; most negative off-diagonal rate over the \
             largest {:.3e}",
            m.lag_time,
            m.timescales
                .iter()
                .map(|v| format!("{v:.6e}"))
                .collect::<Vec<_>>()
                .join(" "),
            m.most_negative
        );
        write("msm.csv", &csv)?;
    }
    if !extra.refinement.is_empty() {
        let mut csv =
            String::from("observable,rank,task,target,parameter,sensitivity,sigma,score\n");
        for (obs, tasks) in &extra.refinement {
            for (r, t) in tasks.iter().enumerate() {
                let _ = writeln!(
                    csv,
                    "{obs},{},{},{},{},{:.6e},{:.6e},{:.6e}",
                    r + 1,
                    t.kind.name(),
                    t.target,
                    t.parameter,
                    t.sensitivity,
                    t.sigma,
                    t.score
                );
            }
        }
        write("refinement.csv", &csv)?;
    }
    Ok(())
}

/// The lines of the screen for the models beyond the canonical one.
pub fn report_lines(evolution: &Evolution, labels: &[String], extra: &ExtraResults) -> Vec<String> {
    let mut out = Vec::new();
    match evolution {
        Evolution::Resolved(r) => {
            out.extend(r.notes.iter().cloned());
            if let Some(p) = &r.phenomenological {
                out.push(format!(
                    "Phenomenological rate constants (1/s; separation of the species from the \
                     relaxation of the energies: {:.3e}):",
                    p.gap
                ));
                let n = p.species.len();
                for j in 0..n {
                    for i in 0..n {
                        let inf = p.high_pressure[(i, j)];
                        if i != j && inf > 0.0 {
                            out.push(format!(
                                "  {} -> {}: {:.4e} (contraction), {} (eigenpairs), k/k_inf {:.4e}",
                                labels[j],
                                labels[i],
                                p.contraction_a[(i, j)],
                                p.eigen
                                    .as_ref()
                                    .map_or("-".to_string(), |e| format!("{:.4e}", e[(i, j)])),
                                p.contraction_a[(i, j)] / inf
                            ));
                        }
                    }
                }
                if let Some(note) = &p.eigen_note {
                    out.push(format!("  eigenpairs: {note}"));
                }
            }
        }
        Evolution::SemiMarkov(s) => out.extend(s.notes.iter().cloned()),
        _ => {}
    }
    if let Some(m) = &extra.msm {
        out.push(format!(
            "Markov state model at the lag {:.3e} s: implied time scales {} s",
            m.lag_time,
            m.timescales
                .iter()
                .take(5)
                .map(|v| format!("{v:.3e}"))
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    for (obs, tasks) in &extra.refinement {
        out.push(format!(
            "Refinement tasks for {obs} (|d ln y / d theta| sigma):"
        ));
        for t in tasks.iter().take(10) {
            out.push(format!(
                "  {:<14} {:<10} {:<20} score {:.3e}",
                t.kind.name(),
                t.target,
                t.parameter,
                t.score
            ));
        }
    }
    out
}

/// The `BTreeMap` of an annotations document for [`apply_outcomes`] (used by the tests).
pub fn outcomes_document(from: &str, to: &[(&str, f64)]) -> JsonValue {
    let mut t = BTreeMap::new();
    for (l, x) in to {
        t.insert((*l).to_string(), JsonValue::Number(*x));
    }
    let mut m = BTreeMap::new();
    m.insert("from".to_string(), JsonValue::String(from.to_string()));
    m.insert("to".to_string(), JsonValue::Object(t));
    JsonValue::Object(m)
}
