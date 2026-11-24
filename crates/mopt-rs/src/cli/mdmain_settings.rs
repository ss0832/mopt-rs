//! The options of `mdmain` as settings of the dynamics: `-mt` (the ensemble and its parts),
//! `-ts`, `-time`, `-temp`, `-ct`, `-press`, `-cc`, and the `KEY=VALUE` settings of
//! `-md_opt`.
//!
//! MultiOptPy: `interface.py:call_mdparser` and the use of the options in
//! `moleculardynamics.py:MD` (`mdtype`, `change_temperature`, `constraint_condition` with
//! `Constraint/constraint_condition.py:shake_parser`).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::path::Path;

use faer::Col;

use crate::core_infra::arg_parser::{parse_atom_indices_one_based, ParsedArgs};
use crate::domain::{AtomId, Units};
use crate::error::{MoptError, MoptResult};
use crate::md::barostat::gigapascal_per_atomic_unit;
use crate::md::settings::{
    kilopascal_to_atomic, ConstraintSolver, DEFAULT_PRESSURE_KPA, DEFAULT_TIME_STEP_FS,
    DEFAULT_VARIABLE_TIME_STEP,
};
use crate::md::{FreezeSettings, InitialVelocities, MdSettings, ModeSelection, RigidMotion};
use crate::pes::constraint::{Constraint, ConstraintKind};

/// The parts of an ensemble named by `-mt`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ensemble {
    /// Thermostat.
    pub thermostat: &'static str,
    /// Integrator.
    pub integrator: &'static str,
    /// Barostat (constant pressure).
    pub barostat: Option<&'static str>,
}

/// The ensemble without `-mt` (measured, see `docs/source/molecular_dynamics.md`).
pub const DEFAULT_ENSEMBLE: &str = "nosehooverchain";

/// The parts of `-mt NAME` (MultiOptPy's names first).
pub fn ensemble(name: &str) -> MoptResult<Ensemble> {
    let e = |thermostat, integrator, barostat| Ensemble {
        thermostat,
        integrator,
        barostat,
    };
    Ok(
        match name.trim().to_ascii_lowercase().replace('-', "_").as_str() {
            "nosehoover" | "nvt" | "nose_hoover" => e("nose_hoover", "velocity_verlet", None),
            "nosehooverchain" | "nose_hoover_chain" | "nhc" => {
                e("nose_hoover_chain", "velocity_verlet", None)
            }
            "langevin" | "baoab" => e("langevin", "baoab", None),
            "velocityverlet" | "velocity_verlet" | "nve" => e("none", "velocity_verlet", None),
            "berendsen" => e("berendsen", "velocity_verlet", None),
            "bussi" | "csvr" => e("bussi", "velocity_verlet", None),
            "andersen" => e("andersen", "velocity_verlet", None),
            "npt" | "npt_mtk" => e("nose_hoover_chain", "velocity_verlet", Some("mtk")),
            "npt_berendsen" => e("berendsen", "velocity_verlet", Some("berendsen")),
            "npt_berendsen_aniso" => e("berendsen", "velocity_verlet", Some("berendsen_aniso")),
            "npt_crescale" => e("bussi", "velocity_verlet", Some("crescale")),
            "npt_aniso" | "npt_mtk_aniso" => {
                e("nose_hoover_chain", "velocity_verlet", Some("mtk_aniso"))
            }
            "npt_crescale_aniso" => e("bussi", "velocity_verlet", Some("crescale_aniso")),
            other => {
                return Err(MoptError::parse(format!(
                    "-mt {other}: nosehoover (nvt), nosehooverchain, langevin (baoab), \
                 velocityverlet (nve), berendsen, bussi (csvr), andersen, npt (npt_mtk), \
                 npt_aniso (npt_mtk_aniso), npt_berendsen, npt_berendsen_aniso, npt_crescale, \
                 npt_crescale_aniso"
                )))
            }
        },
    )
}

/// `-ts`: atomic units, or femtoseconds with `fs` after the number.
pub fn parse_time_step(text: &str) -> MoptResult<f64> {
    let t = text.trim().to_ascii_lowercase();
    let (number, scale) = if let Some(n) = t.strip_suffix("fs") {
        (n, Units::FS_TO_AU_TIME)
    } else if let Some(n) = t.strip_suffix("au") {
        (n, 1.0)
    } else {
        (t.as_str(), 1.0)
    };
    let value: f64 = number
        .trim()
        .parse()
        .map_err(|e| MoptError::parse(format!("-ts '{text}': {e}")))?;
    if !(value.is_finite() && value > 0.0) {
        return Err(MoptError::parse(format!(
            "-ts '{text}': the step must be positive"
        )));
    }
    Ok(value * scale)
}

/// `-ct STEP K STEP K ...`.
pub fn parse_schedule(tokens: &[String]) -> MoptResult<Vec<(usize, f64)>> {
    if !tokens.len().is_multiple_of(2) {
        return Err(MoptError::parse(
            "-ct takes pairs of a step and a temperature (K), e.g. -ct 1000 500 5000 1000",
        ));
    }
    tokens
        .chunks(2)
        .map(|c| {
            let step = c[0]
                .trim()
                .parse::<f64>()
                .map_err(|e| MoptError::parse(format!("-ct step '{}': {e}", c[0])))?;
            let t = c[1]
                .trim()
                .parse::<f64>()
                .map_err(|e| MoptError::parse(format!("-ct temperature '{}': {e}", c[1])))?;
            if step < 0.0 || step.fract() != 0.0 {
                return Err(MoptError::parse(format!(
                    "-ct step '{}' is not a step",
                    c[0]
                )));
            }
            Ok((step as usize, t))
        })
        .collect()
}

/// `-cc VALUE,A,B[,C[,D]] ...`: distances (Angstrom), angles and dihedral angles (degrees).
pub fn parse_constraint_conditions(tokens: &[String]) -> MoptResult<Vec<Constraint>> {
    tokens
        .iter()
        .map(|t| {
            let parts: Vec<&str> = t.split(',').map(str::trim).collect();
            let value: f64 = parts[0]
                .parse()
                .map_err(|e| MoptError::parse(format!("-cc '{t}': {e}")))?;
            let atoms = parts[1..]
                .iter()
                .map(|a| {
                    let k = parse_atom_indices_one_based(a)?;
                    match k.as_slice() {
                        [one] => AtomId::from_one_based(*one),
                        _ => Err(MoptError::parse(format!("-cc '{t}': one atom per number"))),
                    }
                })
                .collect::<MoptResult<Vec<AtomId>>>()?;
            let (kind, target) = match atoms.as_slice() {
                [a, b] => (
                    ConstraintKind::Distance(vec![*a], vec![*b]),
                    value * Units::ANGSTROM_TO_BOHR,
                ),
                [a, b, c] => (
                    ConstraintKind::Angle([*a, *b, *c]),
                    value * Units::DEG_TO_RAD,
                ),
                [a, b, c, d] => (
                    ConstraintKind::Dihedral([*a, *b, *c, *d]),
                    value * Units::DEG_TO_RAD,
                ),
                _ => {
                    return Err(MoptError::parse(format!(
                        "-cc '{t}': a value and two, three, or four atoms"
                    )))
                }
            };
            Ok(Constraint {
                kind,
                target: Some(target),
                anchor: None,
            })
        })
        .collect()
}

/// The settings of `-md_opt` that are not settings of the state machine.
#[derive(Debug, Clone, PartialEq)]
pub struct RunOptions {
    /// Steps between frames of the trajectory.
    pub write_every: usize,
    /// Steps between reports in the log.
    pub print_every: usize,
    /// Steps between the states (structure, velocities, energies) in the log.
    pub structure_every: usize,
    /// `rigid=auto`: the conserved motions are decided from the surface.
    pub rigid_auto: bool,
    /// Whether the seed was given.
    pub seed_given: bool,
    /// Wrap molecules into the cell in the trajectory.
    pub wrap: bool,
    /// Where the dynamics of `-coord` is integrated.
    pub integrate: Integration,
    /// Write the heat current of every step (`heat_flux=on`).
    pub heat_flux: bool,
    /// The start from the Wigner distribution of the normal modes (`velocities=wigner`).
    pub wigner: Option<super::mdmain_wigner::WignerOption>,
}

/// `-md_opt integrate=`: where the dynamics of `-coord` is integrated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Integration {
    /// In Cartesian coordinates, the coordinates for the output (the default: in measured
    /// runs the dynamics in generalized coordinates conserved the energy exactly as well at
    /// 10 to 60 times the cost per step, see the documentation of mdmain).
    Cartesian,
    /// In the coordinates, or refused.
    Coordinates,
}

impl Integration {
    fn parse(value: &str) -> MoptResult<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "cartesian" | "cart" => Ok(Self::Cartesian),
            "coordinates" | "coord" | "generalized" => Ok(Self::Coordinates),
            other => Err(MoptError::parse(format!(
                "-md_opt integrate={other}: cartesian or coordinates"
            ))),
        }
    }
}

/// Steps between the states in the log without `structure_every` (those of the frames of
/// the trajectory).
pub const DEFAULT_STRUCTURE_EVERY: usize = DEFAULT_WRITE_EVERY;

/// Frames of the trajectory without `write_every` (as many as MultiOptPy keeps: it removes
/// the files of the steps that are not multiples of 10).
pub const DEFAULT_WRITE_EVERY: usize = 10;

/// A finite number of `-md_opt` (`NaN` and infinities refused, as by the other settings).
fn number(key: &str, value: &str) -> MoptResult<f64> {
    crate::core_infra::arg_parser::parse_finite(&format!("-md_opt {key}"), value)
}

fn count(key: &str, value: &str) -> MoptResult<usize> {
    value
        .trim()
        .parse::<usize>()
        .map_err(|e| MoptError::parse(format!("-md_opt {key}={value}: {e}")))
}

/// A switch of `-md_opt` (`core_infra::arg_parser::parse_switch`).
fn on(key: &str, value: &str) -> MoptResult<bool> {
    crate::core_infra::arg_parser::parse_switch(&format!("-md_opt {key}"), value)
}

/// `constraint_solver=shake|lincs`, `lincs_order=N`, `lincs_iterations=N`.
fn solver(current: ConstraintSolver, key: &str, value: &str) -> MoptResult<ConstraintSolver> {
    let (order, iterations) = match current {
        ConstraintSolver::Lincs { order, iterations } => (order, iterations),
        ConstraintSolver::Shake => (
            crate::md::lincs::DEFAULT_ORDER,
            crate::md::lincs::DEFAULT_ITERATIONS,
        ),
    };
    Ok(match key {
        "constraint_solver" => match value.trim().to_ascii_lowercase().as_str() {
            "shake" | "rattle" => ConstraintSolver::Shake,
            "lincs" => ConstraintSolver::Lincs { order, iterations },
            other => {
                return Err(MoptError::parse(format!(
                    "-md_opt constraint_solver={other}: shake or lincs"
                )))
            }
        },
        "lincs_order" => ConstraintSolver::Lincs {
            order: count(key, value)?.max(1),
            iterations,
        },
        _ => ConstraintSolver::Lincs {
            order,
            iterations: count(key, value)?,
        },
    })
}

/// `freeze=negative | below:VALUE | modes:0,1 | file:PATH`.
fn parse_freeze(value: &str) -> MoptResult<ModeSelection> {
    let (kind, rest) = value.split_once(':').unwrap_or((value, ""));
    match kind.trim().to_ascii_lowercase().as_str() {
        "negative" => Ok(ModeSelection::Negative),
        "below" => Ok(ModeSelection::Below(number("freeze", rest)?)),
        "modes" => Ok(ModeSelection::Indices(
            rest.split(',')
                .map(|k| count("freeze", k))
                .collect::<MoptResult<_>>()?,
        )),
        "file" => Ok(ModeSelection::Vectors(read_directions(Path::new(
            rest.trim(),
        ))?)),
        other => Err(MoptError::parse(format!(
            "-md_opt freeze={other}: negative, below:VALUE, modes:0,1, or file:PATH"
        ))),
    }
}

/// Cartesian directions, one per line (`#` comments).
pub fn read_directions(path: &Path) -> MoptResult<Vec<Col<f64>>> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| MoptError::config(format!("freeze file {}: {e}", path.display())))?;
    text.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|l| {
            let v = l
                .split([',', ' ', '\t'])
                .filter(|s| !s.is_empty())
                .map(|s| number("freeze", s))
                .collect::<MoptResult<Vec<f64>>>()?;
            Ok(Col::from_fn(v.len(), |i| v[i]))
        })
        .collect()
}

/// `hopping_states=0,1,...` (surface hopping on; the starting state is that of `-es`, set
/// by the command), `hopping_gap=EV`, `hopping_frustrated=keep|reverse`,
/// `hopping_forces=two_state|published`.
fn apply_hopping_key(md: &mut MdSettings, key: &str, value: &str) -> MoptResult<()> {
    use crate::md::hopping::{ForceModel, FrustratedHop, HoppingSettings};
    if key == "hopping_states" {
        let mut states: Vec<usize> = value
            .split(',')
            .filter(|t| !t.trim().is_empty())
            .map(|t| count(key, t))
            .collect::<MoptResult<_>>()?;
        states.sort_unstable();
        states.dedup();
        let start = states.first().copied().unwrap_or(0);
        let mut h = match md.hopping.take() {
            Some(h) => HoppingSettings { states, ..h },
            None => HoppingSettings::new(states, start)?,
        };
        if !h.states.contains(&h.start) {
            h.start = start;
        }
        md.hopping = Some(h);
        return Ok(());
    }
    let h = md.hopping.as_mut().ok_or_else(|| {
        MoptError::parse(format!(
            "-md_opt {key} needs hopping_states (the states of surface hopping) before it"
        ))
    })?;
    match key {
        "hopping_gap" => h.gap = number(key, value)? * Units::EV_TO_HARTREE,
        "hopping_frustrated" => {
            h.frustrated = match value.trim().to_ascii_lowercase().as_str() {
                "keep" => FrustratedHop::Keep,
                "reverse" => FrustratedHop::Reverse,
                other => {
                    return Err(MoptError::parse(format!(
                        "-md_opt hopping_frustrated={other}: keep or reverse"
                    )))
                }
            }
        }
        _ => {
            h.forces = match value.trim().to_ascii_lowercase().as_str() {
                "two_state" => ForceModel::TwoState,
                "published" => ForceModel::Published,
                other => {
                    return Err(MoptError::parse(format!(
                        "-md_opt hopping_forces={other}: two_state or published"
                    )))
                }
            }
        }
    }
    Ok(())
}

/// Apply `-md_opt KEY=VALUE` to `md`.
pub fn apply_md_options(tokens: &[String], md: &mut MdSettings) -> MoptResult<RunOptions> {
    let mut run = RunOptions {
        write_every: DEFAULT_WRITE_EVERY,
        print_every: 1,
        structure_every: DEFAULT_STRUCTURE_EVERY,
        rigid_auto: true,
        seed_given: false,
        wrap: true,
        integrate: Integration::Cartesian,
        heat_flux: false,
        wigner: None,
    };
    let mut exchange: Option<crate::md::settings::Exchange> = None;
    let exchange_or_default = |e: &mut Option<crate::md::settings::Exchange>| {
        *e.get_or_insert(crate::md::settings::Exchange {
            every: 50,
            slabs: 20,
            axis: 2,
        })
    };
    let mut freeze_update = None;
    let mut selection = None;
    let ps = 1e-3 * Units::AU_TIME_TO_FS;
    for token in tokens {
        let (key, value) = token
            .split_once('=')
            .ok_or_else(|| MoptError::parse(format!("-md_opt expects KEY=VALUE, got '{token}'")))?;
        let key = key.trim().to_ascii_lowercase();
        match key.as_str() {
            "thermostat" => md.thermostat = value.trim().to_string(),
            "integrator" => md.integrator = value.trim().to_string(),
            "barostat" => {
                md.barostat = match value.trim().to_ascii_lowercase().as_str() {
                    "none" | "off" => None,
                    other => Some(other.to_string()),
                }
            }
            "tau" => md.thermostat_settings.tau = number(&key, value)? * Units::FS_TO_AU_TIME,
            "friction" => md.thermostat_settings.friction = Some(number(&key, value)? * ps),
            "collision" => md.thermostat_settings.collision = Some(number(&key, value)? * ps),
            "chain" => md.thermostat_settings.chain = count(&key, value)?,
            "yoshida" => md.thermostat_settings.yoshida = count(&key, value)?,
            "substeps" => md.thermostat_settings.substeps = count(&key, value)?,
            "tau_p" => md.barostat_settings.tau = number(&key, value)? * Units::FS_TO_AU_TIME,
            "compressibility" => {
                md.barostat_settings.compressibility =
                    number(&key, value)? * gigapascal_per_atomic_unit()
            }
            "seed" => {
                md.seed = value
                    .trim()
                    .parse::<u64>()
                    .map_err(|e| MoptError::parse(format!("-md_opt seed={value}: {e}")))?;
                run.seed_given = true;
            }
            "velocities" => {
                md.velocities = match value.trim().to_ascii_lowercase().as_str() {
                    "exact" => InitialVelocities::Maxwell { exact: true },
                    "maxwell" => InitialVelocities::Maxwell { exact: false },
                    "zero" => InitialVelocities::Zero,
                    "wigner" => {
                        run.wigner.get_or_insert_with(Default::default);
                        InitialVelocities::Maxwell { exact: true }
                    }
                    other => {
                        return Err(MoptError::parse(format!(
                            "-md_opt velocities={other}: exact, maxwell, zero, or wigner"
                        )))
                    }
                }
            }
            k if k.starts_with("wigner_") => {
                let w = run.wigner.get_or_insert_with(Default::default);
                if !super::mdmain_wigner::apply_wigner_key(w, k, value)? {
                    return Err(MoptError::parse(format!(
                        "-md_opt: unknown setting '{k}' (wigner_temperature, wigner_low, \
                         wigner_imaginary)"
                    )));
                }
            }
            "rigid" => {
                run.rigid_auto = value.trim().eq_ignore_ascii_case("auto");
                if !run.rigid_auto {
                    md.rigid = RigidMotion::parse(value)?;
                }
            }
            "write_every" => run.write_every = count(&key, value)?.max(1),
            "print_every" => run.print_every = count(&key, value)?.max(1),
            "structure_every" => run.structure_every = count(&key, value)?.max(1),
            "freeze" => selection = Some(parse_freeze(value)?),
            "freeze_update" => {
                freeze_update = match value.trim().to_ascii_lowercase().as_str() {
                    "none" | "off" | "0" => None,
                    _ => Some(count(&key, value)?),
                }
            }
            "shake_tolerance" => md.shake_tolerance = number(&key, value)?,
            "wrap" => run.wrap = on(&key, value)?,
            "integrate" => run.integrate = Integration::parse(value)?,
            "stress" => md.stress = on(&key, value)?,
            "constraint_solver" | "lincs_order" | "lincs_iterations" => {
                md.constraint_solver = solver(md.constraint_solver, &key, value)?;
            }
            "heat_flux" => run.heat_flux = on(&key, value)?,
            "hopping_states" | "hopping_gap" | "hopping_frustrated" | "hopping_forces" => {
                apply_hopping_key(md, &key, value)?;
            }
            "exchange" => {
                let mut e = exchange_or_default(&mut exchange);
                e.every = count(&key, value)?.max(1);
                exchange = Some(e);
            }
            "exchange_slabs" => {
                let mut e = exchange_or_default(&mut exchange);
                e.slabs = count(&key, value)?;
                exchange = Some(e);
            }
            "exchange_axis" => {
                let mut e = exchange_or_default(&mut exchange);
                e.axis = match value.trim().to_ascii_lowercase().as_str() {
                    "a" | "x" | "0" => 0,
                    "b" | "y" | "1" => 1,
                    "c" | "z" | "2" => 2,
                    other => {
                        return Err(MoptError::parse(format!(
                            "-md_opt exchange_axis={other}: a, b, or c"
                        )))
                    }
                };
                exchange = Some(e);
            }
            other => {
                return Err(MoptError::parse(format!(
                    "-md_opt: unknown setting '{other}' (thermostat, integrator, barostat, tau, \
                     friction, collision, chain, yoshida, substeps, tau_p, compressibility, seed, \
                     velocities, rigid, write_every, print_every, structure_every, freeze, freeze_update, \
                     shake_tolerance, wrap, integrate, stress, heat_flux, exchange, \
                     exchange_slabs, exchange_axis, constraint_solver, lincs_order, \
                     lincs_iterations, wigner_temperature, wigner_low, wigner_imaginary, \
                     hopping_states, hopping_gap, hopping_frustrated, hopping_forces)"
                )))
            }
        }
    }
    md.exchange = exchange;
    if freeze_update.is_some() && selection.is_none() {
        return Err(MoptError::parse("-md_opt freeze_update needs freeze"));
    }
    md.freeze = selection.map(|selection| FreezeSettings {
        selection,
        update_interval: freeze_update,
    });
    Ok(run)
}

/// The settings of the dynamics from the options of `mdmain` (`variables`: an objective of
/// plain variables, whose default time step is that of unit masses).
pub fn md_settings(p: &ParsedArgs, variables: bool) -> MoptResult<(MdSettings, RunOptions)> {
    // Without -mt: a Nose-Hoover chain (MultiOptPy: one Nose-Hoover thermostat, which does
    // not sample the canonical ensemble of small molecules; see docs/source/molecular_dynamics.md).
    let mdtype = match p.get_opt_str("mdtype") {
        Some(name) if p.is_present("mdtype") => name,
        _ => DEFAULT_ENSEMBLE,
    };
    let parts = ensemble(mdtype)?;
    let time_step = match p.get_opt_str("timestep") {
        Some(text) if p.is_present("timestep") => parse_time_step(text)?,
        _ if variables => DEFAULT_VARIABLE_TIME_STEP,
        _ => DEFAULT_TIME_STEP_FS * Units::FS_TO_AU_TIME,
    };
    let mut md = MdSettings {
        time_step,
        steps: p.get_usize("NSTEP", 100_000)?,
        temperature: p.get_f64("temperature", 298.15)?,
        schedule: parse_schedule(p.get_list("change_temperature"))?,
        integrator: parts.integrator.to_string(),
        thermostat: parts.thermostat.to_string(),
        barostat: parts.barostat.map(str::to_string),
        pressure: kilopascal_to_atomic(p.get_f64("pressure", DEFAULT_PRESSURE_KPA)?),
        constraints: super::constraints::parse_constraints(p)?,
        ..MdSettings::default()
    };
    md.constraints.extend(parse_constraint_conditions(
        p.get_list("constraint_condition"),
    )?);
    let run = apply_md_options(p.get_list("md_options"), &mut md)?;
    md.validate()?;
    Ok((md, run))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn the_options_become_settings() {
        assert_eq!(ensemble("nosehoover").unwrap().thermostat, "nose_hoover");
        assert_eq!(ensemble("NVE").unwrap().thermostat, "none");
        assert_eq!(ensemble("baoab").unwrap().integrator, "baoab");
        assert_eq!(ensemble("npt").unwrap().barostat, Some("mtk"));
        assert!(ensemble("nph").is_err());
        assert_eq!(parse_time_step("20").unwrap(), 20.0);
        assert!((parse_time_step("0.5 fs").unwrap() - 20.6707).abs() < 1e-3);
        assert!(parse_time_step("-1").is_err());
        assert_eq!(
            parse_schedule(&s(&["1000", "500", "5000", "1000"])).unwrap(),
            vec![(1000, 500.0), (5000, 1000.0)]
        );
        assert!(parse_schedule(&s(&["1000"])).is_err());
        let cc = parse_constraint_conditions(&s(&["1.5,1,2", "109.5,1,2,3"])).unwrap();
        assert!((cc[0].target.unwrap() - 1.5 * Units::ANGSTROM_TO_BOHR).abs() < 1e-12);
        assert!(matches!(cc[1].kind, ConstraintKind::Angle(_)));
        assert!(parse_constraint_conditions(&s(&["1.5,1"])).is_err());
        let mut md = MdSettings::default();
        let run = apply_md_options(
            &s(&[
                "thermostat=bussi",
                "tau=50",
                "seed=12",
                "freeze=modes:0,2",
                "write_every=5",
            ]),
            &mut md,
        )
        .unwrap();
        assert_eq!(md.thermostat, "bussi");
        assert!((md.thermostat_settings.tau - 50.0 * Units::FS_TO_AU_TIME).abs() < 1e-9);
        assert!(run.seed_given && md.seed == 12 && run.write_every == 5);
        assert_eq!(
            md.freeze.unwrap().selection,
            ModeSelection::Indices(vec![0, 2])
        );
        assert!(apply_md_options(&s(&["colour=red"]), &mut MdSettings::default()).is_err());
        let mut md = MdSettings::default();
        apply_md_options(
            &s(&[
                "hopping_states=2,1",
                "hopping_gap=0.5",
                "hopping_frustrated=reverse",
            ]),
            &mut md,
        )
        .unwrap();
        let h = md.hopping.unwrap();
        assert_eq!(h.states, vec![1, 2]);
        assert!((h.gap - 0.5 * Units::EV_TO_HARTREE).abs() < 1e-15);
        assert_eq!(h.frustrated, crate::md::hopping::FrustratedHop::Reverse);
        assert!(apply_md_options(&s(&["hopping_gap=0.5"]), &mut MdSettings::default()).is_err());
        assert!(apply_md_options(&s(&["hopping_states=1"]), &mut MdSettings::default()).is_err());
        assert!(apply_md_options(&s(&["freeze_update=10"]), &mut MdSettings::default()).is_err());
    }
}
