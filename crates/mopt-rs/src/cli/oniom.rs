//! `-oniom HIGH_ATOMS LINK LOW_CALCULATOR`, `-oniom_middle ATOMS LINK CALCULATOR`, and
//! `-oniom_opt KEY=VALUE ...`: the ONIOM calculation of the calculator options, the same for
//! every command ([`crate::pes::oniom`]).
//!
//! MultiOptPy: `interface.py` (`-oniom`, read into `force_data["oniom_flag"]` as
//! `[high_layer, link_atoms, low_layer_model]`).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use crate::core_infra::arg_parser::{parse_atom_indices_one_based, ParsedArgs};
use crate::domain::{AtomId, SpinCharge};
use crate::error::{MoptError, MoptResult};
use crate::pes::microiter::InnerMethod;
use crate::pes::oniom::embedding::{BoundaryCharges, ChargeSource, ElectronicEmbedding};
use crate::pes::oniom::micro::{CoreHessian, EnvironmentCoordinates, ReplicaStart, ResponseSolve};
use crate::pes::oniom::{LinkPlacement, LinkSpec, MicroOptions, OniomSpec, RegionSpec};

fn atoms(text: &str, what: &str) -> MoptResult<Vec<AtomId>> {
    parse_atom_indices_one_based(text)
        .map_err(|e| MoptError::parse(format!("{what}: {e}")))?
        .into_iter()
        .map(AtomId::from_one_based)
        .collect()
}

/// Built-in calculators as MultiOptPy writes them (`GFN-FF`, `GFN0-xTB`) by their names.
fn calculator_name(text: &str) -> String {
    let squeezed = text.to_ascii_lowercase().replace(['-', '_'], "");
    match squeezed.as_str() {
        "gfnff" | "gfn0xtb" | "gfn1xtb" | "lj" | "emt" | "eam" | "meam" | "tersoff" => squeezed,
        _ => text.to_string(),
    }
}

/// One region: `ATOMS LINK CALCULATOR`.
fn region(tokens: &[String], what: &str) -> MoptResult<(RegionSpec, String)> {
    let [list, link, calc] = tokens else {
        return Err(MoptError::parse(format!(
            "{what} takes ATOMS LINK CALCULATOR (e.g. {what} 1-6 auto gfnff), got {} value(s)",
            tokens.len()
        )));
    };
    let links = match link.to_ascii_lowercase().as_str() {
        "none" => LinkSpec::None,
        "auto" => LinkSpec::Auto,
        _ => LinkSpec::Given(atoms(link, what)?),
    };
    Ok((
        RegionSpec {
            atoms: atoms(list, what)?,
            links,
            spin_charge: None,
        },
        calculator_name(calc),
    ))
}

/// The most replicas of the environment (`micro_replicas`): each relaxes at every core.
const MAX_REPLICAS: u64 = 64;

/// One of the settings of the relaxations of the environment.
fn micro_setting(m: &mut MicroOptions, key: &str, value: &str) -> MoptResult<()> {
    let lower = value.to_ascii_lowercase();
    let on_off = || match lower.as_str() {
        "on" | "true" | "yes" => Ok(true),
        "off" | "false" | "no" => Ok(false),
        _ => Err(MoptError::parse(format!(
            "-oniom_opt {key}={value}: on or off"
        ))),
    };
    match key {
        "micro_hessian" => {
            m.core_hessian = match lower.as_str() {
                "effective" => CoreHessian::Effective,
                "approximate" => CoreHessian::Approximate,
                "held" => CoreHessian::Held,
                _ => {
                    return Err(MoptError::parse(format!(
                        "-oniom_opt micro_hessian={value}: effective, approximate, or held"
                    )))
                }
            }
        }
        "micro_optimizer" => {
            if value.trim().is_empty() {
                return Err(MoptError::parse(
                    "-oniom_opt micro_optimizer: lbfgs, tn, or a method of -opt",
                ));
            }
            m.optimizer = InnerMethod::parse(value);
        }
        "micro_coord" => {
            m.coordinates = match lower.as_str() {
                "cart" | "cartesian" => EnvironmentCoordinates::Cartesian,
                "fragments" | "fragment" | "hdlc" => EnvironmentCoordinates::Fragments,
                _ => {
                    return Err(MoptError::parse(format!(
                        "-oniom_opt micro_coord={value}: cart or fragments"
                    )))
                }
            }
        }
        "micro_predict" => m.predict = on_off()?,
        "micro_correct" => m.correct = on_off()?,
        "micro_hessian_solve" => {
            m.response_solve = match lower.as_str() {
                "auto" => ResponseSolve::Auto,
                "dense" => ResponseSolve::Dense,
                "cg" | "conjugate_gradients" => ResponseSolve::ConjugateGradients,
                _ => {
                    return Err(MoptError::parse(format!(
                        "-oniom_opt micro_hessian_solve={value}: auto, dense, or cg"
                    )))
                }
            }
        }
        "micro_tolerance_factor" | "micro_core_shell" | "micro_check_tolerance" => {
            let number: f64 = value
                .parse()
                .map_err(|_| MoptError::parse(format!("-oniom_opt {key}={value}: not a number")))?;
            if key == "micro_check_tolerance" {
                if !(number.is_finite() && number > 0.0) {
                    return Err(MoptError::parse(format!(
                        "-oniom_opt {key}={value}: the largest gradient component of the \
                         environment for the Hessians of the checks of stability, in \
                         Hartree/Bohr, above 0"
                    )));
                }
                m.check_tolerance = Some(number);
            } else if key == "micro_tolerance_factor" {
                if !(number.is_finite() && number > 0.0 && number <= 1.0) {
                    return Err(MoptError::parse(format!(
                        "-oniom_opt {key}={value}: a share of the thresholds of the forces of the \
                         run, above 0 and at most 1"
                    )));
                }
                m.tolerance_factor = number;
            } else {
                if !(number.is_finite() && number >= 0.0) {
                    return Err(MoptError::parse(format!(
                        "-oniom_opt {key}={value}: a distance in Angstrom, 0 or more"
                    )));
                }
                m.core_shell = (number > 0.0).then_some(number);
            }
        }
        "micro_precondition" => {
            m.precondition = match lower.as_str() {
                "none" | "off" | "no" => None,
                _ => {
                    crate::model_hessian::resolve(value)
                        .map_err(|e| e.context(format!("-oniom_opt micro_precondition={value}")))?;
                    Some(value.trim().to_string())
                }
            }
        }
        "micro_hessian_memory" | "micro_hessian_shell" => {
            let number: f64 = value
                .parse()
                .map_err(|_| MoptError::parse(format!("-oniom_opt {key}={value}: not a number")))?;
            if !(number.is_finite() && number > 0.0) {
                let unit = if key == "micro_hessian_memory" {
                    "GiB"
                } else {
                    "Angstrom"
                };
                return Err(MoptError::parse(format!(
                    "-oniom_opt {key}={value}: a positive number ({unit})"
                )));
            }
            if key == "micro_hessian_memory" {
                m.hessian_memory = Some(number);
            } else {
                m.hessian_shell = Some(number);
            }
        }
        "micro_precondition_shift" => {
            let shift: f64 = value
                .parse()
                .map_err(|_| MoptError::parse(format!("-oniom_opt {key}={value}: not a number")))?;
            if !(shift.is_finite() && shift > 0.0) {
                return Err(MoptError::parse(format!(
                    "-oniom_opt {key}={value}: a positive number"
                )));
            }
            m.precondition_shift = shift;
        }
        "micro_replicas" | "micro_replica_seed" | "micro_replica_md_steps" => {
            let number: u64 = value.trim().parse().map_err(|_| {
                MoptError::parse(format!("-oniom_opt {key}={value}: a whole number"))
            })?;
            match key {
                "micro_replicas" => {
                    if !(1..=MAX_REPLICAS).contains(&number) {
                        return Err(MoptError::parse(format!(
                            "-oniom_opt micro_replicas={value}: 1 (one environment) to \
                             {MAX_REPLICAS} replicas of the environment"
                        )));
                    }
                    m.replicas.count = number as usize;
                }
                "micro_replica_seed" => m.replicas.seed = number,
                _ => {
                    if number == 0 {
                        return Err(MoptError::parse(format!(
                            "-oniom_opt {key}={value}: at least one step"
                        )));
                    }
                    m.replicas.dynamics_steps = number as usize;
                }
            }
        }
        "micro_replica_temperature" | "micro_replica_amplitude" => {
            let number: f64 = value
                .parse()
                .map_err(|_| MoptError::parse(format!("-oniom_opt {key}={value}: not a number")))?;
            if !(number.is_finite() && number > 0.0) {
                return Err(MoptError::parse(format!(
                    "-oniom_opt {key}={value}: {} above 0",
                    if key == "micro_replica_temperature" {
                        "a temperature in K"
                    } else {
                        "a displacement in Angstrom"
                    }
                )));
            }
            if key == "micro_replica_temperature" {
                m.replicas.temperature = number;
            } else {
                m.replicas.amplitude = number * crate::domain::Units::ANGSTROM_TO_BOHR;
            }
        }
        "micro_replica_start" => {
            m.replicas.start = match lower.as_str() {
                "perturb" => ReplicaStart::Perturb,
                "md" => ReplicaStart::Dynamics,
                // The file comes with micro_replica_files.
                "files" => match &m.replicas.start {
                    ReplicaStart::Files(p) => ReplicaStart::Files(p.clone()),
                    _ => ReplicaStart::Files(std::path::PathBuf::new()),
                },
                _ => {
                    return Err(MoptError::parse(format!(
                        "-oniom_opt micro_replica_start={value}: perturb, md, or files"
                    )))
                }
            }
        }
        "micro_replica_files" => {
            if value.trim().is_empty() {
                return Err(MoptError::parse(
                    "-oniom_opt micro_replica_files: a file of whole structures, one per replica",
                ));
            }
            m.replicas.start = ReplicaStart::Files(std::path::PathBuf::from(value.trim()));
        }
        "micro_curvature" => {
            m.guided_curvature = match lower.as_str() {
                "guided" => true,
                "plain" => false,
                _ => {
                    return Err(MoptError::parse(format!(
                        "-oniom_opt micro_curvature={value}: guided or plain"
                    )))
                }
            }
        }
        _ => {
            let eta: f64 = value
                .parse()
                .map_err(|_| MoptError::parse(format!("-oniom_opt {key}={value}: not a number")))?;
            if !(0.0..=1.0).contains(&eta) {
                return Err(MoptError::parse(format!(
                    "-oniom_opt micro_adaptive={value}: a share between 0 (off) and 1"
                )));
            }
            m.adaptive = eta;
        }
    }
    if m.coordinates == EnvironmentCoordinates::Fragments && m.optimizer != InnerMethod::Lbfgs {
        return Err(MoptError::parse(format!(
            "-oniom_opt micro_coord=fragments is for micro_optimizer=lbfgs, not '{}'",
            m.optimizer.name()
        )));
    }
    Ok(())
}

/// The ONIOM spec of the command line, `None` without `-oniom`.
pub fn oniom_spec(p: &ParsedArgs) -> MoptResult<Option<OniomSpec>> {
    let main = p.get_list("oniom_flag");
    let middle = p.get_list("oniom_middle");
    let options = p.get_list("oniom_options");
    if main.is_empty() {
        if !middle.is_empty() || !options.is_empty() {
            return Err(MoptError::parse(
                "-oniom_middle and -oniom_opt belong to -oniom HIGH_ATOMS LINK LOW_CALCULATOR",
            ));
        }
        return Ok(None);
    }
    let (high, low) = region(main, "-oniom")?;
    let mut regions = vec![high];
    let mut lower_levels = Vec::new();
    if !middle.is_empty() {
        let (mid, calc) = region(middle, "-oniom_middle")?;
        regions.push(mid);
        lower_levels.push(calc);
    }
    lower_levels.push(low);
    let mut lower_options: Vec<Vec<(String, String)>> = vec![Vec::new(); lower_levels.len()];
    let mut traj_high_only = false;
    let mut placement = LinkPlacement::default();
    // Microiterations are on with ONIOM unless turned off.
    let mut micro: Option<MicroOptions> = Some(MicroOptions::default());
    // Mechanical embedding unless electronic is asked for.
    let mut electronic = false;
    let mut boundary = BoundaryCharges::default();
    let mut charges = ChargeSource::Lowest;
    let mut embedding_keys = Vec::new();
    // Charges and multiplicities of the model systems: (charge, multiplicity) per region.
    let mut states: Vec<(Option<i32>, Option<u32>)> = vec![(None, None); regions.len()];
    for item in options {
        let (key, value) = item
            .split_once('=')
            .ok_or_else(|| MoptError::parse(format!("-oniom_opt takes KEY=VALUE, not '{item}'")))?;
        // LAYER.KEY: a setting of the calculator of a lower level, its name as written.
        if let Some((layer, name)) = key.trim().split_once('.') {
            let level =
                match layer.to_ascii_lowercase().as_str() {
                    "low" => lower_levels.len() - 1,
                    "middle" if regions.len() > 1 => 0,
                    "middle" => {
                        return Err(MoptError::parse(
                            "-oniom_opt middle.*: there is no middle layer (-oniom_middle)",
                        ))
                    }
                    "high" => return Err(MoptError::parse(
                        "-oniom_opt high.*: the highest level takes its settings from -calc_opt, \
                         -func, and -bs",
                    )),
                    other => {
                        return Err(MoptError::parse(format!(
                            "-oniom_opt: unknown layer '{other}' (low.KEY=VALUE, middle.KEY=VALUE)"
                        )))
                    }
                };
            lower_options[level].push((name.trim().to_string(), value.trim().to_string()));
            continue;
        }
        let (key, value) = (key.trim().to_ascii_lowercase(), value.trim());
        let bad = || MoptError::parse(format!("-oniom_opt {key}={value}: not a number"));
        let slot = |name: &str| -> MoptResult<usize> {
            match name {
                "high" => Ok(0),
                "middle" if regions.len() > 1 => Ok(1),
                "middle" => Err(MoptError::parse(
                    "-oniom_opt middle_*: there is no middle layer (-oniom_middle)",
                )),
                _ => Err(MoptError::parse(format!(
                    "-oniom_opt: unknown layer '{name}'"
                ))),
            }
        };
        match key.as_str() {
            "traj_high_only" => {
                traj_high_only = match value.to_ascii_lowercase().as_str() {
                    "on" | "true" | "yes" => true,
                    "off" | "false" | "no" => false,
                    _ => {
                        return Err(MoptError::parse(format!(
                            "-oniom_opt {key}={value}: on or off"
                        )))
                    }
                };
            }
            "link" => placement = LinkPlacement::parse(value)?,
            "embedding" => {
                electronic = match value.to_ascii_lowercase().as_str() {
                    "electronic" | "ee" => true,
                    "mechanical" | "me" => false,
                    _ => {
                        return Err(MoptError::parse(format!(
                            "-oniom_opt embedding={value}: mechanical or electronic"
                        )))
                    }
                };
            }
            "boundary" => {
                boundary = BoundaryCharges::parse(value)?;
                embedding_keys.push("boundary");
            }
            "charges" => {
                charges = if value.eq_ignore_ascii_case("lowest") {
                    ChargeSource::Lowest
                } else {
                    ChargeSource::Given(read_charges(value)?)
                };
                embedding_keys.push("charges");
            }
            "micro" => {
                micro = match value.to_ascii_lowercase().as_str() {
                    "on" | "true" | "yes" => Some(MicroOptions {
                        requested: true,
                        ..micro.take().unwrap_or_default()
                    }),
                    "off" | "false" | "no" => None,
                    _ => {
                        return Err(MoptError::parse(format!(
                            "-oniom_opt micro={value}: on or off"
                        )))
                    }
                };
            }
            "micro_hessian"
            | "micro_optimizer"
            | "micro_coord"
            | "micro_predict"
            | "micro_correct"
            | "micro_hessian_solve"
            | "micro_tolerance_factor"
            | "micro_check_tolerance"
            | "micro_core_shell"
            | "micro_adaptive"
            | "micro_curvature"
            | "micro_precondition"
            | "micro_precondition_shift"
            | "micro_hessian_memory"
            | "micro_hessian_shell"
            | "micro_replicas"
            | "micro_replica_temperature"
            | "micro_replica_start"
            | "micro_replica_files"
            | "micro_replica_amplitude"
            | "micro_replica_seed"
            | "micro_replica_md_steps" => {
                let Some(mut m) = micro.take() else {
                    return Err(MoptError::parse(format!(
                        "-oniom_opt {key}: the microiterations are off (micro=off)"
                    )));
                };
                micro_setting(&mut m, &key, value)?;
                m.requested = true;
                micro = Some(m);
            }
            "micro_mode_kill" | "micro_tolerance" => {
                let Some(mut m) = micro.take() else {
                    return Err(MoptError::parse(format!(
                        "-oniom_opt {key}: the microiterations are off (micro=off)"
                    )));
                };
                if key == "micro_mode_kill" {
                    if value.trim().eq_ignore_ascii_case("off") {
                        m.environment_mode_kill = false;
                    } else {
                        m.environment_mode_kill = true;
                        m.mode_kill_every = value.parse().map_err(|_| bad())?;
                    }
                } else {
                    m.tolerance = Some(value.parse().map_err(|_| bad())?);
                }
                m.requested = true;
                micro = Some(m);
            }
            _ => match key.split_once('_') {
                Some((layer, "charge")) => {
                    states[slot(layer)?].0 = Some(value.parse().map_err(|_| bad())?);
                }
                Some((layer, "multiplicity")) => {
                    states[slot(layer)?].1 = Some(value.parse().map_err(|_| bad())?);
                }
                _ => {
                    return Err(MoptError::parse(format!(
                        "-oniom_opt: unknown setting '{key}' (link, high_charge, \
                         high_multiplicity, middle_charge, middle_multiplicity, embedding, \
                         boundary, charges, micro, micro_mode_kill, micro_tolerance, \
                         micro_tolerance_factor, micro_check_tolerance, micro_correct, \
                         micro_core_shell, micro_hessian_solve, \
                         micro_hessian, micro_optimizer, micro_coord, micro_predict, \
                         micro_adaptive, micro_curvature, micro_precondition, \
                         micro_precondition_shift, micro_hessian_memory, \
                         micro_hessian_shell, micro_replicas, micro_replica_temperature, \
                         micro_replica_start, micro_replica_files, micro_replica_amplitude, \
                         micro_replica_seed, micro_replica_md_steps, traj_high_only)"
                    )))
                }
            },
        }
    }
    if let Some(m) = &micro {
        if matches!(&m.replicas.start, ReplicaStart::Files(p) if p.as_os_str().is_empty()) {
            return Err(MoptError::parse(
                "-oniom_opt micro_replica_start=files: give the file of whole structures, one \
                 per replica (micro_replica_files=FILE)",
            ));
        }
    }
    for (region, state) in regions.iter_mut().zip(&states) {
        if let (Some(charge), Some(multiplicity)) = state {
            region.spin_charge = Some(SpinCharge::new(*charge, *multiplicity));
        } else if state.0.is_some() || state.1.is_some() {
            return Err(MoptError::parse(
                "-oniom_opt: give the charge and the multiplicity of a model system together",
            ));
        }
    }
    let embedding = if electronic {
        Some(ElectronicEmbedding { charges, boundary })
    } else if let Some(key) = embedding_keys.first() {
        return Err(MoptError::parse(format!(
            "-oniom_opt {key}: a setting of the electronic embedding (embedding=electronic)"
        )));
    } else {
        None
    };
    Ok(Some(OniomSpec {
        traj_high_only,
        regions,
        lower_levels,
        lower_options,
        placement,
        micro,
        embedding,
    }))
}

/// The charges of a file: one per atom of the real system, separated by white space (e).
fn read_charges(path: &str) -> MoptResult<Vec<f64>> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| MoptError::parse(format!("-oniom_opt charges={path}: {e}")))?;
    text.split_whitespace()
        .map(|t| {
            t.parse::<f64>().map_err(|_| {
                MoptError::parse(format!("-oniom_opt charges={path}: '{t}' is not a number"))
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    #[test]
    fn the_settings_of_the_response_and_the_tolerance_are_read() {
        let mut m = MicroOptions::default();
        assert_eq!(m.response_solve, ResponseSolve::Auto);
        assert!(m.correct);
        assert_eq!(
            m.tolerance_factor,
            crate::pes::oniom::MICRO_TOLERANCE_FACTOR
        );
        micro_setting(&mut m, "micro_hessian_solve", "dense").unwrap();
        assert_eq!(m.response_solve, ResponseSolve::Dense);
        micro_setting(&mut m, "micro_hessian_solve", "cg").unwrap();
        assert_eq!(m.response_solve, ResponseSolve::ConjugateGradients);
        assert!(micro_setting(&mut m, "micro_hessian_solve", "lu").is_err());
        micro_setting(&mut m, "micro_correct", "off").unwrap();
        assert!(!m.correct);
        micro_setting(&mut m, "micro_tolerance_factor", "0.3").unwrap();
        assert_eq!(m.tolerance_factor, 0.3);
        assert!(micro_setting(&mut m, "micro_tolerance_factor", "2").is_err());
        assert!(micro_setting(&mut m, "micro_tolerance_factor", "0").is_err());
        micro_setting(&mut m, "micro_core_shell", "3.5").unwrap();
        assert_eq!(m.core_shell, Some(3.5));
        micro_setting(&mut m, "micro_core_shell", "0").unwrap();
        assert_eq!(m.core_shell, None);
        assert!(micro_setting(&mut m, "micro_core_shell", "-1").is_err());
        // The relaxations go to the share of the thresholds of the run.
        let s = m.settings_for((3e-4, 2e-4));
        assert!((s.inner.gradient_tolerance - 9e-5).abs() < 1e-15);
        assert!((s.inner.rms_tolerance - 6e-5).abs() < 1e-15);
        assert!(!s.correct);
        // Not below the floor of a lowest level in double precision, unless the thresholds
        // of the run are below it themselves.
        let d = MicroOptions {
            tolerance_factor: 0.1,
            ..MicroOptions::default()
        };
        let tight = d.settings_for((1.5e-5, 1e-5));
        assert_eq!(
            tight.inner.gradient_tolerance,
            crate::pes::oniom::MICRO_TOLERANCE_FLOOR
        );
        let tighter = d.settings_for((5e-6, 3e-6));
        assert_eq!(tighter.inner.gradient_tolerance, 5e-6);
        assert_eq!(tighter.inner.rms_tolerance, 3e-6);
    }

    #[test]
    fn every_check_of_stability_has_one_tolerance_of_the_environment() {
        // The relaxations of an optimization go to 3e-4, those of an IRC to 1.5e-5: the
        // Hessians of their checks both take the environment to 1e-5.
        let m = MicroOptions::default();
        let check = crate::pes::oniom::MICRO_CHECK_TOLERANCE;
        for criteria in [(3e-4, 2e-4), (1.5e-5, 1e-5)] {
            let s = m.settings_for(criteria);
            assert_eq!(s.check_tolerance, Some((check, check * 2.0 / 3.0)));
        }
        // Never looser than the relaxations of the run, and as given.
        let s = m.settings_for((5e-6, 3e-6));
        assert_eq!(s.check_tolerance, Some((5e-6, 3e-6)));
        let mut given = MicroOptions::default();
        micro_setting(&mut given, "micro_check_tolerance", "3e-6").unwrap();
        let s = given.settings_for((3e-4, 2e-4));
        assert_eq!(s.check_tolerance, Some((3e-6, 3e-6 * 2.0 / 3.0)));
        assert!(micro_setting(&mut given, "micro_check_tolerance", "0").is_err());
        assert!(micro_setting(&mut given, "micro_check_tolerance", "x").is_err());
    }
}
