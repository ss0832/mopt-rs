//! `ieipmain -coord`: the coordinates the searches move in.
//!
//! Every method moves in the coordinates of `-coord` (any independent system of the
//! coordinate registry: `dic`, `tric`, `tric-cart[:WEIGHT]`, ...): its points are the vectors
//! of the coordinates of structures relative to one reference structure (the midpoint of the
//! two inputs, or the one input) ([`crate::workflow::pair_search::internal`]). iEIP and
//! DS-AFIR keep their distances and forces in Cartesian coordinates, as their papers define
//! them, and take only their steps through the coordinates (redundant systems such as `ric`
//! serve there too).
//!
//! Where a system cannot be set up, or breaks down during the run (a back-transformation
//! that fails, or the system asking to be set up anew), it is set up anew at the structure
//! reached and the search starts again from the inputs; when it breaks down again, the next
//! system takes over (`tric` -> `tric-cart` -> `dic` -> Cartesian; `dic` -> Cartesian), with
//! a warning each time, and the log ends with what happened ([`CoordinatePlan`]). Refused
//! (exit code 2), where a method has no formulation in the coordinates: fixed atoms and
//! constraints (defined on Cartesian coordinates), periodic cells, plain variables, the
//! driving coordinates of `-se_gsm` and the bonds of `-gnt_vec` (Cartesian here), and
//! redundant systems for the methods that move in the coordinates.
//!
//! MultiOptPy: no counterpart (its searches are in Cartesian coordinates).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use super::ieipmain_settings::{IeipMethod, IeipmainConfig};
use crate::coord::system::CoordinateRegistry;
use crate::core_infra::log::Logger;
use crate::domain::MolecularSystem;
use crate::error::{MoptError, MoptResult};
use crate::path::points::StepMap;
use crate::workflow::pair_search::internal::{step_map, InternalSearchSurface, LastStart};
use crate::workflow::pair_search::SearchSurface;

/// Whether `-coord` asks for Cartesian coordinates.
pub fn is_cartesian(name: &str) -> bool {
    matches!(
        name.trim().to_ascii_lowercase().as_str(),
        "" | "cart" | "cartesian"
    )
}

/// The systems to try for `-coord`, in order (Cartesian coordinates after them).
fn candidates(name: &str) -> Vec<String> {
    let name = name.trim().to_ascii_lowercase();
    let family = name.split(':').next().unwrap_or("").to_string();
    let others: &[&str] = match family.as_str() {
        "dic" | "delocalized" => &[],
        "tric" => &["tric-cart", "dic"],
        "tric-cart" => &["dic"],
        _ => &["tric-cart", "dic"],
    };
    let mut out = vec![name];
    out.extend(others.iter().map(|o| (*o).to_string()));
    out.dedup();
    out
}

/// Whether the method keeps Cartesian distances and takes only its steps through the
/// coordinates.
fn steps_only(method: &IeipMethod) -> bool {
    matches!(method, IeipMethod::Ieip | IeipMethod::DsAfir)
}

/// Refuse what `-coord` cannot be combined with.
pub fn check(config: &IeipmainConfig, constrained: bool, periodic: bool) -> MoptResult<()> {
    if is_cartesian(&config.coordinates) {
        return Ok(());
    }
    let reason = if constrained {
        Some("fixed atoms and constraints (-fix, -gfix) are defined on Cartesian coordinates")
    } else if periodic {
        Some("structures in a periodic cell move in Cartesian coordinates here")
    } else if matches!(config.method, IeipMethod::SeGsm(_)) {
        Some("the driving coordinates of -se_gsm are defined on Cartesian coordinates here")
    } else if matches!(config.method, IeipMethod::NewtonTrajectory) && !config.gnt_bonds.is_empty()
    {
        Some("the bonds of -gnt_vec are defined on Cartesian coordinates here")
    } else if !steps_only(&config.method)
        && ["ric", "redundant", "tric-p", "tricp", "tric_p"]
            .contains(&config.coordinates.trim().to_ascii_lowercase().as_str())
    {
        Some(
            "the method moves in the coordinates, which must be independent (dic, tric, \
             tric-cart); redundant ones serve only iEIP and DS-AFIR",
        )
    } else {
        None
    };
    match reason {
        Some(r) => Err(MoptError::unsupported(format!(
            "-coord {} cannot be used with this run: {r} (use -coord cart)",
            config.coordinates
        ))),
        None => Ok(()),
    }
}

/// The reference structure of the coordinates: the midpoint of the inputs.
fn reference(structures: &[MolecularSystem], points: &[Col<f64>]) -> MoptResult<MolecularSystem> {
    let first = structures
        .first()
        .ok_or_else(|| MoptError::config("-coord: no structure"))?;
    match points {
        [a, b, ..] => first.with_flat_coords_bohr(&((a + b) * 0.5)),
        _ => Ok(first.clone()),
    }
}

/// What `-coord` sets up.
pub struct CoordinateSetup<'s> {
    /// The surface in the coordinates (the methods that move in them).
    pub internal: Option<InternalSearchSurface<'s>>,
    /// The points of the inputs in those coordinates.
    pub points: Vec<Col<f64>>,
    /// The map of the steps (iEIP, DS-AFIR).
    pub step_map: Option<StepMap>,
    /// Where the map took its last step from.
    pub last_start: Option<LastStart>,
}

impl CoordinateSetup<'_> {
    /// The structure reached last (Cartesian), where the coordinates broke down.
    pub fn last_structure(&self) -> Option<Col<f64>> {
        if let Some(internal) = &self.internal {
            return internal.last_structure();
        }
        self.last_start
            .as_ref()
            .and_then(|l| l.lock().ok()?.clone())
    }
}

/// The coordinates a run tries, in order, and what happened to them.
pub struct CoordinatePlan {
    names: Vec<String>,
    index: usize,
    reference: Option<Col<f64>>,
    reinitialized: bool,
    /// What happened (set up anew, fallbacks), for the log and the report.
    pub events: Vec<String>,
}

impl CoordinatePlan {
    /// The plan of `-coord` of `config` (nothing for Cartesian coordinates).
    pub fn new(config: &IeipmainConfig) -> Self {
        Self {
            names: if is_cartesian(&config.coordinates) {
                Vec::new()
            } else {
                candidates(&config.coordinates)
            },
            index: 0,
            reference: None,
            reinitialized: false,
            events: Vec::new(),
        }
    }

    /// Whether other coordinates than Cartesian ones are in use.
    pub fn active(&self) -> bool {
        self.index < self.names.len()
    }

    fn next_name(&self) -> String {
        self.names
            .get(self.index)
            .cloned()
            .unwrap_or_else(|| "cart".to_string())
    }

    fn fall_back(&mut self, why: &str) {
        let from = self.next_name();
        self.index += 1;
        self.reinitialized = false;
        self.reference = None;
        self.events
            .push(format!("{from} -> {} ({why})", self.next_name()));
    }

    /// The coordinates broke down near `at` (Cartesian): set them up anew there, or take the
    /// next ones.
    pub fn broke_down(&mut self, at: Option<Col<f64>>, log: &Logger) {
        let name = self.next_name();
        match (self.reinitialized, at) {
            (false, Some(x)) => {
                self.reinitialized = true;
                self.reference = Some(x);
                self.events.push(format!("{name} set up anew"));
                log.warn(&format!(
                    "Coordinates (-coord {name}): set up anew at the structure where they broke \
                     down; the search starts again from the inputs."
                ));
            }
            _ => {
                self.fall_back("broke down again");
                log.warn(&format!(
                    "Coordinates (-coord {name}) broke down again: the search starts again from \
                     the inputs in {}.",
                    self.next_name()
                ));
            }
        }
    }

    /// One line on what happened.
    pub fn summary(&self) -> String {
        if self.events.is_empty() {
            format!("Coordinates: {} for the whole run.", self.next_name())
        } else {
            format!(
                "Coordinates: {} in the end ({} change(s): {}).",
                self.next_name(),
                self.events.len(),
                self.events.join("; ")
            )
        }
    }

    /// Set up the coordinates at hand (or the next that can be set up) for `points`
    /// (Cartesian, the second laid onto the first) on `surface`.
    pub fn set_up<'s>(
        &mut self,
        config: &IeipmainConfig,
        surface: &'s dyn SearchSurface,
        structures: &[MolecularSystem],
        points: &[Col<f64>],
        log: &Logger,
    ) -> MoptResult<CoordinateSetup<'s>> {
        let registry = CoordinateRegistry::default();
        while self.active() {
            let name = self.next_name();
            let at = match &self.reference {
                Some(x) => structures
                    .first()
                    .ok_or_else(|| MoptError::config("-coord: no structure"))?
                    .with_flat_coords_bohr(x)?,
                None => reference(structures, points)?,
            };
            let mut setup = CoordinateSetup {
                internal: None,
                points: points.to_vec(),
                step_map: None,
                last_start: None,
            };
            let made = registry.create(&name).and_then(|coords| {
                if steps_only(&config.method) {
                    let (map, last) = step_map(coords, &at)?;
                    setup.step_map = Some(map);
                    setup.last_start = Some(last);
                } else {
                    let (internal, q) =
                        InternalSearchSurface::new(surface, coords, at.clone(), points)?;
                    setup.points = q;
                    setup.internal = Some(internal);
                }
                Ok(())
            });
            match made {
                Ok(()) => {
                    log.info(&match &setup.internal {
                        None => format!(
                            "Coordinates (-coord {name}): the steps of the images go through \
                             these coordinates; the distances and forces of the method stay \
                             Cartesian."
                        ),
                        Some(i) => format!(
                            "Coordinates (-coord {name}): the search moves in {} coordinates; \
                             steps, distances, and gradients are in these coordinates.",
                            i.dimension()
                        ),
                    });
                    return Ok(setup);
                }
                Err(e) => {
                    log.warn(&format!(
                        "Coordinates (-coord {name}) cannot be set up for these structures: {e}"
                    ));
                    self.fall_back("not set up");
                }
            }
        }
        if !self.events.is_empty() {
            log.warn(
                "None of the coordinates can be used for these structures: the search is in \
                 Cartesian coordinates.",
            );
        }
        Ok(CoordinateSetup {
            internal: None,
            points: points.to_vec(),
            step_map: None,
            last_start: None,
        })
    }
}
