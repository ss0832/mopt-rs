//! The relaxed scan that confirms a barrierless dissociation (`confirm_dissociations=on`): a
//! complex of two molecules is joined to the channel of the two by an edge without a
//! bottleneck, on the assumption that the energy rises from the complex to the separated
//! molecules without a maximum. The scan tests it with the parts of `relaxedscan`: the distance
//! between the centres of the two molecules (the constraint `fbond` of the commands) is held
//! at the values of a [`ScanAxis`], every point minimized under the constraint from the one
//! before, until the molecules are apart by the test of the channels; the maxima of the curve
//! are those of [`stationary_points`], as for the curve of a relaxed scan. A maximum more than
//! the tolerance above the channel is a barrier, and the edge says so.
//!
//! The rules are in `docs/source/mapper_design.md` ("Default on, or asked for").
//!
//! MultiOptPy: no counterpart in its mapper (the scan itself is that of
//! `Entrypoints/relaxed_scan.py`, a coordinate held at a list of values).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use super::channels::{groups, molecules, ChannelSettings};
use super::method::JobContext;
use super::network::{Level, NodeKind};
use super::Configuration;
use crate::domain::{AtomId, MolecularSystem, Units};
use crate::error::MoptResult;
use crate::pes::constraint::{Constraint, ConstraintKind};
use crate::workflow::scan::{stationary_points, CandidateKind, ScanAxis};

/// The settings of the scan.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScanSettings {
    /// The step of the distance between the centres of the two molecules (Bohr;
    /// `dissociation_scan_step` in Angstrom).
    pub step: f64,
    /// A maximum of the curve more than this above the channel is a barrier (Hartree;
    /// `dissociation_scan_tolerance` in kcal/mol).
    pub tolerance: f64,
    /// Points at most beyond the complex.
    pub max_steps: usize,
}

/// The default step of the scan (Angstrom).
pub const DEFAULT_SCAN_STEP: f64 = 0.3;
/// The default tolerance of the scan (kcal/mol).
pub const DEFAULT_SCAN_TOLERANCE: f64 = 0.5;
/// The points of a scan at most (12 Angstrom with the default step).
pub const SCAN_STEPS: usize = 40;

impl Default for ScanSettings {
    fn default() -> Self {
        Self {
            step: DEFAULT_SCAN_STEP * Units::ANGSTROM_TO_BOHR,
            tolerance: DEFAULT_SCAN_TOLERANCE * Units::KCAL_MOL_TO_HARTREE,
            max_steps: SCAN_STEPS,
        }
    }
}

/// What the scan of a complex gave.
#[derive(Debug, Clone, PartialEq)]
pub struct DissociationScan {
    /// The distance between the centres of the two molecules at every point (Bohr), the
    /// complex first.
    pub distances: Vec<f64>,
    /// The energy of every point (Hartree).
    pub energies: Vec<f64>,
    /// The highest maximum of the curve above the channel (Hartree; `None`: the curve has no
    /// maximum).
    pub highest: Option<f64>,
    /// Whether that is a barrier: more than the tolerance above the channel.
    pub barrier: bool,
    /// Whether the scan reached separated molecules (otherwise it ended at its last point, at
    /// a point that did not converge, or where the two molecules were no longer those two).
    pub separated: bool,
}

fn centre(system: &MolecularSystem, atoms: &[usize]) -> [f64; 3] {
    let x = system.coords_flat_bohr();
    let n = atoms.len().max(1) as f64;
    std::array::from_fn(|c| atoms.iter().map(|&i| x[3 * i + c]).sum::<f64>() / n)
}

/// The distance between the centres of the atoms `a` and `b` of `system` (Bohr).
pub fn centre_distance(system: &MolecularSystem, a: &[usize], b: &[usize]) -> f64 {
    let (p, q) = (centre(system, a), centre(system, b));
    (0..3).map(|c| (p[c] - q[c]).powi(2)).sum::<f64>().sqrt()
}

/// The highest maximum of the curve `energies` above `channel_energy` and whether it is a
/// barrier by `tolerance` (the maxima of [`stationary_points`]; the two ends of a curve are no
/// maxima).
pub fn curve_barrier(
    energies: &[f64],
    channel_energy: f64,
    tolerance: f64,
) -> MoptResult<(Option<f64>, bool)> {
    let highest = stationary_points(&[energies.len()], energies)?
        .into_iter()
        .filter(|c| c.kind == CandidateKind::Saddle)
        .map(|c| c.energy - channel_energy)
        .reduce(f64::max);
    Ok((highest, highest.is_some_and(|h| h > tolerance)))
}

/// How a scan ended.
#[derive(Debug, Clone, PartialEq)]
pub enum ScanEnd {
    /// The run was stopped during it.
    Stopped,
    /// The network has the complex with a scanned edge already: nothing was scanned again.
    Known,
    /// The scan.
    Done(DissociationScan),
}

/// The scan of the complex `complex` (a minimum) of the two molecules with the atoms `a` and
/// `b` towards their channel of energy `channel_energy`. The first point is the complex with
/// its distance held (its energy; the minimization ends at once), and a complex that the
/// network has with a scanned edge is not scanned again.
pub fn dissociation_scan(
    ctx: &mut JobContext<'_>,
    complex: &Configuration,
    a: &[usize],
    b: &[usize],
    channel_energy: f64,
    settings: &ScanSettings,
    channels: &ChannelSettings,
) -> MoptResult<ScanEnd> {
    let mut scan = DissociationScan {
        distances: Vec::new(),
        energies: Vec::new(),
        highest: None,
        barrier: false,
        separated: false,
    };
    let Some(start) = complex.structure() else {
        return Ok(ScanEnd::Done(scan));
    };
    let ids = |atoms: &[usize]| -> Vec<AtomId> { atoms.iter().map(|&i| AtomId::new(i)).collect() };
    let first = centre_distance(start, a, b);
    // The values of the distance: those of a scan from the complex outwards.
    let axis = ScanAxis::linspace(
        "fbond",
        "Bohr",
        first,
        first + settings.max_steps as f64 * settings.step,
        settings.max_steps + 1,
    )?;
    let mut current = complex.clone();
    for (k, &target) in axis.values.iter().enumerate() {
        let held = Constraint {
            kind: ConstraintKind::Distance(ids(a), ids(b)),
            target: Some(target),
            anchor: None,
        };
        let point = ctx.jobs.held(
            &current,
            &[held],
            &format!(
                "dissociation scan, point {k} ({:.2} Angstrom)",
                target * Units::BOHR_TO_ANGSTROM
            ),
            ctx.frames,
        )?;
        if point.interrupted {
            return Ok(ScanEnd::Stopped);
        }
        if !point.converged {
            break;
        }
        if k == 0 {
            // The complex itself: one the network has with a scanned edge is done.
            let known = ctx.network.find_same_in(
                &NodeKind::Minimum,
                Level::Exploration,
                &point.system,
                point.energy,
                ctx.matcher,
                ctx.state.as_ref(),
            )?;
            let scanned = known.is_some_and(|id| {
                ctx.network
                    .dissociation_edges
                    .iter()
                    .any(|e| e.minimum == id && e.scan.is_some())
            });
            if scanned {
                return Ok(ScanEnd::Known);
            }
        }
        let Some(sys) = point.system.structure() else {
            break;
        };
        // The same two molecules still: a bond made or broken on the way ends the scan (a
        // reaction is no dissociation).
        let parts = molecules(sys, channels.bond_scale);
        let same = parts.len() == 2
            && ((parts[0] == a && parts[1] == b) || (parts[0] == b && parts[1] == a));
        if !same {
            break;
        }
        scan.distances.push(centre_distance(sys, a, b));
        scan.energies.push(point.energy);
        if groups(sys, &parts, channels.margin).len() >= 2 {
            scan.separated = true;
            break;
        }
        current = point.system;
    }
    (scan.highest, scan.barrier) =
        curve_barrier(&scan.energies, channel_energy, settings.tolerance)?;
    Ok(ScanEnd::Done(scan))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn a_curve_that_only_rises_has_no_barrier_and_a_hump_has_one() {
        // A Morse-like rise to the channel at 0.
        let rising: Vec<f64> = (0..12)
            .map(|k| -0.01 * (-0.6 * f64::from(k)).exp())
            .collect();
        assert_eq!(curve_barrier(&rising, 0.0, 1e-3).unwrap(), (None, false));
        // The same with a hump of 0.004 Hartree above the channel at the fifth point.
        let mut hump = rising.clone();
        hump[5] = 0.004;
        let (highest, barrier) = curve_barrier(&hump, 0.0, 1e-3).unwrap();
        assert!((highest.unwrap() - 0.004).abs() < 1e-12);
        assert!(barrier);
        // A hump below the tolerance is a maximum, and no barrier.
        hump[5] = 0.0005;
        let (highest, barrier) = curve_barrier(&hump, 0.0, 1e-3).unwrap();
        assert!(highest.is_some() && !barrier);
    }
}
