//! The direct MaxFlux method: a reaction path as a smooth curve between two structures,
//! moved until the flux along it is largest at a low temperature, with energies at a few
//! points near its top only (Koda, Saito, J. Chem. Theory Comput. 2024, 20, 2798; the
//! definitions of the authors' program PyDMF, MIT, `third_party/PYDMF.md`).
//!
//! - [`spline`]: the path as a B-spline of a parameter `t` in `[0, 1]`.
//! - [`arc`]: its length, and where a share of the length is reached.
//! - [`action`]: the objective `ln(integral exp(beta E) ds) / beta` and its gradient.
//! - [`profile`]: the energy along the path from the energies and slopes at the points,
//!   its top (the estimate of the transition state), and where the points go next.
//! - [`machine`]: the method as an ask/tell state machine.
//!
//! MultiOptPy: `MEP/pathopt_dmf_force.py` (`nebmain -dmf`): not ported; its spline is fit
//! through the images of a chain anew in every iteration and its objective is handed back
//! as forces on the images, with the speed at the points only, with points at equal
//! steps of the parameter, and with `beta = 10` per Hartree (the method's is 10 per eV). See
//! `docs/source/migration_paths.md`, "The direct MaxFlux method".
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

pub mod action;
pub mod arc;
pub mod machine;
pub mod profile;
pub mod spline;
#[cfg(test)]
mod tests;

pub use machine::{DirectMaxFlux, DmfAction, DmfEnd, DmfPoint, DmfReport, DmfSettings, PointMoves};
pub use spline::SplinePath;
