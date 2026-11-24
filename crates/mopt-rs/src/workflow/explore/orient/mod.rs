//! Orientation search of molecular complexes (`orientsearch`): random relative orientations
//! and placements of the parts of a complex, each optimized, and the distinct minima.
//!
//! - [`parts`]: which atoms move together and which stay;
//! - [`contact`]: van der Waals contacts, the approach of a part, the free side of a slab;
//! - [`placement`]: the sampling schemes (MultiOptPy's, uniform, contact) and [`ring`];
//! - [`search`]: the ask/tell state machine over the samples.
//!
//! MultiOptPy: `Entrypoints/orientation_search.py`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

pub mod contact;
pub mod parts;
pub mod placement;
pub mod ring;
pub mod search;

pub use parts::{ComplexParts, PartSource};
pub use placement::{Perturbation, PlacementSettings, Sampler, Scheme};
pub use search::{
    OrientAction, OrientSettings, OrientationSearch, SampleOrigin, SampleOutcome, SampleRecord,
    SampleRequest,
};
