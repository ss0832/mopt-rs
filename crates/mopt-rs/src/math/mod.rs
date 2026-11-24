//! Numerical foundations: automatic differentiation, dense linear algebra, projectors, and
//! scalar root finding.
//!
//! MultiOptPy: `Utils/calc_tools.py`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

pub mod ad;
pub mod assignment;
pub mod davidson;
// Fourier transforms and time correlation functions (trajectory analysis).
pub mod fft;
pub mod gp;
pub mod linalg;
pub mod line_search;
pub mod minimize;
pub mod mlp;
pub mod projection;
pub mod roots;
pub mod simplex_qp;
// Special functions (the Airy function and the Laplace transform of its square).
pub mod special;
pub mod stable;
