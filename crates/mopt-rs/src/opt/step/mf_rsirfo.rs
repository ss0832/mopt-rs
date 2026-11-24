//! Mode-following RS-I-RFO: restricted-step image RFO whose reflected modes are tracked by
//! overlap from iteration to iteration.
//!
//! The step is that of [`super::rsirfo::Rsirfo`] (Banerjee et al., J. Phys. Chem. 1985, 89, 52;
//! Besalu and Bofill, Theor. Chem. Acc. 1998, 100, 265); the image function reflects the modes
//! chosen by [`super::mode_following::ModeFollowing`] (Baker, J. Comput. Chem. 1986, 7, 385)
//! instead of the lowest ones, so that the search keeps climbing the same mode when the order
//! of the eigenvalues changes. The trust radius is rated with the curvature of the followed
//! mode. Method names: `mf_rsirfo_<updater>` (adaptive references), `smf_rsirfo_<updater>`
//! (static references), and `mwmf_` / `mwsmf_` (mass-weighted overlaps), with the optional
//! suffixes `:<mode index>:ema<rate>:grad<weight>`.
//!
//! The tracker is given the modes with non-zero eigenvalues only. MultiOptPy passes the whole
//! spectrum of the projected Hessian, whose first modes are the rigid-body ones whenever the
//! Hessian has no negative eigenvalue; these are then "followed" and skipped by the reflection,
//! and the step minimizes along every mode.
//!
//! MultiOptPy: `Optimizer/mf_rsirfo.py:MF_RSIRFO`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use super::gradient_descent::isotope_masses;
use super::mode_following::{ModeFollowing, ModeFollowingSettings};
use super::rsirfo::{Rsirfo, RsirfoTrust};
use super::{StepGenerator, StepInput};
use crate::error::MoptResult;

/// Mode-following RS-I-RFO step generator.
#[derive(Debug, Clone)]
pub struct MfRsirfo {
    inner: Rsirfo,
    follower: ModeFollowing,
    saddle_order: usize,
    masses_set: bool,
}

impl MfRsirfo {
    /// Generator following `saddle_order` modes.
    pub fn new(trust: RsirfoTrust, saddle_order: usize, settings: ModeFollowingSettings) -> Self {
        Self {
            inner: Rsirfo::new(trust),
            follower: ModeFollowing::new(saddle_order, settings),
            saddle_order,
            masses_set: false,
        }
    }

    /// The mode tracker.
    pub fn follower(&self) -> &ModeFollowing {
        &self.follower
    }

    /// Current internal trust radius.
    pub fn trust_radius(&self) -> f64 {
        self.inner.trust_radius()
    }
}

impl StepGenerator for MfRsirfo {
    fn name(&self) -> &str {
        "mf_rsirfo"
    }

    fn uses_hessian(&self) -> bool {
        true
    }

    fn propose(&mut self, input: &StepInput<'_>) -> MoptResult<Col<f64>> {
        if !self.masses_set {
            self.masses_set = true;
            if let (Some(sys), true) = (input.system, input.cartesian_coordinates) {
                let m = isotope_masses(sys);
                if 3 * m.len() == input.gradient.nrows() {
                    self.follower
                        .set_masses(Col::from_fn(3 * m.len(), |i| m[i / 3]));
                }
            }
        }
        if self.saddle_order != input.saddle_order {
            // The number of followed modes is fixed at construction.
            self.follower =
                ModeFollowing::new(input.saddle_order, self.follower.settings().clone());
            self.saddle_order = input.saddle_order;
            self.masses_set = false;
        }
        self.inner
            .propose_following(input, Some(&mut self.follower))
    }

    fn reset(&mut self) {
        self.inner.reset();
        self.follower.reset();
    }

    fn seed_followed_modes(&mut self, directions: &[Col<f64>]) -> bool {
        self.follower.seed(directions);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::opt::stepper::plain_input;
    use faer::Mat;

    #[test]
    fn the_followed_mode_stays_the_same_when_the_spectrum_reorders() {
        let x = Col::zeros(2);
        let g = Col::from_fn(2, |i| [0.02, 0.01][i]);
        let trust = RsirfoTrust::for_saddle_order(1, None, None);
        let mut mf = MfRsirfo::new(trust, 1, ModeFollowingSettings::default());
        // Mode x is the lowest: it is followed (uphill along x, downhill along y).
        let h = Mat::from_fn(2, 2, |i, j| if i == j { [-0.5, 1.0][i] } else { 0.0 });
        let s = mf
            .propose(&plain_input(0, &x, 0.0, &g, &h, None, 1))
            .unwrap();
        assert!(s[0] > 0.0 && s[1] < 0.0, "{s:?}");
        // Now y has the lower eigenvalue; the tracker stays on x.
        let h = Mat::from_fn(2, 2, |i, j| if i == j { [-0.5, -1.0][i] } else { 0.0 });
        let s = mf
            .propose(&plain_input(1, &x, 0.0, &g, &h, None, 1))
            .unwrap();
        assert_eq!(mf.follower().indices(), &[1]);
        // Uphill along x (the followed mode, eigenvalue -0.5); y has negative curvature and
        // is minimized: the step goes against its gradient component.
        assert!(s[0] > 0.0 && s[1] < 0.0, "{s:?}");
        // Plain RS-I-RFO would switch to y.
        let mut plain = Rsirfo::new(trust);
        let s = plain
            .propose(&plain_input(0, &x, 0.0, &g, &h, None, 1))
            .unwrap();
        assert!(s[1] > 0.0, "{s:?}");
    }
}
