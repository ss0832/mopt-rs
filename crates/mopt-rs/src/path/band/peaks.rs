//! How far the images at the maxima of the profile are moved along the chain from one
//! redistribution to the next.
//!
//! The redistribution towards the predicted maxima puts the image at a maximum of the
//! energy profile where a polynomial through it and its neighbours has its maximum. The
//! prediction is as good as the energies and slopes it is made from: with images that are
//! not on the path yet (steps from an approximate Hessian) it can be off by more than the
//! distance to the true maximum, and then the image goes back and forth between two
//! places for ever (HCN to HNC with a model Hessian: the barrier alternated between 68.29
//! and 68.05 kcal/mol every second redistribution, the forces stayed at 2e-3). A move that
//! reverses the last one is therefore limited to half of it, as in a bisection; moves in
//! the same direction may grow again by half.
//!
//! MultiOptPy: no counterpart (new in mopt-rs; MultiOptPy moves the image by what the
//! polynomial says).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

/// A move that reverses the last one is at most this share of it.
const REVERSED_SHARE: f64 = 0.5;
/// The limit grows by this factor with a move in the direction of the last one.
const GROWTH: f64 = 1.5;

/// The last moves of the images at maxima and their limits.
#[derive(Debug, Clone, Default)]
pub struct PeakMoves {
    last: Vec<f64>,
    limit: Vec<f64>,
}

impl PeakMoves {
    /// Nothing moved yet.
    pub fn new() -> Self {
        Self::default()
    }

    /// Forget the moves (the chain was replaced).
    pub fn clear(&mut self) {
        self.last.clear();
        self.limit.clear();
    }

    /// The images of a chain of `images` that are at maxima now: what is kept for the
    /// others is forgotten.
    pub fn keep_only(&mut self, images: usize, maxima: &[usize]) {
        if self.last.len() != images {
            self.last = vec![0.0; images];
            self.limit = vec![f64::INFINITY; images];
        }
        for image in 0..images {
            if !maxima.contains(&image) {
                self.last[image] = 0.0;
                self.limit[image] = f64::INFINITY;
            }
        }
    }

    /// The move along the chain of the image `image` when the prediction asks for
    /// `wanted` (call [`Self::keep_only`] first).
    pub fn limited(&mut self, image: usize, wanted: f64) -> f64 {
        let (Some(last), Some(limit)) = (self.last.get_mut(image), self.limit.get_mut(image))
        else {
            return wanted;
        };
        if wanted * *last < 0.0 {
            *limit = REVERSED_SHARE * limit.min(last.abs());
        } else if *last != 0.0 {
            *limit *= GROWTH;
        }
        let moved = wanted.clamp(-*limit, *limit);
        *last = moved;
        moved
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_prediction_that_goes_back_and_forth_comes_to_rest() {
        // The prediction is off: from `x` it asks for the move `-1.6 (x - 1) `, which
        // overshoots the fixed point at 1 by more than it started from it.
        let ask = |x: f64| -1.6 * (x - 1.0) - 0.9 * (x - 1.0).signum();
        let mut plain = 0.0_f64;
        let mut x = 0.0_f64;
        let mut moves = PeakMoves::new();
        for _ in 0..40 {
            plain += ask(plain);
            moves.keep_only(5, &[2]);
            x += moves.limited(2, ask(x));
        }
        assert!((plain - 1.0).abs() > 0.5, "{plain}");
        assert!((x - 1.0).abs() < 1e-6, "{x}");
    }

    #[test]
    fn moves_in_one_direction_are_not_limited_and_other_images_start_anew() {
        let mut moves = PeakMoves::new();
        moves.keep_only(4, &[1]);
        assert_eq!(moves.limited(1, 0.3), 0.3);
        assert_eq!(moves.limited(1, 0.5), 0.5);
        // A reversal: half of the last move.
        assert_eq!(moves.limited(1, -0.4), -0.25);
        // The same direction again: the limit grows by half.
        assert_eq!(moves.limited(1, -0.4), -0.375);
        // The maximum is at another image now: no limit there, and none is left behind.
        moves.keep_only(4, &[2]);
        assert_eq!(moves.limited(2, -0.9), -0.9);
        moves.keep_only(4, &[1]);
        assert_eq!(moves.limited(1, 0.9), 0.9);
        // An image the chain does not have is left alone.
        assert_eq!(moves.limited(7, 0.2), 0.2);
    }
}
