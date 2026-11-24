//! Deterministic pseudo-random number generator (`SplitMix64` + `Xoshiro256**` + Box-Muller).
//!
//! MultiOptPy: no counterpart (new in mopt-rs) (MultiOptPy uses `numpy.random`).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::f64::consts::TAU;

/// Deterministic `Xoshiro256**` pseudo-random number generator with Gaussian sampling.
#[derive(Debug, Clone)]
pub struct Rng {
    state: [u64; 4],
    spare_normal: Option<f64>,
}

impl Rng {
    /// Seed a new generator from a 64-bit integer using `SplitMix64`.
    pub fn from_seed(seed: u64) -> Self {
        let mut sm = seed;
        let mut next_sm = || {
            sm = sm.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = sm;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        };
        let mut s = [next_sm(), next_sm(), next_sm(), next_sm()];
        if s == [0, 0, 0, 0] {
            s[0] = 1;
        }
        Self {
            state: s,
            spare_normal: None,
        }
    }

    /// Generate the next raw 64-bit unsigned integer.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let result = (self.state[1].wrapping_mul(5))
            .rotate_left(7)
            .wrapping_mul(9);
        let t = self.state[1] << 17;

        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];

        self.state[2] ^= t;
        self.state[3] = self.state[3].rotate_left(45);

        result
    }

    /// Uniform sample in `[0.0, 1.0)`.
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        let bits = self.next_u64() >> 11;
        (bits as f64) * (1.0 / ((1u64 << 53) as f64))
    }

    /// Uniform sample in `[low, high)`.
    #[inline]
    pub fn uniform(&mut self, low: f64, high: f64) -> f64 {
        low + (high - low) * self.next_f64()
    }

    /// Uniform integer in `[0, upper)` (panics if `upper == 0`).
    pub fn gen_range_usize(&mut self, upper: usize) -> usize {
        assert!(upper > 0, "upper bound must be positive");
        (self.next_u64() % (upper as u64)) as usize
    }

    /// Standard normal `N(0, 1)` sample via the Box-Muller transform.
    pub fn standard_normal(&mut self) -> f64 {
        if let Some(val) = self.spare_normal.take() {
            return val;
        }
        let mut u1 = self.next_f64();
        while u1 <= f64::MIN_POSITIVE {
            u1 = self.next_f64();
        }
        let u2 = self.next_f64();
        let mag = libm::sqrt(-2.0 * libm::log(u1));
        let angle = TAU * u2;
        self.spare_normal = Some(mag * libm::sin(angle));
        mag * libm::cos(angle)
    }

    /// Gaussian `N(mean, std_dev^2)` sample.
    #[inline]
    pub fn normal(&mut self, mean: f64, std_dev: f64) -> f64 {
        mean + std_dev * self.standard_normal()
    }

    /// Uniformly distributed 3D unit vector on `S^2`.
    pub fn unit_vector_3d(&mut self) -> [f64; 3] {
        loop {
            let x = self.standard_normal();
            let y = self.standard_normal();
            let z = self.standard_normal();
            let n2 = x * x + y * y + z * z;
            if n2 > 1e-24 {
                let inv = 1.0 / libm::sqrt(n2);
                return [x * inv, y * inv, z * inv];
            }
        }
    }

    /// Uniform random 3D rotation matrix (`SO(3)`) via Shoemake quaternion sampling.
    pub fn rotation_matrix_3d(&mut self) -> [[f64; 3]; 3] {
        let u1 = self.next_f64();
        let u2 = TAU * self.next_f64();
        let u3 = TAU * self.next_f64();
        let s1 = libm::sqrt(1.0 - u1);
        let s2 = libm::sqrt(u1);
        let qx = s1 * libm::sin(u2);
        let qy = s1 * libm::cos(u2);
        let qz = s2 * libm::sin(u3);
        let qw = s2 * libm::cos(u3);

        [
            [
                1.0 - 2.0 * (qy * qy + qz * qz),
                2.0 * (qx * qy - qz * qw),
                2.0 * (qx * qz + qy * qw),
            ],
            [
                2.0 * (qx * qy + qz * qw),
                1.0 - 2.0 * (qx * qx + qz * qz),
                2.0 * (qy * qz - qx * qw),
            ],
            [
                2.0 * (qx * qz - qy * qw),
                2.0 * (qy * qz + qx * qw),
                1.0 - 2.0 * (qx * qx + qy * qy),
            ],
        ]
    }

    /// Fisher-Yates in-place slice shuffle.
    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        let len = slice.len();
        for i in (1..len).rev() {
            let j = self.gen_range_usize(i + 1);
            slice.swap(i, j);
        }
    }
}

impl Default for Rng {
    fn default() -> Self {
        Self::from_seed(42)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rng_reproducibility_and_statistics() {
        let mut r1 = Rng::from_seed(12345);
        let mut r2 = Rng::from_seed(12345);
        for _ in 0..32 {
            assert_eq!(r1.next_u64(), r2.next_u64());
        }

        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        let n = 10_000;
        for _ in 0..n {
            let x = r1.standard_normal();
            sum += x;
            sum_sq += x * x;
        }
        let mean = sum / (n as f64);
        let var = sum_sq / (n as f64) - mean * mean;
        assert!(mean.abs() < 0.05);
        assert!((var - 1.0).abs() < 0.08);
    }
}
