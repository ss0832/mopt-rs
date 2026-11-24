//! Places of the nodes of a graph in the plane for a drawing: the force-directed layout of
//! Fruchterman and Reingold (Softw. Pract. Exper. 1991, 21, 1129), with their grid variant of
//! the repulsion (only between nodes closer than twice the ideal length, found through a grid
//! of cells of that size), so that a step costs O(n + edges) and a graph of tens of thousands
//! of nodes is laid out in seconds.
//!
//! Nodes joined by an edge attract (`d^2 / k`), all nodes repel within `2k` (`k^2 / d`), and a
//! weak pull to the centre keeps parts that nothing joins near the rest. The nodes start on a
//! sunflower spiral in the order given (the first in the middle; a caller that orders them by
//! energy gets the low ones in the middle), and the step is limited by a temperature that falls
//! linearly to zero. Nothing is random: the same graph gives the same picture.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::HashMap;

use rayon::prelude::*;

/// Settings of the layout.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LayoutSettings {
    /// Steps of the layout.
    pub iterations: usize,
    /// Pull to the centre, relative to the attraction of an edge of the ideal length.
    pub gravity: f64,
}

impl Default for LayoutSettings {
    fn default() -> Self {
        Self {
            iterations: 300,
            gravity: 0.02,
        }
    }
}

/// The places of `n` nodes joined by `edges` (pairs of node indices; loops and edges to nodes
/// that are not there are left out), in units of the ideal edge length.
pub fn force_layout(
    n: usize,
    edges: &[(usize, usize)],
    settings: &LayoutSettings,
) -> Vec<[f64; 2]> {
    let golden = std::f64::consts::PI * (3.0 - 5f64.sqrt());
    let mut p: Vec<[f64; 2]> = (0..n)
        .map(|i| {
            let r = (i as f64 + 0.5).sqrt();
            let t = i as f64 * golden;
            [r * t.cos(), r * t.sin()]
        })
        .collect();
    if n < 2 {
        return p;
    }
    let mut adjacent: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &(a, b) in edges {
        if a < n && b < n && a != b && !adjacent[a].contains(&b) {
            adjacent[a].push(b);
            adjacent[b].push(a);
        }
    }
    let k = 1.0;
    let cutoff = 2.0 * k;
    let start = (n as f64).sqrt() * 0.1 + 1.0;
    let steps = settings.iterations.max(1);
    for step in 0..steps {
        let temperature = start * (1.0 - step as f64 / steps as f64);
        // The grid of the repulsion, cells of the cutoff.
        let mut grid: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
        for (i, q) in p.iter().enumerate() {
            let cell = (
                (q[0] / cutoff).floor() as i64,
                (q[1] / cutoff).floor() as i64,
            );
            grid.entry(cell).or_default().push(i);
        }
        let moves: Vec<[f64; 2]> = (0..n)
            .into_par_iter()
            .map(|i| {
                let [x, y] = p[i];
                let (cx, cy) = ((x / cutoff).floor() as i64, (y / cutoff).floor() as i64);
                let mut f = [0.0, 0.0];
                for gx in cx - 1..=cx + 1 {
                    for gy in cy - 1..=cy + 1 {
                        let Some(cell) = grid.get(&(gx, gy)) else {
                            continue;
                        };
                        for &j in cell {
                            if j == i {
                                continue;
                            }
                            let (dx, dy) = (x - p[j][0], y - p[j][1]);
                            let d2 = dx * dx + dy * dy;
                            if d2 >= cutoff * cutoff {
                                continue;
                            }
                            // Nodes on top of each other part along a direction of their own.
                            let (dx, dy, d2) = if d2 < 1e-12 {
                                let t = (i as f64 - j as f64) * golden;
                                (1e-3 * t.cos(), 1e-3 * t.sin(), 1e-6)
                            } else {
                                (dx, dy, d2)
                            };
                            let s = k * k / d2;
                            f[0] += dx * s;
                            f[1] += dy * s;
                        }
                    }
                }
                for &j in &adjacent[i] {
                    let (dx, dy) = (p[j][0] - x, p[j][1] - y);
                    let d = (dx * dx + dy * dy).sqrt();
                    f[0] += dx * d / k;
                    f[1] += dy * d / k;
                }
                f[0] -= settings.gravity * x;
                f[1] -= settings.gravity * y;
                let size = (f[0] * f[0] + f[1] * f[1]).sqrt();
                if size > temperature && size > 0.0 {
                    let s = temperature / size;
                    [f[0] * s, f[1] * s]
                } else {
                    f
                }
            })
            .collect();
        for (q, m) in p.iter_mut().zip(&moves) {
            q[0] += m[0];
            q[1] += m[1];
        }
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    fn length(p: &[[f64; 2]], a: usize, b: usize) -> f64 {
        ((p[a][0] - p[b][0]).powi(2) + (p[a][1] - p[b][1]).powi(2)).sqrt()
    }

    #[test]
    fn joined_nodes_come_close_and_the_others_keep_apart() {
        // A ring of 12 nodes: neighbours near the ideal length, opposite nodes far apart, no
        // two nodes on top of each other.
        let n = 12;
        let edges: Vec<(usize, usize)> = (0..n).map(|i| (i, (i + 1) % n)).collect();
        let p = force_layout(n, &edges, &LayoutSettings::default());
        for &(a, b) in &edges {
            let d = length(&p, a, b);
            assert!(d > 0.4 && d < 2.0, "edge {a}-{b}: {d}");
        }
        assert!(length(&p, 0, 6) > 2.0 * length(&p, 0, 1));
        for a in 0..n {
            for b in 0..a {
                assert!(length(&p, a, b) > 0.2);
            }
        }
        // The same graph gives the same picture; loops and missing nodes are left out.
        let mut more = edges.clone();
        more.extend([(3, 3), (2, 99)]);
        assert_eq!(p, force_layout(n, &more, &LayoutSettings::default()));
        assert_eq!(force_layout(1, &[], &LayoutSettings::default()).len(), 1);
    }
}
