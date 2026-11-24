//! Disconnectivity graphs and lowest-barrier paths of a network of minima joined by transition
//! states.
//!
//! Minima merge into superbasins at the energy of the lowest transition state that joins them
//! (Becker and Karplus, J. Chem. Phys. 1997, 106, 1495; Wales, Miller and Walsh, Nature 1998,
//! 394, 758). The merges are those of Kruskal's minimum spanning forest of the network with the
//! energies of the transition states as the weights, and the path of that forest between two
//! minima is the path whose highest transition state is the lowest of all paths between them
//! (the minimax path, a property of minimum spanning trees).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::VecDeque;

/// Two minima joined by a transition state of energy `top`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Connection {
    /// One minimum (index into the energies of the minima).
    pub a: usize,
    /// The other minimum.
    pub b: usize,
    /// The energy of the transition state.
    pub top: f64,
}

/// A basin of the graph: a minimum (a leaf), or the superbasin of the basins that merge at its
/// energy.
#[derive(Debug, Clone, PartialEq)]
pub struct Basin {
    /// The energy of the minimum, or of the transition state where the children merge.
    pub energy: f64,
    /// The basins that merge here (none for a minimum).
    pub children: Vec<usize>,
    /// The minimum of a leaf.
    pub minimum: Option<usize>,
    /// The connection whose transition state merges the children.
    pub connection: Option<usize>,
    /// The lowest minimum in the basin.
    pub lowest: f64,
    /// Horizontal place for a drawing: leaves at 0, 1, 2, ..., with the lowest basin first
    /// among siblings; a superbasin at the middle of its first and last child.
    pub x: f64,
}

/// The disconnectivity graph of a network.
#[derive(Debug, Clone, PartialEq)]
pub struct Disconnectivity {
    /// The leaves (one per minimum, in the order of the minima) and the superbasins.
    pub basins: Vec<Basin>,
    /// The basins that nothing joins further (one per connected component), lowest first.
    pub roots: Vec<usize>,
    /// The connections of the minimum spanning forest, in the order they merge basins.
    pub forest: Vec<usize>,
}

fn find(parent: &mut [usize], mut i: usize) -> usize {
    while parent[i] != i {
        parent[i] = parent[parent[i]];
        i = parent[i];
    }
    i
}

/// The graph of minima of `energies` joined by `connections` (connections of a minimum to
/// itself, to minima that are not there, and with an energy that is not finite are left out).
pub fn disconnectivity(energies: &[f64], connections: &[Connection]) -> Disconnectivity {
    let n = energies.len();
    let mut basins: Vec<Basin> = energies
        .iter()
        .enumerate()
        .map(|(i, &e)| Basin {
            energy: e,
            children: Vec::new(),
            minimum: Some(i),
            connection: None,
            lowest: e,
            x: 0.0,
        })
        .collect();
    let mut order: Vec<usize> = (0..connections.len())
        .filter(|&k| {
            let c = connections[k];
            c.a < n && c.b < n && c.a != c.b && c.top.is_finite()
        })
        .collect();
    order.sort_by(|&p, &q| {
        connections[p]
            .top
            .total_cmp(&connections[q].top)
            .then(p.cmp(&q))
    });
    let mut parent: Vec<usize> = (0..n).collect();
    let mut basin_of: Vec<usize> = (0..n).collect();
    let mut forest = Vec::new();
    for k in order {
        let c = connections[k];
        let (ra, rb) = (find(&mut parent, c.a), find(&mut parent, c.b));
        if ra == rb {
            continue;
        }
        let (left, right) = (basin_of[ra], basin_of[rb]);
        let (l, r) = (&basins[left], &basins[right]);
        // A transition state below one of its minima (an inconsistent network) merges at the
        // higher of the two, so that the drawing goes up.
        let energy = c.top.max(l.energy).max(r.energy);
        let lowest = l.lowest.min(r.lowest);
        let mut children = vec![left, right];
        children.sort_by(|&p, &q| basins[p].lowest.total_cmp(&basins[q].lowest));
        basins.push(Basin {
            energy,
            children,
            minimum: None,
            connection: Some(k),
            lowest,
            x: 0.0,
        });
        parent[rb] = ra;
        basin_of[ra] = basins.len() - 1;
        forest.push(k);
    }
    let mut roots: Vec<usize> = (0..n)
        .filter(|&i| find(&mut parent, i) == i)
        .map(|i| basin_of[i])
        .collect();
    roots.sort_by(|&p, &q| {
        basins[p]
            .lowest
            .total_cmp(&basins[q].lowest)
            .then(p.cmp(&q))
    });
    place_leaves(&mut basins, &roots);
    Disconnectivity {
        basins,
        roots,
        forest,
    }
}

/// Leaves left to right in the order of a walk that visits the lower child first; a
/// superbasin in the middle of its first and last child.
fn place_leaves(basins: &mut [Basin], roots: &[usize]) {
    let mut next = 0.0;
    let mut stack: Vec<(usize, bool)> = roots.iter().rev().map(|&r| (r, false)).collect();
    while let Some((b, done)) = stack.pop() {
        if basins[b].children.is_empty() {
            basins[b].x = next;
            next += 1.0;
        } else if done {
            let (first, last) = (
                basins[b].children[0],
                basins[b].children[basins[b].children.len() - 1],
            );
            basins[b].x = 0.5 * (basins[first].x + basins[last].x);
        } else {
            stack.push((b, true));
            for &c in basins[b].children.iter().rev() {
                stack.push((c, false));
            }
        }
    }
}

/// The graph of the minima for which `keep` is true (by the index of the minimum; missing
/// entries are false), for a drawing of a part of a large network: a superbasin with one kept
/// basin below it gives way to that basin, one with none is left out. The merges that stay are
/// those of the whole graph, so the energy where two kept minima meet is that of the whole
/// network (also when the path between them passes minima that are left out).
pub fn prune(graph: &Disconnectivity, keep: &[bool]) -> Disconnectivity {
    let mut basins: Vec<Basin> = Vec::new();
    let mut forest = Vec::new();
    let mut now: Vec<Option<usize>> = vec![None; graph.basins.len()];
    // Children have smaller indices than their superbasin: one pass from the leaves up.
    for (i, b) in graph.basins.iter().enumerate() {
        now[i] = if let Some(m) = b.minimum {
            keep.get(m).copied().unwrap_or(false).then(|| {
                basins.push(Basin {
                    x: 0.0,
                    ..b.clone()
                });
                basins.len() - 1
            })
        } else {
            let mut below: Vec<usize> = b.children.iter().filter_map(|&c| now[c]).collect();
            match below.len() {
                0 => None,
                1 => Some(below[0]),
                _ => {
                    below.sort_by(|&p, &q| basins[p].lowest.total_cmp(&basins[q].lowest));
                    let lowest = basins[below[0]].lowest;
                    basins.push(Basin {
                        energy: b.energy,
                        children: below,
                        minimum: None,
                        connection: b.connection,
                        lowest,
                        x: 0.0,
                    });
                    if let Some(k) = b.connection {
                        forest.push(k);
                    }
                    Some(basins.len() - 1)
                }
            }
        };
    }
    let mut roots: Vec<usize> = graph.roots.iter().filter_map(|&r| now[r]).collect();
    roots.sort_by(|&p, &q| {
        basins[p]
            .lowest
            .total_cmp(&basins[q].lowest)
            .then(p.cmp(&q))
    });
    place_leaves(&mut basins, &roots);
    Disconnectivity {
        basins,
        roots,
        forest,
    }
}

/// The path of the forest of `graph` from minimum `from` to minimum `to` (connections in the
/// order they are passed), or `None` when nothing joins them. Its highest transition state is
/// the lowest of all paths between the two.
pub fn lowest_barrier_path(
    graph: &Disconnectivity,
    connections: &[Connection],
    minima: usize,
    from: usize,
    to: usize,
) -> Option<Vec<usize>> {
    if from >= minima || to >= minima {
        return None;
    }
    if from == to {
        return Some(Vec::new());
    }
    let mut adjacent: Vec<Vec<(usize, usize)>> = vec![Vec::new(); minima];
    for &k in &graph.forest {
        let c = connections.get(k)?;
        adjacent[c.a].push((c.b, k));
        adjacent[c.b].push((c.a, k));
    }
    let mut came: Vec<Option<(usize, usize)>> = vec![None; minima];
    let mut seen = vec![false; minima];
    seen[from] = true;
    let mut queue = VecDeque::from([from]);
    while let Some(i) = queue.pop_front() {
        if i == to {
            break;
        }
        for &(j, k) in &adjacent[i] {
            if !seen[j] {
                seen[j] = true;
                came[j] = Some((i, k));
                queue.push_back(j);
            }
        }
    }
    if !seen[to] {
        return None;
    }
    let mut path = Vec::new();
    let mut at = to;
    while let Some((before, k)) = came[at] {
        path.push(k);
        at = before;
    }
    path.reverse();
    Some(path)
}

/// For every minimum, the highest point (transition state or minimum) on its lowest-barrier
/// path from `from` (`None` where nothing joins them; `from` itself has its own energy), in one
/// walk over the forest.
pub fn highest_on_the_way(
    graph: &Disconnectivity,
    connections: &[Connection],
    energies: &[f64],
    from: usize,
) -> Vec<Option<f64>> {
    let n = energies.len();
    let mut out: Vec<Option<f64>> = vec![None; n];
    if from >= n {
        return out;
    }
    let mut adjacent: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for c in graph.forest.iter().filter_map(|&k| connections.get(k)) {
        if c.a < n && c.b < n {
            adjacent[c.a].push((c.b, c.top));
            adjacent[c.b].push((c.a, c.top));
        }
    }
    out[from] = Some(energies[from]);
    let mut stack = vec![from];
    while let Some(i) = stack.pop() {
        let here = out[i].unwrap_or(f64::NEG_INFINITY);
        for &(j, top) in &adjacent[i] {
            if out[j].is_none() {
                out[j] = Some(here.max(top).max(energies[j]));
                stack.push(j);
            }
        }
    }
    out
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn c(a: usize, b: usize, top: f64) -> Connection {
        Connection { a, b, top }
    }

    #[test]
    fn minima_merge_at_their_lowest_transition_state() {
        // Three minima; the direct route 0-2 is higher than the way through 1.
        let energies = [0.0, 1.0, 2.0];
        let joins = [c(0, 1, 5.0), c(1, 2, 3.0), c(0, 2, 7.0), c(1, 1, 0.5)];
        let g = disconnectivity(&energies, &joins);
        assert_eq!(g.forest, vec![1, 0]);
        assert_eq!(g.roots.len(), 1);
        let root = &g.basins[g.roots[0]];
        assert_eq!((root.energy, root.lowest), (5.0, 0.0));
        // 1 and 2 merge at 3 under the root; the lower child first.
        let below: Vec<f64> = root.children.iter().map(|&b| g.basins[b].energy).collect();
        assert_eq!(below, vec![0.0, 3.0]);
        // Leaves 0, 1, 2 left to right, superbasins in the middle of their children.
        let xs: Vec<f64> = (0..3).map(|i| g.basins[i].x).collect();
        assert_eq!(xs, vec![0.0, 1.0, 2.0]);
        assert_eq!(root.x, 0.75);
        // The path from 0 to 2 goes through 1, its highest point 5 (not the 7 of the direct one).
        let path = lowest_barrier_path(&g, &joins, 3, 0, 2).unwrap();
        assert_eq!(path, vec![0, 1]);
        assert_eq!(
            highest_on_the_way(&g, &joins, &energies, 0),
            vec![Some(0.0), Some(5.0), Some(5.0)]
        );
    }

    #[test]
    fn a_pruned_graph_keeps_the_merges_of_the_whole_network() {
        // 0 and 2 meet only through 1 (at 5); drawn without 1 they still meet at 5.
        let energies = [0.0, 1.0, 2.0, 0.5];
        let joins = [c(0, 1, 5.0), c(1, 2, 3.0), c(0, 2, 7.0), c(3, 0, 4.0)];
        let g = disconnectivity(&energies, &joins);
        let p = prune(&g, &[true, false, true, true]);
        let leaves: Vec<usize> = p.basins.iter().filter_map(|b| b.minimum).collect();
        assert_eq!(leaves, vec![0, 2, 3]);
        assert_eq!(p.roots.len(), 1);
        let root = &p.basins[p.roots[0]];
        assert_eq!((root.energy, root.connection), (5.0, Some(0)));
        // 0 and 3 merge at 4 below the root; 2 joins them at 5.
        assert_eq!(p.forest, vec![3, 0]);
        let xs: Vec<f64> = p
            .basins
            .iter()
            .filter(|b| b.minimum.is_some())
            .map(|b| b.x)
            .collect();
        assert_eq!(xs, vec![0.0, 2.0, 1.0]);
        // Nothing kept, nothing drawn.
        assert!(prune(&g, &[]).basins.is_empty());
    }

    #[test]
    fn minima_that_nothing_joins_are_roots_of_their_own() {
        let energies = [0.0, 2.0, -1.0, 4.0];
        let g = disconnectivity(&energies, &[c(0, 1, 6.0), c(9, 1, 1.0), c(2, 3, f64::NAN)]);
        // 0-1 together; 2 and 3 alone; the lowest root first.
        assert_eq!(g.roots.len(), 3);
        assert_eq!(g.basins[g.roots[0]].minimum, Some(2));
        assert!(lowest_barrier_path(&g, &[c(0, 1, 6.0)], 4, 0, 3).is_none());
        assert_eq!(
            lowest_barrier_path(&g, &[c(0, 1, 6.0)], 4, 1, 1),
            Some(vec![])
        );
        assert!(lowest_barrier_path(&g, &[], 4, 0, 7).is_none());
    }
}
