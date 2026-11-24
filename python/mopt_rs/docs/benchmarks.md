# Benchmarks

## CCQN saddle-region step: RS-P-RFO vs RS-I-RFO (GFN2-xTB)

Script: `tools/bench/ccqn_xtb.py` (mopt-rs driven from Python, GFN2-xTB through tblite,
finite-difference Hessians for `-fc`). Protocol of Wu and Wang (J. Chem. Theory Comput. 2025):
for each reaction a reference transition state (TS) is located (AFIR path from the reactant,
highest point refined with RS-I-RFO and exact Hessians), and CCQN starts from
`TS + alpha (reactant - TS)` with `alpha` = 0.1, 0.3, 0.5, using the reactive bonds as the axis
(`cos(phi) = 0.1`, steps 0.1-0.2 Angstrom). Reactions and reactive atoms are from MultiOptPy's
test set: SN2 (OH- + CH3F), Diels-Alder (butadiene + methyl acrylate), Claisen rearrangement.
Each saddle-step method was combined with three Hessian updates (TS-BFGS, block Bofill, block
FSB), with and without an exact Hessian every 5 steps; 27 runs per method and Hessian setting.

A run counts as the reference TS when it converges to a first-order saddle point (one negative
Hessian eigenvalue after removing rigid-body modes) within 0.2 kcal/mol of the reference energy.

| | RS-P-RFO | RS-I-RFO |
|---|---|---|
| `-fc 5`: first-order saddle points | 20/27 | 15/27 |
| `-fc 5`: reference TS | 10/27 | 6/27 |
| updates only: reference TS | 7/27 | 5/27 |
| mean iterations of successful runs (`-fc 5`) | 19 | 16 |

Observations:

- RS-P-RFO reaches first-order saddle points and the reference TS more often; the difference is
  largest for SN2 (`-fc 5`: 6/9 vs 2/9), whose surface is flat around the ion-dipole complex; the
  RS-I-RFO failures there drift away from the reactants.
- Among the updates, TS-BFGS and block Bofill perform alike; block FSB finds no first-order saddle
  point without exact Hessians.
- The Claisen reference TS (34.5 kcal/mol above the reactant) is a high-lying saddle point from the
  AFIR path: all methods mostly converge to saddle points 16.2 and 31.3 kcal/mol lower, so the
  Claisen counts do not discriminate between the methods.

Decision: CCQN keeps RS-P-RFO with TS-BFGS as the default (`-opt ccqn`); RS-I-RFO remains
available as `ccqn_rsirfo`, and any update can be named (`ccqn_block_bofill`, ...). The sample
is small; the comparison will be repeated with the quantum-chemistry calculators.

On Muller-Brown (96 starting points per setting) both saddle-step methods reach the same success
rates; RS-P-RFO needs as many or fewer steps.

## Build profile of the tests

`cargo test` builds with the `test` profile, which inherits `dev`. Without per-package
settings faer and rayon ran unoptimized in tests. Three settings were measured from a clean
target directory with 4 build jobs: the build of `gfn1_xtb`, `cli_optmain` and
`golden_optimizers` (with the library and all dependencies), the summed memory of all build
processes at its peak, and the wall time of each test:

| Setting | Build (s) | Peak memory (GB) | `gfn1_xtb` (s) | `cli_optmain` (s) | `golden_optimizers` (s) | Total (s) |
|---|---|---|---|---|---|---|
| A: everything at opt-level 0 | 85 | 3.7 | 317 | 18.6 | 0.29 | 421 |
| B: dependencies at opt-level 3 | 119 | 3.3 | 222 | 16.6 | 0.25 | 358 |
| C: B and the workspace crates at opt-level 1 | 231 | 4.1 | 33 | 0.77 | 0.10 | 265 |

Decision: C (`[profile.dev.package."*"] opt-level = 3` and `opt-level = 1` in
`[profile.test]`). The three tests pass with the same assertions in all three settings;
opt-level changes no floating-point semantics (Rust has no fast-math), so the results are
those of the unoptimized build. `cargo build` and `cargo clippy` keep opt-level 0 for the
workspace crates. Other builds ran at the same time, so the times are indicative to about 30 %.

### Debug information of the tests

The build of the whole suite with full debug information (`debug = true`, the default of the
`dev` and `test` profiles) left a target directory of 52 GB. The same three test binaries as
above, built with setting C from a clean target directory with 4 build jobs, for three levels
of debug information (sizes from `du -sb`; the backtrace of a panic in a test with
`RUST_BACKTRACE=1`):

| `debug` | Target directory (GB) | The three test binaries (GB) | `libmopt_rs` rlib (GB) | Build (s) | Backtrace |
|---|---|---|---|---|---|
| `true` | 2.76 | 1.15 | 0.66 | 393 | functions with file and line |
| `"line-tables-only"` | 1.20 | 0.38 | 0.28 | 285 | functions with file and line (the same 8 frames) |
| `"none"` | 0.46 | 0.06 | 0.11 | 279 | function names only |

Decision: `debug = "line-tables-only"` in `[profile.dev]` and `[profile.test]`: the target
directory is 44 % of that with full debug information and backtraces keep file and line, which
`"none"` loses (a debugger still gets full information with `CARGO_PROFILE_DEV_DEBUG=true`).
The run times of `gfn1_xtb` (38, 198 and 242 s in the order of the table) were taken on a
machine loaded by other work that grew during the measurement and say nothing about the
setting; debug information does not change the generated code.

## Built-in engines: threads and cost

`examples/bench_engines.rs` evaluates each case in rayon pools of 1 and 8 threads (one
evaluation to warm up, then the median of 3 to 20 timed ones, fewer for the costly cases) and
compares the bits of the energy, gradient, Hessian and stress between the two pools. Release
build (`lto = "fat"`, no `-C target-cpu`), with other work running at the same time (load
average 3 to 8 during the runs): differences below about
30 % are noise (the EAM case, whose code did not change, moved by that much between runs).
Structures: caffeine (24 atoms) and taxol (113) of `tests/golden/gfnff.json`, 1331 water
molecules on a cubic grid (3993 atoms, 1331 fragments), the benzene crystal 2x1x1 of
`tests/golden/gfnff_pbc_reference.json` (96 atoms), an fcc copper cluster (864 atoms). GFN1-xTB
starts every SCC from the superposition (`scc_guess=sad`) so that the repeats do the same work.
"Same bits" compares 1 and 8 threads; `tests/thread_independence.rs` checks 1, 2, 4 and 8.

| case | before, 1 thread (ms) | before, 8 threads (ms) | same bits before | after, 1 thread (ms) | after, 8 threads (ms) | same bits after |
|---|---|---|---|---|---|---|
| gfnff gradient caffeine (24 atoms) | 0.24 | 0.24 | yes | 0.16 | 0.17 | yes |
| gfnff gradient taxol (113 atoms) | 8.46 | 9.55 | yes | 8.48 | 8.10 | yes |
| gfnff gradient water 3993 (3993 atoms) | 5924.60 | 4888.83 | NO | 3946.99 | 2283.97 | yes |
| gfnff gradient crystal (96 atoms) | 195.11 | 195.28 | yes | 191.13 | 214.75 | yes |
| gfnff hessian caffeine (24 atoms) | 4.02 | 4.70 | yes | 4.07 | 4.55 | yes |
| gfnff hessian taxol (113 atoms) | 160.71 | 164.90 | yes | 167.96 | 167.94 | yes |
| gfnff numerical hessian caffeine (24 atoms) | 30.51 | 30.11 | yes | 20.65 | 4.10 | yes |
| gfn0 gradient caffeine (24 atoms) | 4.28 | 7.60 | NO | 2.77 | 2.75 | yes |
| gfn0 gradient taxol (113 atoms) | 205.97 | 246.60 | NO | 54.37 | 52.14 | yes |
| gfn0 hessian caffeine (24 atoms) | 46.57 | 128.34 | NO | 25.29 | 13.83 | yes |
| gfn0 hessian taxol (113 atoms) | 15280.76 | 13018.55 | NO | 5555.47 | 2042.20 | yes |
| gfn0 numerical hessian caffeine (24 atoms) | 663.77 | 1140.38 | NO | 422.48 | 72.30 | yes |
| gfn1 gradient caffeine (24 atoms) | 87.68 | 172.54 | NO | 37.49 | 38.20 | yes |
| gfn1 gradient taxol (113 atoms) | 8132.17 | 9116.36 | NO | 1315.88 | 1003.06 | yes |
| gfn1 hessian caffeine (24 atoms) | 211.59 | 484.63 | NO | 108.55 | 84.74 | yes |
| gfn1 numerical hessian caffeine (24 atoms) | 12669.92 | 27767.68 | NO | 5853.01 | 959.69 | yes |
| d3 gradient taxol (113 atoms) | 6.49 | 6.48 | yes | 1.43 | 1.46 | yes |
| d3 hessian caffeine (24 atoms) | 0.62 | 0.82 | yes | 0.66 | 0.64 | yes |
| d4 gradient taxol (113 atoms) | 188.77 | 204.59 | yes | 215.25 | 220.04 | yes |
| d4 hessian caffeine (24 atoms) | 292.25 | 282.48 | yes | 275.13 | 46.53 | yes |
| eam gradient cu864 (864 atoms) | 19.43 | 19.59 | yes | 19.84 | 19.76 | yes |

"Before" is the base of the work (load average 5 during the run), "after" the final code (load
average 4 to 5); a second run of the final code at load average 8 to 9 gave the same bits
and times up to 70 % longer, the unchanged EAM case among them. Peak resident memory of one
water-grid gradient case (topology setup, two pools, `/usr/bin/time -v`): 2.25 GB before,
1.13 GB after. D4 (its three-body term, `O(N^3)`), the periodic GFN-FF (the Ewald sums) and the
analytic GFN-FF Hessian were not changed and cost what they did; the topology setup of GFN-FF
(7 s for the water grid, minutes for the 5696 atoms of the enzyme model: the topology
distances are a Floyd-Warshall loop, `O(N^3)`) was not part of this work either (it is now:
[below](#gfn-ff-topology-setup)).
What changed, by engine (the results differ from before by the order of summation only; every
golden test passes at its tolerance):

- GFN-FF: the derivatives of the coordination numbers are kept as a list of pairs instead of
  three dense `N x N x 3` arrays (about 1.2 GB at 5000 atoms) and contracted once for the
  dispersion, the electrostatics and the bonds; the EEQ is solved by a Cholesky factorization
  of the Coulomb block with the fragment constraints through its Schur complement (`W = L^-1 B`,
  `S = W^T W`) instead of an LU of the bordered matrix; the hydrogen-bond loop visits, for
  every A-B pair, only the H atoms bonded to A or B and those within the reach of the
  threshold of the triple (from a grid of the H atoms), in the same order, so the terms are
  the same; the terms of a few atoms are differentiated without allocations; the pair loops
  (repulsion, coordination numbers, EEQ matrix, electrostatic gradient, dispersion) run in
  fixed chunks of 32 atoms, in parallel from 256 atoms (`pes::native::parallel`). The bonding
  check of every gradient (a second neighbour search to warn about formed or broken bonds) is
  skipped when every atom is within half the margin of a structure searched before (the
  smallest distance of any pair from its bond threshold, for structures without metal atoms,
  whose bonds are those of the full neighbour list alone); its result is then that of the
  search, so the warnings are the same.
- GFN0-xTB and GFN1-xTB: the density and the energy-weighted density are one product of
  matrices over the occupied columns (they were triple loops); the iterations of the SCC skip
  the energy-weighted density; the columns of the analytic Hessians run in parallel, with the
  response densities back-transformed by rectangular products (the block of two empty
  orbitals is zero) and the Mulliken diagonal without a full product; in the GFN0-xTB Hessian
  the second derivatives of the D4 C6 through the CN and the charges are collected per pair and
  formed by one product `V^T M V` instead of a loop over the coordinates per pair; the EEQ
  matrices are factorized once per structure.
- D3: the gradient (every GFN1-xTB gradient) has closed-form first derivatives and contracts
  the CN chain over the pairs of the counting function, instead of hyper-dual Hessians per
  pair and a dense `N x 3N` Jacobian; the Hessian path is unchanged.
- D4: the reference C6 integrals are computed once per pair of elements present instead of
  per pair of atoms; the EEQ is factorized once per structure; the Hessian by differences
  evaluates its displaced structures in parallel.
- Central-difference Hessians (`-calc_opt hessian=numerical` of GFN-FF, GFN0-xTB, GFN1-xTB;
  D4; the rounds of O1NumHess when the calculator allows it) evaluate the displaced structures
  in parallel (`pes::native::fd_hessian`); the displaced SCCs of GFN1-xTB start from the
  charges of the structure itself (before: from those of the previous displacement), which
  changes the Hessian by the convergence of the SCC (below 1e-8 Hartree/Bohr^2).
- Dense kernels (`math::linalg`): faer's default parallelism takes the size of the pool, and
  its products, factorizations and eigendecompositions gave other bits in pools of 1 and 8
  threads (every GFN0-xTB and GFN1-xTB case above, GFN-FF at 3993 atoms). The engines now call
  kernels with a fixed degree of 8 tasks, on one thread for products of fewer than `2^21`
  multiplications, LU and Cholesky factorizations below order 1024 and eigendecompositions
  below order 1536 (`PARALLEL_PRODUCT`, `PARALLEL_ORDER`, `PARALLEL_EIGEN_ORDER`).

Thresholds: chunks of 32 atoms; the pair loops in parallel from 256 atoms (GFN-FF taxol, 113
atoms, runs the chunks one after the other; the 648-atom water grid of the test runs them in
parallel). The kernel thresholds come from `examples/bench_kernels.rs`: square matrices in a
pool of 8 threads, the median of seven, one task against 8 (milliseconds):

| order | product, 1 | product, 8 | eigen, 1 | eigen, 8 | LU, 1 | LU, 8 | Cholesky, 1 | Cholesky, 8 |
|---|---|---|---|---|---|---|---|---|
| 64 | 0.009 | 0.037 | 0.102 | 0.822 | 0.017 | 0.388 | 0.011 | 0.010 |
| 128 | 0.065 | 0.063 | 0.391 | 4.597 | 0.106 | 0.744 | 0.039 | 0.480 |
| 192 | 0.223 | 0.178 | 1.068 | 4.229 | 0.192 | 1.185 | 0.085 | 0.576 |
| 256 | 0.619 | 0.199 | 2.135 | 5.232 | 0.617 | 1.591 | 0.188 | 0.831 |
| 384 | 2.583 | 0.738 | 5.856 | 23.800 | 1.415 | 2.595 | 0.566 | 1.245 |
| 512 | 4.722 | 1.294 | 15.367 | 70.768 | 3.928 | 7.089 | 1.654 | 2.727 |
| 768 | 18.317 | 4.077 | 64.238 | 236.783 | 9.374 | 12.284 | 4.802 | 4.745 |
| 1536 | 151.042 | 32.682 | 879.623 | 767.490 | 81.015 | 85.558 | 30.462 | 17.747 |

Decision: products in 8 tasks from `2^22` multiplications (about `161^3`; 8 tasks lose at
`128^3` and win from `192^3`), LU and Cholesky factorizations from order 1024, the
eigendecomposition (faer's tridiagonalization with its partial sums per task) from 1536. The
choice depends on the sizes alone, so it does not change the bits between pools.

GFN1-xTB from the start of the superposition and from the charges of another structure
(`examples/bench_scc_start.rs`, caffeine and taxol, displacements 0.02, 0.05, 0.1 Bohr):

| molecule | thresholds | iterations from the superposition | from other charges | largest energy difference (Hartree) | largest gradient difference (Hartree/Bohr) |
|---|---|---|---|---|---|
| caffeine | default (charges 1e-10) | 32 | 24 to 29 | 2.3e-13 | 1.7e-11 |
| caffeine | charges 1e-12 | 38 | 30 to 35 | 5.7e-14 | 3.7e-13 |
| taxol | default | 44 | 25 to 29 | 7.1e-13 | 6.4e-11 |
| taxol | charges 1e-12 | 52 | 33 to 37 | 2.6e-13 | 4.4e-13 |

Decision: the defaults stay. The spread is below 1e-12 Hartree and 1e-10 Hartree/Bohr
already, a hundredth of the convergence criteria of any optimizer; tighter charges would cost
6 to 8 iterations per structure (about 20 %) for nothing measurable. A run that must give the
same bits whatever the order of the structures uses `-calc_opt scc_guess=sad`
(`docs/source/calculators.md`, "Threads of the built-in engines").
## Whole commands: faer's default parallelism against sequential

Outside the engines, every product, factorization and eigendecomposition that names no
parallelism takes faer's global setting, whose default sizes the work from the pool of the
call (`docs/source/running.md`, "Threads and reproducibility"). The commands now set it to
sequential. `examples/bench_commands.rs` runs each case in a pool of 1 and of 8 threads
(`-core` alike), with the default (`Par::rayon(threads)`, what the default takes in the pool
of the command) and sequential. Release build, load average 11 to 16 during the run (other
workers building): differences below about 15 % are noise. Wall times (s):

| case | default, 1 | default, 8 | sequential, 1 | sequential, 8 |
|---|---|---|---|---|
| `opt`: taxol (113 atoms), GFN-FF, RS-I-RFO with an exact Hessian at every step, 8 steps | 3.99 | 3.84 | 3.48 | 3.51 |
| `dic`: the same in delocalized internal coordinates | 1.79 | 1.79 | 1.85 | 1.84 |
| `neb`: HCN to HNC, `-p 5` images between the three inputs, GFN0-xTB, 8 iterations | 0.64 | 0.68 | 0.63 | 0.62 |
| `freq`: 343 argon atoms (1029 coordinates), LJ | 4.29 | 4.40 | 4.54 | 4.43 |
| `kinetics`: propagation of 300 states, 897 bottlenecks, 20 times | 1.20 | 1.02 | 1.17 | 1.13 |
| `md`: 125 argon atoms, a bond constraint, 200 steps | 1.32 | 1.38 | 1.30 | 1.40 |

Decision: no call site outside the engines grows by more than the noise, so none was
switched to the kernels of `math::linalg`. The dense work of these commands is either in
the engines (already on the kernels) or below the sizes from which faer splits it (its
tridiagonalization from about order 314, products from `16^3` multiplications), where
8 tasks were slower than one anyway (the table of the kernels above). With the default the
same optimization of a 125-atom argon cluster with exact Hessians gave other last bits in a
pool of 2 threads than of 1 (`tests/thread_independence_commands.rs`, run with the setting
disabled); sequential it gives the same bits in pools of 1, 2, 4 and 8.
## GFN-FF topology setup

The topology distances of xtb's `gfnff_ini` are a Floyd-Warshall loop over the bonded radii in
single precision whose steps skip every pair farther than `tdist_thr` (12 Angstrom) from the
intermediate atom; the edges are weighted, so a breadth-first search over the bonds would not
give its numbers (nor would a Dijkstra search: the sums of single-precision lengths would be
taken in another order). The loop now visits, in step `k`, only the pairs of atoms within the
threshold of `k` (the row of `k`, listed once per step: it does not change in the step, and the
matrix stays symmetric), in the same order, so every number is xtb's. The EEQ of the topology
charges is factorized once per pass and solved again for every pi fragment (xtb factorizes it
for every right-hand side: `2 + picount` LU factorizations of order `N + nfrag`). A test
compares the whole setup with xtb's algorithms on the 173 golden molecules and the periodic
references (`topology::tests::the_setup_is_that_of_the_reference_algorithms`).

`topology::tests::time_of_the_topology_distances` (test build, `MOPT_FULL_MEASUREMENTS=1`,
pool of 16 threads, load average 5 to 7; seconds):

| structure | distances, now | distances, xtb's loop | one EEQ factorization and solve |
|---|---|---|---|
| water grid, 648 atoms | 0.003 | 0.010 | 0.084 (216 fragments) |
| water grid, 3993 atoms | 0.30 | 0.60 | 4.07 (1331 fragments) |
| enzyme model, 5696 atoms | 0.66 | 2.38 | 4.85 (7 fragments) |

So the loop was not what made the setup of large systems slow: its skips already kept it near
`N^2`. The factorizations were: one per pi fragment, each about 5 s at 5696 atoms. The
setup of the enzyme model now takes 34.5 s on one thread, and one energy and gradient 8.7 s
(`examples/bench_commands.rs gfnff`, release, load average 8; peak resident memory 1.7 GB;
the run on 8 threads was stopped by the memory guard while other builds ran). The water grid
of 3993 atoms: setup 23.3 s on one thread and 11.7 s on eight, energy and gradient 8.1 and
5.2 s. What remains of the setup is the two dense LU factorizations of order `N + nfrag`
(xtb's algorithm) and the `N x N` matrices of the distances.

## Verlet lists of the metal potentials

EAM, MEAM and Tersoff searched their neighbours afresh at every evaluation. They now keep a
Verlet list (`neighbors::VerletList`): the pairs within the cutoff plus a skin of 1 Bohr,
from the same linked cells, kept until an atom has moved half the skin, an atom has been
wrapped through the cell, or the cell has changed; a call filters the kept pairs with the
same operations as a fresh search, so the neighbours, their vectors and their order are the
same bits (`tests/verlet_lists.rs`: dynamics and a cell relaxation of copper with EAM and
MEAM, silicon with Tersoff; `neighbors::tests::a_verlet_list_gives_the_bits_of_a_fresh_search`).
Each calculator keeps up to eight lists (structures evaluated in turn, as the images of a
band) behind a lock that is held only to look one up or store one. `bench_commands verlet`:
200 gradients along small random steps, release, load average 9 (seconds):

| system | fresh search | kept lists |
|---|---|---|
| Cu864, periodic | 18.8 | 16.1 |
| copper slab, 500 atoms | 8.2 | 8.1 |

The search is 15 % of an EAM gradient of Cu864 and less in the slab; a dynamics run of Cu864
(`bench_commands eam_md`, 100 steps) took 15.8 to 16.3 s. Skin: 1 Bohr, about 39 % more candidates than the cutoff alone for copper (cutoff 10.8 Bohr,
`(11.8 / 10.8)^3`); other skins were not measured, since the search is a small share of the
gradient at any skin.

## Log files

The file log flushed after every line, behind the lock of the logger; a message of many lines
(the structure of every iteration) was a write per line. It now flushes once per message, so
every message is in the file whole when the call returns (a run that is killed loses no
message it had finished). `bench_commands log`, 20000 messages of 30 lines: 1.24 s flushed per
line, 0.12 s per message. A Lennard-Jones optimization of 64 atoms for 20000 iterations and
100000 steps of dynamics of 8 atoms took 0.95 to 1.06 s and 18.7 to 20.8 s. The trajectory
files are still opened for every frame: their writers live as long as the observers that
hold them, and one observer of the mapper keeps the trajectory of every job it has seen, so a
file held open per writer would keep one file handle per job.