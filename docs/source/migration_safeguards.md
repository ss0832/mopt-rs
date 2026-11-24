# Differences from MultiOptPy: numerical safeguards

The guards against division by zero, arguments outside the domain of a function, overflow, and
values that are not finite, compared with MultiOptPy's.

## Numerical safeguards

MultiOptPy's guards against division by zero, arguments outside the domain of a function,
overflow, and values that are not finite were read file by file and compared with mopt-rs.
Guards that are sound are kept with the same thresholds; the ones below differ.

### Kept as in MultiOptPy

| Guard | MultiOptPy | mopt-rs |
|---|---|---|
| RFO secular equation: denominators below `1e-30` / `1e-20`, Brent bracket with Newton fallback | `Optimizer/rsirfo.py` | `opt/step/rfo_core.rs` |
| RS-I-RFO: modes with `abs(eigenvalue) < 1e-6` dropped, identity Hessian when the eigenvalues are not finite, steepest descent when the step is not finite, no trust update for `abs(predicted) < 1e-10` | `Optimizer/rsirfo.py` | `opt/step/rsirfo.rs` |
| Hessian updates: terms with `abs(denominator) < 1e-10` dropped, mixing weights clamped to `[0, 1]`, no update for `norm(s)` or `norm(y) < 1e-10` | `Optimizer/hessian_update.py` | `opt/update/formulas.rs`, `opt/update/mod.rs` |
| Block updates: rank filter `1e-8`, inverse with `1e-10` regularization and pseudo-inverse fallback, weights that are NaN set to 0 | `Optimizer/block_hessian_update.py` | `opt/update/block.rs` |
| Outer trust radius: predicted change kept away from zero by `1e-8`, radius clamped to its bounds | `Optimizer/trust_radius.py` | `opt/trust.rs` |
| RMS over components above `1e-10` in the convergence test | `optimization.py:ConvergenceChecker` | `opt/convergence.rs` |
| Angle restraints: series in `1 - cos` within `1e-3` rad of 0 and pi; dihedral and out-of-plane restraints switched off smoothly for collinear atoms; distances kept above `1e-12` | `Potential/keep_*_potential.py` | `pes/bias/restraints.rs`, `coord/primitives.rs` |
| MECI / MESX projections: gradient differences below `1e-8` | `ModelFunction/opt_me*.py` | `pes/combine/projection.rs` |
| EMT: atoms without neighbours | `Calculator/emt_calculation_tools.py` | `pes/native/emt.rs` |

Every arc cosine of an automatic-differentiation scalar clamps its argument to `[-1, 1]` and
its derivative away from the end points (`math/ad`), so the bias potentials, constraints, and
user expressions cannot produce NaN from a cosine that rounding put outside the interval.

### Saddle searches learn negative curvature

MultiOptPy skips every Hessian update with `s . y <= 0` (`RSIRFO.update_hessian`). That is the
curvature condition of a minimization; in a saddle search a step that runs mostly along the
transition vector has exactly this sign, and the Hessian then does not learn the curvature of
the mode that is maximized. mopt-rs uses these pairs when `saddle_order > 0` and the update
formula has no BFGS term: SR1, PSB, Bofill (also `cfd_bofill`, `pcfd_bofill`), MSP, TS-BFGS,
and the block Bofill forms (`block_bofill`, `block_bofill_weighted`, `block_cfd_bofill`,
`block_cfd_bofill_weighted`: block SR1 mixed with block PSB). Minimizations are unchanged, and
so are FSB, BFGS, `flowchart`, and the block FSB and block BFGS forms: the BFGS term divides
by `s . y`, and with a small negative value it puts a large negative curvature along `y`.

Measured with one exact Hessian at the start, or one every 10 or 20 iterations:

| Surface | Update | MultiOptPy rule | with the pairs |
|---|---|---|---|
| Muller-Brown, two saddle points from eight starts (total iterations, exact Hessian at the start only) | Bofill | 65 | 41 |
| | MSP | 64 | 41 |
| Diels-Alder transition state of butadiene + ethylene, GFN0-xTB, four guesses x three Hessian intervals (total iterations; all 12 runs reach the transition state either way) | Bofill | 324 | 318 |
| | MSP | 330 | 325 |

In a molecule such pairs are rare (the many modes of positive curvature dominate `s . y`), so
the gain is small there; no run became slower by more than one iteration. With FSB the pairs
helped on the Muller-Brown surface but not on the Diels-Alder reaction (5 of 12 runs reach
the transition state with them and 5 of 12 without, different ones). FSB is an update for
minimizations in any case and keeps MultiOptPy's rule.

The block forms, measured with the Hessian of the start only on the Muller-Brown surface
(two saddle points from eight starts) and with a Hessian every 10 or 20 iterations or at the
start only for the Diels-Alder transition state (one guess):

| Update | Muller-Brown, MultiOptPy rule | with the pairs | Diels-Alder, MultiOptPy rule | with the pairs |
|---|---|---|---|---|
| `block_bofill` | 62 | 55 | 22, 22, 38 | 24, 22, 33 |
| `block_bofill_weighted` | 82 | 46 | | |
| `block_cfd_bofill` | 48, one start not converged | 75, all converged | | |
| `block_fsb` | 134 | 60 | 24, 24, 35 | 24, 24, 35 |
| `block_cfd_fsb` | 40, two starts not converged | 125, one not converged | | |
| `bofill` (one pair, for comparison) | | 40 | | 22 (every 20) |

The block Bofill forms therefore take the pairs. FSB and block FSB do not, by design rather
than by this measurement: their BFGS term is there to keep the curvature positive, which makes
them updates for minimizations, not for saddle searches (use Bofill, MSP, SR1, PSB, or a
block Bofill form there, or CCQN with its TS-BFGS update; a saddle search started with BFGS, FSB, `flowchart`, or one of
their block forms says so in the log and runs as asked).

What this measurement does not show: the cases are two- and three-variable surfaces and one
molecular transition state that the single-pair Bofill update finds without difficulty (40
iterations on the Muller-Brown starts, 22 for the molecule, the same as `block_bofill` there).
The block updates exist for systems on which single-pair updates converge with difficulty,
which is where they proved useful in MultiOptPy; no such system is among these cases, so the
numbers above say that the block Bofill forms should take the pairs, not how block and
single-pair updates compare. The next table is for such a system.

A system on which single-pair updates do converge with difficulty: the transition state of
the epoxidation of ethylene by peroxyacetic acid, with GFN0-xTB (15 atoms; even with an exact
Hessian at every iteration RS-I-RFO takes 54 to 59 iterations from these guesses). Iterations
of `crates/mopt-rs/examples/bench_ts_updates.rs` from five guesses with the exact Hessian at
the start only, or every 20 or 10 iterations (`n`: not converged in 200 iterations, `x`:
converged to another stationary point):

| Guess | Hessian every | rsirfo_bofill | rsirfo_msp | rsirfo_block_bofill | rsirfo_block_bofill_weighted | rsirfo_block_cfd_bofill | gpmin |
|---|---|---|---|---|---|---|---|
| O-O 1.98, O-C 1.86, shifted 0.2 | start only | 115 | 130 | 112 | 167 | 161 | n |
| O-O 1.98, O-C 1.86, shifted 0.2 | 20 | 71 | 73 | n | 122 | 136 | 158 |
| O-O 1.98, O-C 1.86, shifted 0.2 | 10 | 60 | 60 | 138 | 172 | 128 | 157 |
| O-O 1.98, O-C 1.86, turned 30 | start only | 125 | 137 | 171 | 162 | n | n |
| O-O 1.98, O-C 1.86, turned 30 | 20 | 77 | 79 | 169 | 143 | 63 | n |
| O-O 1.98, O-C 1.86, turned 30 | 10 | 65 | 65 | 164 | 112 | 120 | 140 |
| O-O 1.98, O-C 1.86, turned 90 | start only | x | x | 127 | 154 | 151 | n |
| O-O 1.98, O-C 1.86, turned 90 | 20 | x | x | 134 | n | 162 | n |
| O-O 1.98, O-C 1.86, turned 90 | 10 | x | x | 112 | 114 | 122 | 194 |
| O-O 1.98, O-C 1.86, displaced | start only | 131 | 135 | 57 | 59 | 160 | n |
| O-O 1.98, O-C 1.86, displaced | 20 | 71 | 71 | 140 | n | 138 | n |
| O-O 1.98, O-C 1.86, displaced | 10 | 60 | 60 | 114 | 117 | 151 | 167 |
| O-O 1.90, O-C 2.00, shifted 0.2, turned 30 | start only | 114 | 164 | 43 | 172 | 134 | 107 |
| O-O 1.90, O-C 2.00, shifted 0.2, turned 30 | 20 | 74 | 72 | 170 | 165 | n | 89 |
| O-O 1.90, O-C 2.00, shifted 0.2, turned 30 | 10 | 77 | 82 | 113 | n | 149 | 76 |
| **found** | | 12 of 15 | 12 of 15 | 14 of 15 | 12 of 15 | 13 of 15 | 8 of 15 |
| **iterations of the runs that found it** | | 1040 | 1128 | 1764 | 1659 | 1775 | 1088 |

- `block_bofill` finds the transition state in 14 of the 15 runs, the single-pair updates in
  12: from the guess with the ethylene turned by 90 degrees, Bofill and MSP end at another
  stationary point with every Hessian interval, and the block forms reach the transition
  state.
- With the Hessian of the start only, `block_bofill` needs fewer iterations than Bofill from
  three of the five guesses (112 against 115, 57 against 131, 43 against 114) and more from
  one (171 against 125).
- With exact Hessians every 10 or 20 iterations, the single-pair updates are faster where they
  find the transition state (60 to 82 iterations against 112 to 170).
- GPMin finds it in 8 of 15 runs; it is not a method for this system.

`-modekill` (see running.md) is the remedy for the runs marked `x`.

Tests: `tests/numerical_safeguards.rs::saddle_searches_learn_negative_curvature_from_their_steps`,
`::block_bofill_updates_learn_negative_curvature_too`,
`::negative_curvature_pairs_and_the_update_formulas`,
`::diels_alder_transition_state_with_updates_from_negative_curvature_pairs`.

### RFO step with a gradient that is zero within rounding

The secular equation of the RFO step is solved below its first pole, the lowest curvature
whose gradient component counts (its square above `1e-20`). With no such component the
lowest curvature itself was returned as the eigenvalue, and the step along that mode
became the gradient divided by the smallest number the solver allows, `1e-20`: a point
with a gradient of `1e-11` was sent away by `1e9` (cut to the trust radius or the step
limit of the caller). An optimization ends long before its gradient is that small, but an
image of a chain that has converged while others still move met this in every iteration;
it showed as converged images that jumped by 0.05 to 0.2 Bohr every few iterations. The
step is now the Newton step along the modes of positive curvature and nothing along the
others (`opt::step::rfo_core::solve_rfo`). MultiOptPy's solver (`RSIRFO`) guards the same
division with `1e-20`. Test:
`opt::step::rfo_core::tests::a_gradient_that_is_zero_within_rounding_gives_no_long_step`.

### Restricted step with negative curvature and almost no gradient along it

The restricted-step RFO searches the scaling `alpha` (up to `alpha_max = 1000`) for the step
on the trust radius. With a negative curvature and a gradient that has almost nothing along
that mode, the step along it is about `|lambda| / (alpha g)` for every `alpha`, so no `alpha`
within the bound makes it short enough. MultiOptPy then returns the step of its last Newton
cycle (which may be far longer than the radius) or the steepest-descent step on the radius,
which ignores the curvature: where the gradient points along a stiff mode it raises the
quadratic model. Both happen at a structure that converged at a saddle point and was left
along its mode by the mode kill (the exact Hessian has the negative curvature, the gradient
is zero along it), and where a quasi-Newton update turned a curvature negative. In 3000
random minimizations (up to two negative curvatures, gradient components from `1e-9` to
`3e-2`, radii from 0.05 to 0.5 Bohr) 214 steps raised the model (by up to 0.15 Hartree) and 10
were longer than the radius (one by a factor of six million); on an enzyme on the ONIOM
surface the minimization climbed for a hundred iterations after the mode kill. mopt-rs takes
the step of the trust-region problem there instead: the shifted Newton step
`-g_i / (lambda_i + mu)` with the shift that puts it on the radius, and in the hard case the
rest of the radius along the lowest mode, downhill (Moré and Sorensen 1983;
`opt::step::rfo_core::trust_region_step`); its model never rises. The RFO steps of every
`alpha` lie on the same curve of shifted Newton steps, so wherever the search in `alpha`
finds its step the result is MultiOptPy's. Tests:
`opt::step::rsirfo::tests::a_minimization_step_never_climbs_its_model`,
`opt::step::rfo_core::tests::negative_curvature_with_almost_no_gradient_along_it_stays_on_the_radius`,
`opt::step::rfo_core::tests::the_trust_region_step_handles_the_hard_case_and_the_newton_step`.

### Mode kill at convergence

New in mopt-rs (`-modekill`, off by default; `opt::mode_kill`). MultiOptPy accepts whatever
stationary point an optimization converges to; its `ModeKill` class removes surplus imaginary
modes after an IRC only. With the option, the optimizer requests the Hessian at the converged
structure and, when it has more imaginary modes than the order asked for, leaves the
structure along the surplus modes and goes on (see running.md). A mode is imaginary by the
criterion of MultiOptPy's `saddle_check`: an eigenvalue of the mass-weighted Hessian below
minus 1e-8 (about 0.51 cm^-1), the same in the optimizers, the IRC, and the mapper
(`opt::mode_kill::IMAGINARY_CURVATURE`; it replaced a threshold of 1e-4 Hartree/Bohr^2 on the
curvature without masses, which let collective modes of whole molecules of 10 cm^-1 pass). It works for minimizations
(`-order 0`), saddle points of any order, molecules in every coordinate system, and plain
variables. Tests: `opt::mode_kill::tests`, `tests/optimize_variables.rs`,
`tests/cli_variables.rs`, `tests/numerical_safeguards.rs` (the epoxidation), and
`tests/test_variables.py`.

### Steps and model Hessians that are not finite

MultiOptPy checks the step only inside RS-I-RFO. With any other method a step that is not
finite moves the structure to NaN, and the run fails later inside the calculator. mopt-rs
checks the final step of every method (after enhancers, blending, and the trust radius) and
stops with an error that names the method; the structure is left as it was.

A model Hessian that is not finite (a legacy model at a degenerate structure) is taken as it is
by MultiOptPy; RS-I-RFO then falls back to steepest descent for the rest of the run, because
the stored Hessian stays NaN through every update. mopt-rs keeps the Hessian in use and writes
a note into the log.

Tests: `tests/numerical_safeguards.rs::steps_that_are_not_finite_are_refused`,
`::model_hessians_that_are_not_finite_are_not_used`.

### Model Hessians on surfaces that are not invariant under rigid-body motions

The model Hessians of the bonded curvature (Lindh, Swart, Fischer, Schlegel) have no curvature
along the translations and rotations. RS-I-RFO drops modes whose eigenvalue is below `1e-6` in
size, which are rigid-body modes once the gradient is projected. On a surface that is not
invariant under those motions (an external field or a tether inside the calculator, a
calculator whose `is_rigid_body_invariant` is false) nothing is projected: the gradient keeps
its components along them, the step never goes there, and the updates never learn the
curvature, because no step has a component along them. MultiOptPy takes the model as it is and
the run stalls with a constant gradient (on the tethered pairs of `tests/combined_surfaces.rs`
with Lindh's model, 500 iterations at a gradient norm of 0.055, against 32 iterations from the
unit matrix). mopt-rs replaces the block of the rigid-body motions of the model by the
curvature of a start without a model (the unit matrix, or the scale of the method), before a
combination of two states takes it. Constraints that pin the frame (frozen atoms, fixed
coordinates, orientations) are left as they were: they have their own projection, and the bonds
to the frozen atoms give the model its curvature there. Periodic systems are left as they were.

Test: `tests/combined_surfaces.rs::a_model_hessian_has_curvature_along_rigid_body_motions_of_a_surface_that_has_it`.

### Legacy model Hessians at linear arrangements

`bend2` and `outofplane2` of `ModelHessian/calc_params.py` take the arc cosine of a dot product
of unit vectors, which rounding puts outside `[-1, 1]` for some linear arrangements: the angle
and the B-vectors become NaN (the branch meant for this case tests a quantity that is always 2).
The `legacy_` models clamp the cosine, so that the angle is 0 or pi and the existing guards set
the B-vectors to zero; everywhere else the values are unchanged.

Test: `model_hessian::legacy::common::tests::linear_arrangements_stay_finite`.

### MultiOptPy safeguards that are not taken over

- **Level shift before the eigendecomposition** (`RSIRFO.compute_eigendecomposition_with_shift`,
  on when the condition number exceeds `1e8`): `H + c I` is diagonalized and `c` subtracted.
  The eigenvalues of a symmetric matrix are computed with an absolute error of the order of
  `eps * norm(H)` with or without the shift and the eigenvectors are the same, so the shift
  improves nothing; near-zero modes are handled by the `1e-6` filter.
- **Initial Hessian scaling** (`ModelHessianUpdate._auto_scale_hessian`, Nocedal and Wright
  eq. 6.20): never called in MultiOptPy. Not added, to keep the trajectories of the
  quasi-Newton methods; a model Hessian (`-modelhess`) is the better starting point, and
  L-BFGS has its own scaling (see "L-BFGS" above).
- **Clipping in the Tersoff potential** (`safe_exp`, `safe_value`: exponents to `+-50`,
  intermediate values to `+-1e6`): it makes energy and gradient inconsistent. mopt-rs
  differentiates the unclipped energy exactly and handles the one singular point (`zeta = 0`,
  where the bond order is 1) explicitly.
- **`rho = 1000` in L-BFGS** when `abs(y . s) < 1e-10`: replaced by the curvature condition
  (see "L-BFGS" above).
- **Inverse of the G matrix** (`calc_inv_G_mat`): singular values below the threshold are kept
  as they are instead of being set to zero; mopt-rs uses the pseudo-inverse
  (`coord/system/transform.rs`).

Guards MultiOptPy does not have and mopt-rs adds: FIRE and LARS at a zero gradient or step
(`0/0` in MultiOptPy), and the checks of every calculator result for values that are not
finite (`pes::evaluate_checked`).
