# Differences from MultiOptPy: optimization

Hessian updates, the step methods and their enhancers, trust radii, coordinate systems, periodic
cells (stress and variable cells), the optimization loop and its output, constraints, and model
Hessians.

## Hessian updates

### Block (multi-secant) updates keep their history

MultiOptPy's `BlockHessianUpdate` appends the newest secant pair and removes the oldest pair
right after each update (`_push_history` followed by `delete_old_data`). With one pair stored at
a time, the history is empty after every call, so every "block" update equals the corresponding
single-secant update.

This was confirmed by replaying the reference class the way `RSIRFO.update_hessian` calls it
(one persistent instance, one call per iteration) on a 5-dimensional quadratic: after each of six
calls the stored history length was 0, and the Hessian matched single-secant BFGS to within
`3e-15`.

mopt-rs keeps a window of the latest 8 pairs and imposes the latest 4 simultaneously, as the
reference's `block_size` / `max_window` parameters intend. Test:
`opt::update::block::tests::block_bfgs_satisfies_all_secant_conditions`.

### Block history restarts at an exact Hessian

With `-fc N` the exact Hessian replaces the quasi-Newton one every `N` iterations, and the newest
secant pair is then applied on top of it, as in MultiOptPy. Because mopt-rs keeps the block
window, the older pairs in it would also be imposed again on the exact Hessian, although the
exact Hessian already contains that information and the old pairs were taken elsewhere on the
surface. The window is therefore cleared when an exact Hessian arrives (MultiOptPy's block
updates kept no history at all). Without this, `ccqn_rsirfo_block_bofill` with `-fc 5` did not
find the Muller-Brown saddle from a start 60% of the way into the well (300 iterations); with
it, the search converges in 22. Test:
`ccqn_reaches_the_saddle_from_inside_the_wells_with_exact_hessian_every_5_steps`.

### Updated Hessians keep no rigid-body curvature (`-update_proj`, mopt-rs 2.1.1)

MultiOptPy projects the translations and rotations out of the copy of the Hessian a step is
computed from (`Calculationtools.project_out_hess_tr_and_rot_for_coord`), never out of the
Hessian it keeps: every update adds curvature along them (the secant pair joins two
structures and is not orthogonal to the rigid motions of the newer one), the next update starts
from that Hessian, and the trust radius predicts with it. mopt-rs up to 2.1.0 did the same, and
kept even the exact Hessian of `-fc` unprojected. From 2.1.1 the kept Hessian is projected with
the rigid-body basis of its structure after every update and when an exact Hessian is taken
over, in the optimizers, `-vc`, the chains, the IRC, and DS-AFIR; `-update_proj off` restores
the updated Hessians of 2.1.0 (the exact ones are projected either way). Where constraints pin
the frame (`-fix`) nothing more is projected. Details in
[optimization.md](optimization.md), "The rigid-body motions of updated Hessians". The tests that
compare with MultiOptPy's numbers step by step (`tests/golden_neb.rs`) run with the option off.
Tests: `tests/update_projection.rs`.

### Block PSB (inside block Bofill) is a genuine multi-secant update

The reference's `_block_PSB_update` adds independent single-secant PSB corrections computed from
the same `B`. The sum does not satisfy the block secant conditions: for 3 pairs from a quadratic
the residual `||(B + dB) S - Y||` was `1.34` (block SR1 and block BFGS: below `1e-15`), and block
Bofill inherited a residual of `0.27`. mopt-rs uses the symmetric multi-secant PSB
`dB = R M^-1 S^T + S M^-1 R^T - S M^-1 sym(R^T S) M^-1 S^T` (`M = S^T S`, `R = Y - B S`).

### "Weighted" block FSB / Bofill no longer double-count

The reference's weighted-subspace variants apply block SR1 to `(S W, Y W)` and the partner update
to `(S (1 - W), Y (1 - W))` and add the results. Block SR1, BFGS, and PSB are invariant under
column scaling, so the result equals the full SR1 correction plus the full partner correction;
this was confirmed numerically (`np.allclose` against `B + dSR1 + dBFGS` / `B + dSR1 + dPSB`).
The secant residuals were `3.2` (FSB) and `3.8` (Bofill). mopt-rs mixes the two corrections with
a single weight, the per-pair weights averaged with weights `||s_j||^2`, which keeps the secant
conditions.

### Block updates scale and screen their pairs

MultiOptPy's block updates (and mopt-rs's until this was found) test their pairs with
fixed numbers: steps whose singular values are above `1e-8` count as independent, and
matrices are regularized by `1e-10`. The block formulas do not change when a pair is
scaled, but these tests do, so that a pair of a step of `1e-6` was treated differently
from one of `0.1`; and pairs whose steps are almost parallel passed the test although
imposing them together divides what the surface is not quadratic by by the small angle
between them. mopt-rs now

- divides every pair by the length of its step before the formulas are applied;
- uses a pair, going from the newest to older ones, only if its step makes an angle of at
  least `asin(0.1)` with the steps already taken (`BlockUpdater::independence`);
- leaves a pair out of block SR1 when its residual is perpendicular to its step (below
  `1e-8` in the cosine), the usual safeguard of SR1.

The value 0.1 is a judgement (the error of a pair is amplified by at most ten); it was not
tuned. With these changes the chain relaxation above converges with the window of pairs as
well as with one pair. `examples/bench_ts_updates.rs` has not been run again with them.
Tests: `opt::update::block::tests`.

### Block Bofill on current NumPy

`_block_PSB_update` calls `float()` on a `1 x 1` array, which raises `TypeError` on NumPy 2.5
(it only warns on the pinned NumPy 2.2). Not applicable to mopt-rs; noted for users of the
reference.

### Block BFGS curvature filter is applied

The reference computes which columns satisfy `y_j . s_j > 1e-12` but then uses all columns.
mopt-rs drops the columns that fail the check.

## RS-I-RFO

### Restricted-step scaling of the RFO secular equation

RS-RFO restricts the step to the trust radius by scaling the augmented Hessian with
`diag(alpha I, 1)`. The lowest eigenvalue `nu` then solves
`nu + sum_i g_i^2 / (lambda_i - alpha nu) = 0`, and the step is
`s_i = -g_i / (lambda_i - alpha nu)`. In the scaled variables used by MultiOptPy
(`lambda' = lambda / alpha`) the squared gradient must be divided by `alpha`;
`RSIRFO._solve_secular_more_sorensen` divides it by `alpha^2`. Substituting `mu = alpha nu`
shows that the reference equation, `mu + sum_i g_i^2 / (lambda_i - mu) = 0`, no longer depends on
`alpha`, so every `alpha` gives the same step: the Brent bracket never forms, the Newton
refinement stalls, and the internal trust radius is not enforced (only the outer step cap scales
the step down). For the eigen-system `lambda = (0.1, 0.4, 2.0)`, `g = (0.3, -0.2, 0.5)` the
reference formula gives `||s|| = 0.7398` for every `alpha` from `1e-6` to `1e3`.

mopt-rs uses the correct scaling, so the step length decreases monotonically with `alpha` and
the step lands on the trust radius. Tests: `opt::step::rfo_core::tests::step_norm_decreases_with_alpha`,
`opt::step::rfo_core::tests::restricted_step_has_trust_radius_norm`.

## L-BFGS

### Curvature condition and scaled initial Hessian

MultiOptPy's L-BFGS stores every secant pair and replaces `1 / (s . y)` by `1000` when
`|s . y| < 1e-10`, and always uses `H0 = 1 / alpha = 0.1`. The absolute threshold depends on the
energy scale: on an argon cluster (curvatures around `6e-4` Hartree/Bohr^2) the method reached the
minimum energy and then stalled with a maximum force of `6e-6` Hartree/Bohr and steps of
`1e-6` Bohr for hundreds of iterations. mopt-rs skips pairs with
`s . y <= 1e-8 ||s|| ||y||` and scales the initial inverse Hessian with `gamma = s . y / y . y`
of the newest pair (Nocedal and Wright, eq. 7.20); `1 / alpha` is used until a pair is accepted.
Tests: `opt::step::lbfgs::tests::negative_curvature_pairs_are_skipped_and_quadratic_is_solved`
and the argon-cluster test in `tests/optimize_model_surfaces.rs`.

## First-order optimizers

Steepest descent (`gradientdescent`, `steepest_descent`, `mwgradientdescent`), Adam (`adam*`),
RAdam (`radam`, `ranger*`), AdaBelief, Eve, and FIRE 2.0 are ported as they are in MultiOptPy;
their first 30 iterations on the Ar7 cluster agree with MultiOptPy to `1e-9` Bohr
(`tests/golden_optimizers.rs`; the half step back of FIRE 2.0 is covered by a unit test instead,
see below). The module headers state where MultiOptPy's variants differ from the cited papers
(position of `eps` in Adam, no bias correction in AdaBelief, the AdaBelief variance in RAdam).
Fixed-step methods are stable only for curvatures below `2 / delta` in the units of the problem
(`delta = 1` for steepest descent).

### Conjugate gradients have a line search

MultiOptPy's `ConjgateGradient` returns `-g` first, then `alpha d` with
`alpha = g . d / (d . d + 1e-8)` and a direction `d` built from the previous gradient. Its second
iteration has `d = 0` and does not move, so the third iteration sees the same gradient; later
steps are the projection of the current gradient on a stale direction, with a unit step length.
`beta` enters as `|beta|`, and the plain `cg` name divides the Polak-Ribiere numerator by
`(g_k . g_k)^2`. None of this is the conjugate-gradient method: there is no minimization along
the direction, on which conjugacy depends.

mopt-rs implements nonlinear conjugate gradients with a strong Wolfe line search
(`c1 = 1e-4`, `c2 = 0.1`; Nocedal and Wright, Algorithms 3.5 and 3.6). The optimizer is an
ask/tell state machine, so the line search is spread over the iterations: each iteration
evaluates one trial point of the current line. `cg` and `cg_pr` are Polak-Ribiere+, `cg_hs`
Hestenes-Stiefel+ (`beta >= 0`, Gilbert and Nocedal), `cg_fr` Fletcher-Reeves, `cg_dy` Dai-Yuan;
the direction is reset to `-g` when it is not a descent direction and every `n` directions. On a
five-variable quadratic the directions are conjugate (`|d_i . A d_j|` below `1e-6` of the
diagonal) and the minimum is reached within 16 evaluations for every variant; Ar7 converges to
the tight criteria in 37 to 45 evaluations. Tests:
`opt::step::conjugate_gradient::tests::directions_are_conjugate_and_the_quadratic_is_solved_in_n_lines`,
`conjugate_gradients_minimize` (`tests/optimizers_ported.rs`).

### Eve: relative change of negative energies

Eve divides Adam's step by a smoothed relative change of the objective,
`d = |E_t - E_{t-1}| / (min(E_t, E_{t-1}) - E*)`, clipped to `[1/c, c]`; the paper assumes a
positive objective. MultiOptPy divides by `min(E_t, E_{t-1})` itself: for negative (electronic)
energies `d` is negative and always clipped to `1/c`, whatever the change. mopt-rs divides by the
magnitude `|min(E_t, E_{t-1}) - E*|` (`E* = 0` unless `EveParams::reference_energy` is set) and
defines `d = c` for a vanishing denominator. The two agree whenever the relative change is
below `1/c = 0.1` (as in the golden trajectory). Test:
`opt::step::eve::tests::relative_change_is_clipped_and_defined_for_any_energies`.

### Prodigy accumulates its denominator

Prodigy estimates the distance to the solution as `d = r / ||s||_1` from two running sums; the
paper shows that this estimate is a lower bound of the true distance. MultiOptPy computes the
new `s` but never stores it, so `||s||_1` holds only the newest term and the quotient is no
longer that estimate. Replaying the reference class on `f = (x - 5)^2 / 2` from `x = 0` (true
distance 5) gave `d = 1.47` after 10 iterations and `d = 5.09` from iteration 30 on, above the
true distance; with the running sum the estimate ends at `2.46` and the minimum is reached as
well. mopt-rs accumulates `s` as Algorithm 4 of the paper defines, uses `d_k` (not
`d_{k+1}`) in the step, and leaves `d` unchanged when `||s||_1 = 0`. MultiOptPy's `gamma = 0.1`
and `d_0 = 0.03` are kept. (The reference class also raises `TypeError` on NumPy 2.5, where
`float()` of a `1 x 1` array is an error.) Test:
`opt::step::prodigy::tests::distance_estimate_grows_toward_the_true_distance_and_never_shrinks`.

### FIRE 2.0 goes half a step back, not forward

When the power `F . v` turns negative, FIRE 2.0 moves the positions back by half a step
(`x <- x - dt v / 2`). MultiOptPy stores the velocity with the sign of the gradient (its
displacement is `-dt v`), but applies the correction as `x += -dt v / 2` in that convention,
which is half a step further in the direction of the last move, uphill. mopt-rs goes back along
the last displacement. Test: `opt::step::fire2::tests::first_steps_and_half_step_back`.

### ABC-FIRE follows the published algorithm

MultiOptPy's `ABC_FIRE` multiplies the old velocity by `1 / (1 - (1 - alpha)^i + 1e-10)` with a
counter `i` that starts at zero and is reset to zero whenever `v . g < 0` after the update. At
`i = 0` the factor is `1e10`; after such a reset the next iteration multiplies the kept velocity
by `1e10` whenever it is still downhill. The correction is also applied to the old velocity only
instead of the mixed one. mopt-rs implements the method of the paper (as in ASE's `FIRE2` with
`use_abc`): FIRE 2.0 with the mixed velocity divided by `1 - (1 - alpha)^(i + 1)`, `i` the number
of downhill steps since the last restart, and the displacement limited per component. The
factor is evaluated as `-1 / expm1((i + 1) log1p(-alpha))`, which stays accurate for small
`alpha`. MultiOptPy's parameters (`alpha_start = 0.1`, `dt_max = 0.8`, `n_acc = 5`) are kept.
Tests: `opt::step::abc_fire::tests`.

### Lookahead keeps a slow point

In MultiOptPy's `LookAhead` the "slow" geometry is overwritten with the current geometry in
every iteration, and every tenth iteration moves to the average of that (the previous
iteration's geometry) and the lowest-energy geometry since the last synchronization. This is
not the method of Zhang et al. mopt-rs keeps the slow point `phi` of the last synchronization
and, every `k = 10` iterations, replaces the step to the fast point `theta` by the step to
`phi + alpha (theta - phi)` with `alpha = 0.5`; no evaluation is added. `adam` and `ranger`
include Lookahead as in MultiOptPy's table of names. Test:
`opt::enhance::lookahead::tests::every_kth_step_lands_between_the_slow_and_the_fast_point`.

## Hessian-based optimizers

### `rsprfo_*`: restricted step, rigid-body modes, and the gradient cap

mopt-rs ports `EnhancedRSPRFO` with its eigenvalue shifting (maximized modes at most `-0.001`,
other modes `0.001` when below `1e-6`; a uniform shift for minimizations), its mode following
(overlap thresholds 0.5 and 0.3), its trust-radius rule (below a ratio of 0.25 a quarter of the
step length, above 0.75 at the boundary twice the radius), and the radius tied to the gradient
norm below `0.001`. Differences:

- The step restricted to the radius is found by bisection on the scale of the augmented
  Hessians (the step length is monotonic in it; `opt::step::prfo::partitioned_rfo_step`, shared
  with CCQN). MultiOptPy iterates a Newton correction with an approximate derivative for at most
  50 cycles and scales the best step found onto the radius.
- Rigid-body modes are projected out of the gradient and the Hessian and skipped. MultiOptPy
  projects the gradient only, so the six zero modes of the Hessian are shifted to `0.001` and
  take part in the mode selection. (From 2.1.1 the kept Hessian has no curvature along them
  either; see "Updated Hessians keep no rigid-body curvature" above.)
- The last step is rated with the energy change predicted when it was proposed. MultiOptPy
  recomputes the prediction with the Hessian after its update.
- MultiOptPy scales the step down (to as little as 10%) whenever it is longer than 50 times the
  gradient norm. For soft systems that is every step near the minimum: with the Ar7 curvatures
  of about `6e-4` Hartree/Bohr^2 the Newton step is some 1700 times the gradient norm. The cap is
  an absolute threshold on a quantity with units and is dropped; the trust radius limits the
  step.

Tests: `opt::step::rsprfo::tests`, and `rsprfo_*` in
`hessian_based_and_trust_region_methods_minimize` and
`saddle_points_with_the_ported_hessian_methods` (`tests/optimizers_ported.rs`; Ar7 converges to
the tight criteria in 86 iterations, both Muller-Brown saddles are found).

### Mode following never follows rigid-body modes

`MF_RSIRFO` passes the whole spectrum of the projected Hessian to the mode tracker. When the
Hessian has no negative eigenvalue, the lowest modes are the six rigid-body modes (eigenvalues
near zero of either sign), so the "target" is one of them; the reflection then skips it
(`|lambda| <= 1e-10`), and the step minimizes along every mode. mopt-rs gives the tracker the
modes with `|lambda| > 1e-10` only. The method string options (`:index:ema0.5:grad0.3`), the
mass-weighted metric (`mw`), static references (`s`), and the Hungarian assignment are as in
MultiOptPy; an unknown option is an error instead of being ignored. Tests:
`opt::step::mf_rsirfo::tests::the_followed_mode_stays_the_same_when_the_spectrum_reorders`,
`opt::step::mode_following::tests`, `math::assignment::tests`.

### `crsirfo_*` is RS-I-RFO under the optimizer's constraints

MultiOptPy's `CRSIRFO` takes the RS-I-RFO step in the null space of the constraint gradients and
restores the constraints (SHAKE). In mopt-rs constraints are held by the optimizer for every
method (see "Projection constraints for every optimizer and coordinate system"), so the
`crsirfo_*` names select RS-I-RFO; without constraints the two give the same trajectory
(`crsirfo_is_rsirfo_under_the_optimizers_constraints` in `tests/optimizers_ported.rs`).

### Newton steps with any Hessian

`Optimizer/newton.py` is not imported by `optimizer.py` (the import is commented out), so no
method name reaches it. It solves `H dx = g` directly, which is undefined for the singular
Hessian of a molecule and goes to whatever stationary point the signs of the eigenvalues select;
its momentum mode updates the Hessian from a mixture of the averaged displacement and the
absolute geometry. mopt-rs registers `newton_<updater>` and `mnewton_<updater>`: the step is
`-0.5 sum_i sigma_i (v_i . g) / |lambda_i| v_i` in the eigenbasis (`sigma_i = -1` for the modes
to maximize, rigid-body modes left out), and the momentum variant updates its own Hessian from
the secant pairs of the exponentially averaged positions and gradients (`beta = 0.5`), which obey
the secant equation of a quadratic. The momentum variant skips pairs with `s . y <= 0`. Tests:
`opt::step::newton::tests`.

### Trust-region L-BFGS

Ported with MultiOptPy's parameters (memory 30, radius 0.375 in `[0.01, 0.5]`, quarter below a
ratio of 0.25, double above 0.75 at the boundary). Differences:

- The newest secant pair enters the step of the same iteration. MultiOptPy computes the step
  first and stores the pair afterwards, so every step uses a model one iteration old.
- The reduction ratio compares the actual decrease with the prediction of the model and
  gradient that produced the step. MultiOptPy evaluates the prediction with the new gradient.
- The trust-region subproblem is solved from the eigenvalues of the compact representation
  with a bracketed root of the secular equation (`||p|| = Delta` to `1e-9`); MultiOptPy's
  Newton iteration can stop at its iteration limit and the step is then scaled.
- Powell damping uses `B s` of the current matrix (MultiOptPy uses `s` in its place), and is
  applied, as in MultiOptPy, only to pairs that fail the curvature condition.
- As in MultiOptPy, a step that raised the energy is not undone; the radius shrinks.

Tests: `opt::step::tr_lbfgs::tests` (secant equations, the secular condition, damping); Ar7
converges to the tight criteria in 24 iterations.

### TRIM

The level shift of the trust-region image minimization is bracketed below the lowest
eigenvalue of the image function, where the step length is monotonic, and found by Brent's
method. MultiOptPy starts SciPy's Newton iteration at zero without bounds, which can converge
between two eigenvalues (a step that does not minimize the model), and falls back to scaling.
Rigid-body modes are skipped; MultiOptPy can reverse one when the Hessian has no negative
eigenvalue. The radius is the optimizer's initial trust radius. Tests:
`opt::enhance::trim::tests`.

### GPMin follows the published method

MultiOptPy's `GPmin` cannot converge: it enforces a minimum step of 0.1 Bohr, a minimum trust
radius (2.0) above its initial radius (1.0), replaces the model step by a move of 80% of the
trust radius along the force whenever the model does not predict a decrease, and fits the
energies only (in inverse interatomic distances, which do not exist for plain variables).
mopt-rs implements GPMin as published (Garijo del Rio et al. 2019; ASE's `GPMin` without
hyperparameter updates): a Gaussian process with a squared-exponential kernel trained on
energies and gradients (`math::gp`), the prior mean at the highest energy seen, and the model
minimized from the lowest-energy point; an uphill step adds a point and the model is minimized
again from the same best point. Length scale 0.4 Angstrom and relative noise 0.002 are ASE's
defaults (`scale = 0.4`, `noise = 0.005`, `weight = 1`); the training set is limited to the
latest 25 points and the best one to bound the cost. Muller-Brown, the quadratic form, and Ar7
converge in 12, 16, and 42 iterations (`gpmin_minimizes`); because of the noise term the method
is not meant for the tight criteria. Tests: `opt::step::gpmin::tests`, `math::gp::tests`.

### Gaussian processes without the covariance matrix

`gpmin` and the `gpr` step refinement fit a Gaussian process to energies and gradients. Written
out, the covariance matrix of `N` points in `D` dimensions has `(N (D + 1))^2` elements: 450 MB
for 25 points of a 100-atom molecule, 4 GB for 300 atoms, with further copies during the
factorization. mopt-rs does not form it (`math::gp`). All differences between the points lie
in the space the points span (dimension `R <= N - 1`), gradient components orthogonal to that
space are uncorrelated with everything but themselves, and their covariance between the
points is the same `N x N` matrix for every orthogonal direction. The fit therefore factorizes
one block of size `N (R + 1)` (at most 650) and one of size `N`, and stores `O(N D)` numbers:
some hundred kilobytes for the 100-atom case instead of more than a gigabyte, and
`O(N^2 D + (N R)^3)` operations instead of `O((N D)^3)`. The result is the same up to rounding
(`math::gp::tests::the_block_form_is_the_full_covariance_matrix` compares weights and
variances with the matrix written out; the iterations of `gpmin` on the test surfaces did not
change).

### GPMin for saddle points of any order

New in mopt-rs (`-opt gpmin -order k`; MultiOptPy's `GPmin` and the published method
minimize). Putting the prior mean above all data, as the minimization does, would create a
minimum where a saddle point is sought, and a Gaussian process of a few points knows
curvature only along the directions sampled. So:

- The prior mean is the quadratic model at the current point with the optimizer's Hessian
  (`-fc`, `-modelhess`, `-modelhess o1numhess`), and the Gaussian process (the same kernel and
  length scale) is fitted to what the evaluated points show beyond it.
- On this surface, whose gradient and Hessian are available in closed form, the next point is
  found by restricted-step image-function iterations as in RS-I-RFO, reflecting the lowest `k`
  modes of the surface's Hessian at every iteration. The walk starts at the current point and
  ends at a stationary point of the surface or one length scale (0.4 Angstrom) away. It costs
  no evaluations.
- The gradient of the Gaussian process at the current point is subtracted, so that the surface
  has the gradient of the function there in spite of the noise term of the fit. Without this
  the search stalls near the saddle point (19 to 33 iterations instead of 5 to 9 below).
- The training set keeps the latest 25 points and the one with the smallest gradient.

The first step is the RS-I-RFO step of the given Hessian; afterwards every gradient seen
corrects the surface, which takes the place of Hessian updates. Fitting a surface and searching
it with a partitioned RFO method is the approach of Denzel and Kastner (*J. Chem. Theory
Comput.* **2018**, 14, 5777); the quadratic prior, the step control, and the fixed
hyperparameters here are not theirs. Like RS-I-RFO it needs a Hessian with the right number of
negative eigenvalues to start from.

Measured in the optimizer (iterations to the default criteria; `tests/optimizers_ported.rs`,
`tests/numerical_safeguards.rs`):

| Case | `gpmin` | `rsirfo_bofill` | `rsirfo_block_bofill` |
|---|---|---|---|
| saddle of order 2, three variables, three starts, Hessian at the start only | 8, 14, 13 | 9, 16, 16 | 10, 28, 16 |
| the same with a Hessian at every iteration | 8, 14, 13 | 9, 16, 14 | 9, 16, 14 |
| Diels-Alder transition state, GFN0-xTB, Hessian every 20 iterations | 23 | 22 | 22 |
| Muller-Brown saddles, Hessian every 20 iterations | 3 and 3 | | 3 and 3 |

GPMin is on a par with the Bofill updates in these cases, not ahead of them. The defaults of
the saddle search were measured with the step generator alone on the first case (iterations to
a gradient of 1e-5, Hessian at the start only / at every iteration): relative noise `1e-4`
(5, 8, 6 / 5, 6, 5; 0.002, the value of the minimization, takes one iteration more; `1e-6`
is the same), reach of one length scale and walk step of a quarter (half and twice change at
most one iteration).

Each walk iteration diagonalizes the surface's Hessian, so the method is for expensive
energies and moderate sizes.

Tests: `opt::step::gpmin::tests` (order 2), `tests/optimizers_ported.rs`
(`gpmin_finds_saddle_points`), `tests/numerical_safeguards.rs`
(`diels_alder_transition_state_with_gpmin`), `math::gp::tests` (the Hessian of the mean).

## Step enhancers

### The DIIS family

MultiOptPy's `GDIIS`, `EDIIS`, `ADIIS`, `C2DIIS`, `KrylovDIIS`, and `GEDIIS` share three
problems:

1. They return `w (x_DIIS - x_k) + (1 - w) m`, where `m` is the step of the base method in
   MultiOptPy's convention (`x_{k+1} = x_k - m`). The driver subtracts the result, so the DIIS
   part moves away from the extrapolated geometry. (In a replay of `GDIIS` on a three-variable
   quadratic the extrapolation was rejected by its own filters in every iteration, so the
   blend was not observed there; the statement is from the code.)
2. The extrapolated geometry `sum_i c_i x_i` is taken as the target without a relaxation step,
   so the method cannot leave the span of the earlier geometries.
3. The coefficients are clipped, renormalized, and blended with weights from success counters;
   the result is not the solution of any of the published equations. `C2DIIS` projects each
   gradient on its own direction (which is the gradient itself) and solves the DIIS equations;
   `EDIIS` and `ADIIS` minimize functions that are not the EDIIS energy or the ADIIS model.

mopt-rs implements the published methods on one shared core (`opt::enhance::diis_core`): the
error vectors are the gradients, and the new point is the combination of the relaxed points,
`sum_i c_i (x_i + r_i)` with `r_i` the step of the base method at point `i`.

- `gdiis`: Pulay coefficients (least squares, rank-revealing).
- `c2diis`: Sellers' eigenvector choice.
- `kdiis`: the Krylov-subspace accelerated inexact Newton method (KAIN; Harrison 2004), a
  Galerkin condition on the subspace of the history.
- `ediis`: the interpolated energy
  `sum_i c_i E_i - (1/4) sum_ij c_i c_j (g_i - g_j).(x_i - x_j)` minimized over the simplex.
- `adiis`: the quadratic model about the newest point,
  `sum_i c_i g_n.d_i + (1/2) sum_ij c_i c_j d_i.y_j`, minimized over the simplex.
- `gediis`: Li and Frisch's switch: the step of the base method above an RMS gradient of
  `1e-2`, EDIIS below it while the last RMS step is above `2.5e-3`, GDIIS near convergence
  (the thresholds and stages as in pysisyphus's implementation of the method; the text of the
  publication was not available). Saddle searches use GDIIS below the gradient threshold.

The simplex problems are solved exactly by enumerating the faces (`math::simplex_qp`; SciPy's
SLSQP from the uniform start finds a local minimum only). The extrapolating methods use the
acceptance rules of Farkas and Schlegel (step at most ten times the base step, the angle limits
by number of points, positive coefficients summing to at most 15); the interpolating ones are
limited to ten times the base step and are switched off in saddle searches. History sizes
(5; 6 for `kdiis`) and starting points (3 for `gdiis` and `kdiis`, 2 otherwise) are MultiOptPy's.

On a quadratic surface each method reproduces the stationary point exactly from `n + 1` points
(inside their hull for the interpolating ones): tests `extrapolation_is_exact_on_a_quadratic`
(`diis_core`), `null_eigenvector_gives_the_exact_extrapolation_on_a_quadratic` (`c2diis`),
`subspace_newton_step_is_exact_on_a_quadratic_and_agrees_with_gdiis` (`kdiis`),
`interpolated_energy_minimum_is_the_minimum_of_a_quadratic_inside_the_hull` (`ediis`),
`model_minimum_is_the_minimum_of_a_quadratic_inside_the_hull` (`adiis`). With RS-I-RFO on Ar7
(tight criteria) GDIIS needs 58 iterations instead of 66
(`diis_family_and_step_modifiers_minimize`).

### Coordinate locking and component-wise scaling

Both are ported rule by rule (they scale or zero step components, so no sign convention is
involved); FIRE with either agrees with MultiOptPy for 30 iterations on Ar7 to `1e-9` Bohr
(`coordinate_locking_and_component_wise_scaling_follow_multioptpy`). The modes that MultiOptPy's
command line cannot select (schedules, a manual mask, groups of correlated coordinates, the
curvature strategy) are available through the parameter structs. Where MultiOptPy would draw
coordinates at random (no gradient history) mopt-rs has nothing to do, because the history
always holds the current gradient. Component-wise scaling multiplies components by up to 5: with
Newton-type steps this overshoots (in mopt-rs `rsirfo_bfgs_component_wise_scaling` did not
converge on Ar7 within 3000 iterations), so it is meant for first-order methods.

### Geodesic step

MultiOptPy's `GeodesicStepper` integrates `q'' = -Gamma q' q'` in the space of the bond lengths
with `Gamma_ijk = G^-1_ik sum_a (dB_i B_j)_a`, which is not the Christoffel symbol of any
metric (it does not contain derivatives of the metric with respect to the internal
coordinates), and maps the end point back with the pseudo-inverse of `B` at the start.
mopt-rs integrates the geodesic equation of the metric `G = B^T B` that the bond lengths induce
on Cartesian space (Zhu, Thompson, and Martinez 2019), `x'' = -B^+ w`, with the fourth-order
Runge-Kutta method in 20 steps. Along this curve every bond length changes linearly: a step
that rotates a bond follows the arc and keeps its length to `1e-6` Bohr where the straight step
stretches it by more than 0.05 Bohr
(`opt::enhance::geodesic_step::tests::bond_lengths_change_linearly_along_the_geodesic`). Bonds
are found as in MultiOptPy (1.5 times the sum of the covalent radii).

### Perturbation

`Optimizer/purtubation.py` is not called by any MultiOptPy command and draws from NumPy's
global generator at a constant temperature. mopt-rs offers it as the enhancer `perturbation`
(for example `-opt lbfgs_perturbation`) with a seeded generator and a temperature annealed by a
factor of 0.9 per iteration from 298.15 K. The noise is 0.008 Bohr per coordinate at the
start, five times the displacement threshold of the normal criteria, so with a constant
temperature no optimization could converge; with the factor 0.9 the amplitude is below the
threshold after about 55 iterations by this estimate, and L-BFGS and FIRE with the
perturbation converge in 48 to 138 iterations in `tests/optimizers_ported.rs`. Test:
`opt::enhance::perturbation::tests::noise_has_the_thermal_variance_is_reproducible_and_anneals`.

### Line search

MultiOptPy's `LineSearch` starts from the previous energy and gradient but takes the current
geometry as the origin of the line, multiplies the step length by 1.5 while the energy falls
and `|cos(g, p)| >= 0.05`, and otherwise steps back to the best point; neither a sufficient
decrease nor a curvature condition is tested. mopt-rs searches along the step of the base
method for a point satisfying the strong Wolfe conditions (`c1 = 1e-4`, `c2 = 0.9`), with the
full step as the first trial, bracketing, and cubic interpolation (`math::line_search`). Every
trial point is one iteration of the ask/tell optimizer; no evaluation happens inside the
enhancer, and no trait had to be extended. Steepest descent with a factor that diverges on its
own (0.5 on a curvature of 10) converges with it, and steps that satisfy the conditions are
not changed (`opt::enhance::linesearch::tests`). Saddle searches and uphill steps pass
unchanged.

### Gaussian-process step

MultiOptPy's `GPRStep` fits a Gaussian process to the energies only, scores 25 random candidate
steps (different in every run) with a lower-confidence bound, and blends the best one with the
base step under opposite sign conventions. mopt-rs trains the process on energies and
gradients (the model of GPMin), minimizes its mean inside a ball of twice the length of the
base step around the current point, starting from the end of the base step, and takes the
minimizer if the model predicts an energy below the current one and below that of the base
step. The factor of two is the smallest tried that gave a gain: with a ball of the length of
the base step, steepest descent on an ill-conditioned bowl ended at `8.54e-3` against `8.59e-3`
without the refinement; with twice that length it ends below half of it
(`refined_steps_stay_in_the_ball_and_speed_up_steepest_descent`). RS-I-RFO on Ar7 converges to
the tight criteria in 31 iterations with it, 66 without.

### GAN step

In MultiOptPy's `GANStep` the generator is never trained: its output is detached with
`.item()` before it enters the discriminator, so no gradient reaches it and the method scales
every coordinate by a fixed random function of the coordinate, the gradient, and the step. The
discriminator is given the energy change, which labels its samples by itself, and the absolute
coordinates, which depend on the frame. PyTorch is not available to mopt-rs. mopt-rs
implements the scheme with a small network of its own (`math::mlp`): the generator maps the
scale-free features `(g_i / rms g, s_i / rms s)` to a factor in `[0.2, 3]`, the discriminator
tells its scaled proposals from the components of steps that lowered the energy, and the
gradient reaches the generator through the scaled component. The untrained generator returns
exactly 1, and training is seeded. Mixing ratio, step clip, learning rates, and batch sizes are
MultiOptPy's. Tests: `opt::enhance::gan_step::tests`
(`the_generator_learns_the_scale_of_successful_steps`), `math::mlp::tests`.

### Reinforcement-learning step

MultiOptPy's `RLStepSizeOptimizer` counts a step as successful only if the energy changed by
more than 70% of its absolute value. For total energies that never happens, so after three
steps the policy is paused and its weight decays to the minimum of 0.01: the method multiplies
every step by about 0.5, its default scale. Its rewards divide by the absolute energy, and it
reads and writes `./rl_models/step_size_policy.pt` in the working directory. mopt-rs keeps the
structure (a stochastic policy for the scale, proximal policy optimization every ten steps with
MultiOptPy's `gamma`, `lambda`, clip range, epochs, and learning rate, a policy weight between
0.01 and 0.5 that grows on success) with these changes: success means that the energy fell;
the reward is the energy decrease relative to the first-order prediction plus half the relative
decrease of the gradient norm, which does not depend on the units or the zero of the energy;
the neutral scale is 1 instead of 0.5, so that the base method is not slowed down before
anything is learned; the policy is linear in six scale-free features instead of an attention
network; the generator is seeded; and the policy weights are read and set through the API
instead of a file. Tests: `opt::enhance::rl_step::tests`
(`the_policy_learns_to_shorten_steps_that_overshoot`).

## Trust radius

### Predicted energy change in the outer trust-radius update

`TrustRadius.update_trust_radii` predicts the energy decrease as `g . m + m^T H m / 2` with
`m = x_k - x_{k+1}`; the quadratic term has the wrong sign (the decrease is
`g . m - m^T H m / 2`). mopt-rs uses the correct quadratic model. Test:
`opt::trust::tests::poor_prediction_shrinks_radius`.

### The trust radius of RS-I-RFO grows on any ratio above 0.75 (kept from MultiOptPy)

The RS-I-RFO steps (`opt::step::rsirfo`, `adjust_conventional` and the curvature-weighted
rule beside it) rate the last step by the ratio of the actual to the predicted energy change
and enlarge the trust radius whenever the ratio exceeds 0.75, also when it is far above 1 (a
step that went much further down than the quadratic model said, which means that the model
is poor, not good). Steps that are cut to the radius every iteration do not shrink it either:
only a ratio below 0.25 does. This is MultiOptPy's rule (`Optimizer/rsirfo.py`), and it is kept:
a version that also shrank the radius for ratios far above 1 and for repeatedly
radius-limited steps was tried and reverted, because it changed the step sequences of the
comparisons with CCQN that the tests hold.

The consequence seen in practice: runs on flat or anharmonic surfaces sit at the largest
radius (`-tr`, in Bohr) for many iterations, with ratios that stay
above 1 and steps of the full length, and can oscillate around the minimum before the ratio
drops. The remedy is on the user's side: a smaller largest radius with `-tr`, which shortens
those steps (the radius rules of RS-P-RFO and of the trust-region L-BFGS, which
grow only when the step reached the boundary, do not have the behaviour).

## Coordinate systems

### Polar (hyperspherical) coordinates

MultiOptPy's `Coordinate/polar_coordinate.py` has the n-dimensional polar coordinates of a
point relative to another, and only iEIP uses them, inside its microiterations. In mopt-rs
they are a coordinate system like the internal ones (`-coord hyperspherical`, also
`polar`, about the centroid or `hyperspherical:FILE` about a given structure; and
`spherical`, every atom by its own `r, theta, phi` about the centroid), for every command
([coordinate_systems.md](coordinate_systems.md)). What differs in the functions
(`coord::hyperspherical`):

- `compute_analytical_jacobian(p)` fills its first column, `dx/dr = x / r`, with
  `cart2polar(p)`, the polar coordinates of the polar coordinates, where `polar2cart(p)` is
  meant: `dx/dr` is wrong unless the point happens to be its own image. Both Jacobians are
  analytic here (`dx/dq` and `dq/dx`), checked against differences and against each other
  (`dq/dx dx/dq = 1`).
- The angles come from `arccos(x_i / |x_i...|)`, which loses half the digits near 0 and pi;
  here from `atan2` of the partial norm and the component, with the partial norms
  accumulated from the end by `hypot`.
- At a vanishing partial norm MultiOptPy returns an angle of 0 and goes on; here the
  Jacobian is refused there, and the coordinate systems turn their frame (an orthogonal
  map, per-atom polar axes) so that a structure starts far from such points and are set up
  anew when one comes near.

`coord::hyperspherical::tests`, `coord::system::spherical::tests`,
`tests/coordinates_spherical.rs`.

### Delocalized internal coordinates

MultiOptPy's `DIC_RSIRFO` is not reachable from any command (it is not registered in the
optimizer table) and builds coordinates from interatomic distances only; bends and torsions are
unimplemented placeholders. mopt-rs implements delocalized internal coordinates following Baker,
Kessi, and Delley (J. Chem. Phys. 1996, 105, 192) from bonds, angles, linear-bend complements,
and dihedrals, with iterative back-transformation. They are selected with `-coord dic` and work
with every step generator.

Separate molecules are joined into one set of internal coordinates by the closest atom pairs
of the molecules along a minimum spanning tree, which makes the set connected, and of
molecules that are neighbours (within 1.3 times the distance of either to its nearest
molecule, the factor of the auxiliary interfragment bonds of Bakken and Helgaker, J. Chem.
Phys. 2002, 117, 9160), which keeps the contacts of a cluster as coordinates. The number of
links grows with the number of molecules. Linking every pair of molecules gives
`F (F - 1) / 2` links and, through the angles and dihedrals between the links at each atom,
tens of thousands of primitives for twenty molecules: the transformation then no longer fits
into memory.

Test: `coord::internal::tests::fragments_are_linked_along_a_spanning_tree`.

### Internal coordinates are set up anew where they break down

MultiOptPy keeps its internal coordinates for the whole run; an angle that becomes linear, or
a dihedral with a linear angle inside, then has no usable derivative (its formulas switch only
within `1e-6` of pi). In mopt-rs every coordinate system can ask to be set up anew at a
structure, and the optimizer does it before the next step: the Hessian is carried over through
Cartesian coordinates, the previous point and the histories of the methods and of the Hessian
update are dropped, and the log notes it. This happens for

- `ric`, `dic`, `tric`, `tric-p`: an angle, or an angle inside a dihedral, has reached 175
  degrees (the angle is replaced by two linear-bend coordinates, the dihedral is left out),
  or a linear bend has bent back below 160 degrees or its direction has come within 30
  degrees of the axis;
- `dic`, `tric`: the fixed active space has lost rank at the structure;
- `tric`, `tric-p`: a fragment has turned by 90 degrees from its reference orientation.

A primitive set given by the caller (`RedundantInternal::with_primitives`) is never replaced.
In the back-transformation of a step, an iterate at which the coordinates cannot be evaluated
ends the iteration with the best iterate so far instead of an error.

Tests: `tests/coordinate_fallbacks.rs`, `coord::internal::tests::sets_are_regenerated_at_linear_angles`,
`coord::system::tric::tests::new_coordinates_are_asked_for_beyond_a_quarter_turn`.

### Translation-rotation internal coordinates

MultiOptPy has no coordinates for complexes of several molecules other than Cartesian ones
(its redundant internal coordinates join the fragments by bonds). mopt-rs adds TRIC (Wang and
Song, J. Chem. Phys. 2016, 144, 214108) as `-coord tric` (delocalized) and `-coord tric-p`
(the primitive set): every bonded fragment has its own bonds, angles, and dihedrals plus
three translations of its centroid and three rotations (the exponential map of the rotation
from its geometry at the start, scaled by its radius of gyration), with exact derivatives.
Single atoms have translations only; linear fragments use a dummy point, as in Wang and
Song's implementation; in periodic cells fragments are unwrapped along their bonds, and a
fragment bonded to its own image (a slab) has translations only. The rotation map is singular
at 180 degrees from the reference orientation; when a fragment has turned by 90 degrees, the
optimizer sets the coordinates up anew at the current structure (the log says so), carries
the Hessian over through Cartesian coordinates, and starts the histories of its methods
afresh. A step that jumps past the range does not stop the run either.

Tests: `coord::system::tric::tests` (rigid rotations give `angle x axis`, Wilson matrix against
finite differences, linear fragments, periodic cells), `tests/coordinates_tric.rs` (two
separated waters reach the minimum found in Cartesian coordinates in 57 iterations instead of
106 with GFN-FF and RS-I-RFO; molecules that have to turn over in a field; command line).

### TRIC stacked with Cartesian coordinates

`-coord tric-cart` (new in mopt-rs) stacks the Cartesian coordinates of all atoms, each times
a weight `w`, under the TRIC primitives and delocalizes the stack, in the manner of hybrid
delocalized internal coordinates (Billeter, Turner, Thiel, Phys. Chem. Chem. Phys. 2000, 2,
2177). The metric of the transformation is `B^T B + w^2 I`: it has full rank at every
structure, its condition number is bounded by `(s_max^2 + w^2) / w^2`, and the coordinates
go over into TRIC for `w -> 0` and into Cartesian coordinates for large `w`. `tric-cart:W`
sets the weight; the default, 0.1, and the rule of `-coord auto` (`tric-cart`, or Cartesian
coordinates for atoms without bonds, periodic networks, dense bonding, and very large systems)
come from the benchmark in [coordinate_systems.md](coordinate_systems.md): over molecules and
complexes, with and without a model Hessian, `tric-cart:0.1` needs the fewest iterations in
total (2044; TRIC 2125, delocalized internal coordinates 2291, Cartesian coordinates 2959).

Test: `coord::system::tric::tests::stacked_cartesian_coordinates_give_full_rank_and_both_limits`.

## Periodic cells

### Stress

New in mopt-rs (`pes::stress`, `EvalOutput::stress`). The stress is defined by the strain
derivative of the energy at fixed fractional coordinates, `sigma = (1 / V) dE / d(eps)`,
with the sign of ASE. The analytic stress of EMT, EAM, MEAM (also with the smooth cutoff
function and with pairs screened in part), Tersoff, and Lennard-Jones with a cutoff agrees
with central differences of the energy in the strain (`1e-4`) to `1e-5` of its largest
component in skewed cells with the atoms off their sites, and `dE / dh = V h^-T sigma`
agrees with differences of the energy in the lattice vectors (`pes::stress::tests`).
Lennard-Jones over nearest images (the default) has no analytic stress: its energy is not
a smooth function of the cell. Slabs and wires have a stress in the same sense, per volume
of the stored cell (only `V sigma` means something there). Not done: stress from Python
engines (ASE), and a comparison with the stress of ASE's EMT.

### Variable-cell optimization

New in mopt-rs (`opt::cell`, `optmain -vc`). MultiOptPy keeps the cell fixed.

The choice of the variables follows Gubler, Krummenacher, Huber, and Goedecker, "Efficient
variable cell shape geometry optimization" (J. Comput. Phys. X 2023, 17, 100131): atoms in
quasi-Cartesian coordinates and lattice vectors scaled by `w sqrt(N) / |a_k0|`, which keeps
the curvatures of the lattice of the size of those of the atoms whatever the number of
atoms and the shape of the cell. Their quasi-Newton method with its own step control is not
taken over: the variables go to the optimization methods of mopt-rs. The gradient in the
variables agrees with differences of the enthalpy for crystals, slabs, and wires
(`opt::cell::tests`).

The weight `w`: iterations to convergence (`-tcc`) from cells that are 4 % too large and
sheared, the atoms of the larger cells 0.1 Angstrom off their sites; copper, aluminium
(EAM), and fcc iron (MEAM) with 4 atoms, copper with 16 atoms in a long cell (1 x 1 x 4)
and a flat one (2 x 2 x 1) and with 32 atoms, aluminium with 24 atoms in a long cell
(1 x 1 x 6); sums over the seven cases
(`tests/console_stream.rs:iterations_of_variable_cell_optimizations`):

| Method | w = 0.25 | w = 0.5 | w = 1 | w = 2 | w = 4 |
|---|---|---|---|---|---|
| `rsirfo_bfgs` | 292 | 225 | 241 | 254 | 325 |
| `rsprfo_bfgs` | 317 | 245 | 240 | 255 | 326 |
| `lbfgs` | 345 | 201 | 135 | 108 | 122 |
| `fire` | 3286 | 1162 | 830 | 794 | 866 |

The default is `w = 2`, the value of the paper: the best for `lbfgs` and `fire` and within
13 % of the best for `rsirfo_bfgs`. For the cells of four atoms alone `rsirfo_bfgs` needs
10 iterations with `w = 0.5` and 15 to 22 with `w = 2`; for the larger cells the weight
hardly matters to it. `lbfgs` needs the fewest iterations of the methods tried.

A saddle point of atoms and cell together: the hop of a vacancy in copper (31 atoms, the
hopping atom near the middle of its path, the cell 1 % too large) with `-order 1 -fc 8`
ends at a point with one negative eigenvalue of the Hessian in the variables and no stress
left, in 13 iterations with `rsirfo_bofill` and in 13 with `rsprfo_bofill`, at the same energy.

Compared with Takagi, Taketsugu, Kino, Tateyama, Terakura, and Maeda (Phys. Rev. B 2017,
95, 184110, sections II.B and II.C, and S2 of its supplemental material):

- The gradient with respect to the lattice vectors is their Eq. (1),
  `dE/dh = -V0 sigma^t h0^-1`, with the stress counted positive under compression there
  and negative here (ASE).
- Degrees of freedom: a crystal has the 9 components of its three lattice vectors and the
  `3N` of its atoms, less three translations and three rotations: `3N + 3`. A slab has the
  in-plane components of two lattice vectors, less three translations and the rotation
  about its normal: `3N`. A wire has the component of one lattice vector along its line,
  less three translations and the rotation about its axis: `3N - 3`. mopt-rs has the same
  variables and takes the same motions out of the Hessian (`VariableCell::rigid_modes`);
  the paper fixes the other vectors on Cartesian axes, mopt-rs leaves them where the input
  has them, and for a slab the rotation is taken out only if the third vector stands on
  the plane (otherwise the lattice cannot turn with the atoms and there is nothing to
  take out).
- The force on the cell: the paper pushes or pulls one pair of dummy atoms, at the origin
  and at the tip of a lattice vector or at the tips of two, with the AFIR function. One
  distance per `-vc_afir` is that; a list of distances, weighted by `r^-6` as atoms of
  equal radius are, is an addition of mopt-rs.

The search settings of that paper (collision energy, the temperature of the stochastic
choice, the stopping rule, the clustering of structures) belong to the reaction network
search and are not part of this optimization.

Not done: exact second derivatives in the lattice variables (they come from differences of
the gradient), variable cells in coordinates other than these, a pressure or surface
tension for slabs, and the stress of Python engines (ASE), for which the stress comes from
differences of the energy.

## Optimization loop

### The starting structure is centered

MultiOptPy aligns every new geometry onto the current one with an unweighted Kabsch fit that
leaves the result centered at the origin, but it does not center the starting structure. The
first secant pair therefore contains the translation of the centroid to the origin, which wastes
the first quasi-Newton update (for an argon cluster, L-BFGS spent its second step translating the
cluster by the full 0.5 Bohr trust radius). mopt-rs centers the starting structure the same way;
energies and forces are unchanged.

### A trajectory that grows instead of a directory per iteration

MultiOptPy writes the structure of every iteration into a directory of its own
(`samples_<stem>_<iter>/<stem>_0.xyz`, made for the input files of the programs it runs) and
assembles `<stem>_traj.xyz` from those directories at the end of the run; a run that is
killed leaves hundreds of directories and no trajectory. mopt-rs writes no such directory:
`optmain` appends the structure of every iteration to `<stem>_traj.xyz` as it goes (the
frames and their comments `Frame k` are the ones MultiOptPy writes at the end), so the
trajectory is complete after every iteration. The same holds for `ieipmain` (below) and for
`-mf bitss`; `mdmain` appended its frames from the start. Scripts that read
`samples_<stem>_<k>/` read frame `k` of the trajectory instead.

Tests: `workflow::trajectories::tests`, `crates/mopt-rs/tests/cli_optmain.rs`,
`crates/mopt-rs/tests/cli_ieipmain.rs`, `crates/mopt-rs/tests/cli_bitss.rs`,
`tests/test_ieipmain.py`.

## Console output

mopt-rs prints what MultiOptPy 1.21.2 prints during an optimization, taken from runs of
MultiOptPy itself (`tests/console_stream.rs` checks lines, order, and formats): the
heading with the input file, the line of the potential, the structure in Angstrom before
the first iteration, under every `# ITR.` heading, and at the end (`%2s` and three times
`%17.12f`, two empty lines after it), the block between the rules with trust radius and
step length, the eigenvalues of the Hessian in iterations in which it was evaluated (six
`%12.8f` per line), the table of the convergence test (`%15.12f`), the messages of the
trust radius (`Energy ratio (actual/predicted)`, `Adaptive factor`, `Current trust
radius`, and what was done), and the closing lines. Numbers that MultiOptPy prints without
a format look as Python prints them (`0.5`, `8.517104648778046e-07`). As in MultiOptPy,
the energy shift of the first iteration is the energy itself.

Differences:

- Trust radius and step length are labelled `unit. Bohr`; MultiOptPy labels them
  `unit. ang.` although the numbers are in Bohr.
- The messages of RS-I-RFO about its step are at the debug level: always in `mopt.log`,
  on the screen with `-debug`. The count of the iterations of its inner solver
  (`Safeguard Solver: Converged in N iterations`) and the optimized `alpha` are not
  reported.
- Not printed: the Python objects of the optimizer (`Optimizer instance 0: <...>`), the
  user-defined trust radii, and `processing geometry collection ...` (mopt-rs writes the
  trajectory as it goes).
- `-gi`: MultiOptPy writes every ordered triple and quadruple of the chosen atoms, so each
  angle twice and each dihedral angle twice among 24 orders; mopt-rs writes each
  coordinate once. Dihedral angles carry their sign.
- Results of a run over several inputs: MultiOptPy keeps them in `Optimize.all_results`;
  mopt-rs returns them from `mopt_rs.run_optmain` (and `cli::optmain::run` in Rust). The
  thermochemistry and the gradients of bias parameters are not part of the summary yet.
- Added: the papers to cite under the calculator line, and for PySCF its own output under
  each iteration (MultiOptPy lets PySCF print to the screen only).
- Added: the values of the coordinates in use when they are not Cartesian, `<S**2>` where
  the calculator tells it, the energies of the parts of the surface, and the lines on the
  calculator that was loaded.

### Counts of the evaluations (new)

At the end of each input and of each command the number of gradient calculations and of
exact Hessian calculations is written (`Evaluations (this input): ...`, `Evaluations (in
total): ...`); MultiOptPy does not count them. `pes::counting`,
`tests/test_python_api.py::test_run_counted_reports_the_evaluations_of_a_command`.

## Constraints

### Projection constraints for every optimizer and coordinate system

MultiOptPy applies `-pc` only when the method list contains `crsirfo` and ignores it silently
otherwise. mopt-rs applies `-pc`, `-fix`, and `-gfix` with every method and in every coordinate
system (`crsirfo` is RS-I-RFO). The constraint gradients are carried into the optimizer's
coordinates and projected out of the gradient, the step, and the Hessian; convergence is
tested on the gradient without the constraint forces.

### Constrained Hessian and restoration

| | MultiOptPy | mopt-rs |
|---|---|---|
| Hessian | `P H P` plus stiffness along the constraints and along constraint directions of the last five iterations | `P (H - sum_k lambda_k d2c_k) P` plus the same hard stiffness: the Hessian of the Lagrangian, so curved constraints (bonds, angles, dihedrals) get the right second-order step |
| Restoration after a step | constraints adjusted one after another (both-side moves) for up to 10000 sweeps | all constraints together by Newton projection `x <- x - J^T (J J^T)^+ r` (quadratic convergence), after rotating oriented groups back with the Kabsch rotation |
| `rot` | rotation generators of the group at the current structure | Eckart conditions with respect to the starting orientation (linear, exact) |
| `eigvec`, `atoms_pair` | rotate the whole structure onto the start (Kabsch) | linear constraint along the fixed direction |
| Start | adjusted after the first step | the starting structure is moved onto the constraints before the first evaluation |
| Periodic cells | raw Cartesian geometry | minimum-image distances, angles, and dihedrals |

### Command-line fixes

- `-pc fbond`: MultiOptPy counts the trailing fragment size as an atom of the second fragment;
  mopt-rs does not.
- `-fix`: all arguments are used (MultiOptPy reads only the first).
- `-gfix`: MultiOptPy parses it without using it; mopt-rs holds the distances between the given
  atoms, which zeroes the gradient along them.

Tests: `tests/constraints.rs` (restoration in isolated and 2D-periodic cells, constraint
gradients against finite differences, constrained minima equal to stiff-penalty minima with
RS-I-RFO, `crsirfo`, DIC, and FIRE, frozen atoms, the command line), `cli::constraints::tests`,
`tests/test_python_api.py::test_constraints_from_python`.

## Model Hessians

### Two versions of every model

`-modelhess NAME` selects the model as published; `-modelhess legacy_NAME` selects MultiOptPy's
implementation reproduced as it is, for comparison with earlier runs. xtb's models are also
available as `xtb_lindh`, `xtb_lindh_d2`, and `xtb_swart`. An unknown name is an error
(MultiOptPy silently falls back to `lindh2007d3`). The model replaces the Hessian every `-mfc`
iterations when no exact Hessians are requested, as in MultiOptPy; rigid-body modes are
projected out for isolated systems.

| Model | MultiOptPy | Published model (mopt-rs default) |
|---|---|---|
| `lindh` | only interatomic distances as coordinates; bend and torsion constants added onto the distances of their bonds; covalent radii instead of the `r_ref` table; stretch constants divided by the reduced mass; UFF LJ and electrostatic terms added | Lindh 1995: stretch, bend, and torsion coordinates over all pairs, triples, and quadruples with `k rho_ij ...`, `rho = exp(alpha (r_ref^2 - r^2))`, `k = 0.45 / 0.15 / 0.005` (traversal as in xtb) |
| `lindh2007` | covalent bond lengths instead of `r_ref`; Hessian blocks of the D2 term mis-indexed; bend and torsion pair factors multiplied where they should be added; eigenvalues replaced by `0.1 exp(-|g|^2) |w|` | Lindh 2007 tables with `k = 0.45 / 0.10 / 0.0025`, out-of-plane `0.16`, van der Waals `0.05`, and xtb's D2 term |
| `schlegel` | only distances as coordinates (as `lindh`); H-H stretch parameter +0.2573 instead of -0.2573 and 0.8335 instead of 0.8355; Na-Ar treated as period 6 | stretch, bend, torsion, and wag coordinates with the Wittbrodt-Schlegel parameters |
| `swart` | O1NumHess's model with its near-0 and near-180 degree treatments swapped (as in the reference code) | as in the O1NumHess paper (Appendix B) |
| `swartd2` | xtb-like model with constants `0.35 / 0.15 / 0.005` and van der Waals weight 2.0 | xtb's Swart model (xtb's defaults) |
| `gfnff` | MultiOptPy's own approximation inspired by GFN-FF: its own force-constant estimates (not the GFN-FF energy or parameters), charges counting every bond twice, hydrogen bonds only for D->H / H->A angles above 120 degrees, unsorted table keys never matched | the Hessian of the built-in GFN-FF (below); `legacy_gfnff` reproduces MultiOptPy for H to Rn (the elements of its tables) |
| `gfn0xtb` | MultiOptPy's own approximation inspired by GFN0-xTB (no tight-binding calculation): hybridization guessed from neighbour counts, radii in Bohr that detect few heavy-atom bonds, torsion vectors zero unless `sin(phi) > 1e-10` | the analytic Hessian of the built-in GFN0-xTB (below); `legacy_gfn0xtb` reproduces MultiOptPy for H to Xe (the elements of its radii and electronegativities) |
| `gfn1xtb` | none (MultiOptPy has no GFN1-xTB model; `legacy_gfn1xtb` is an error) | the analytic Hessian of the built-in GFN1-xTB (below) |

### Hessians of the built-in methods as models

`-modelhess gfn0xtb`, `-modelhess gfn1xtb` and `-modelhess gfnff` take the Hessian of the
built-in GFN0-xTB, GFN1-xTB or GFN-FF at the current structure as the model Hessian, whatever
the calculator of the run: a minimization or a saddle search on the surface of another method
(`-os gfnff -modelhess gfn1xtb`, or a DFT calculator through PySCF) starts from, and refreshes
every `-mfc` iterations, the curvature of a tight-binding method. The structure keeps the
charge and multiplicity of the run (`-elec`, `-spin`). GFN0-xTB and GFN1-xTB give analytic
Hessians; GFN1-xTB starts the SCC of each structure from the converged charges of the one
before. Their limits are those of the methods: H to Rn, isolated systems only for
GFN0-xTB and GFN1-xTB (a structure with another element, or a periodic cell, is refused with
the reason; GFN-FF also takes three-dimensional cells). The modifiers apply as to every model
(`gfn1xtb_ts` reflects the lowest mode for a saddle search).

Tests: `crates/mopt-rs/tests/model_hessian_gfn1.rs` (the model is the analytic GFN1-xTB
Hessian with the charge and multiplicity of the structure; cells and elements without
parameters are refused; minimizations on GFN-FF and GFN0-xTB surfaces and the saddle of HCN to
HNC on the GFN0-xTB surface with it).

### Dispersion in model Hessians

MultiOptPy's `*d3` and `*d4` models are not DFT-D3/D4: they use `C6 = (<r^4>/<r^2>)^2` and a
damping function mixing the zero- and Becke-Johnson forms, and add `+42 C6 / r^8` (the
negative of the pair curvature). The published versions use genuine PBE-D3(BJ) (CN-dependent
C6 from the reference table, two-body) and add the exact Hessian of the dispersion energy,
including its coupling through the coordination numbers; long-range pairs can therefore add
small negative curvature. xtb's D2 term (in `lindh2007`, `swartd2`, `xtb_*`) keeps xtb's
convention: the negative of the scaled (`s6 = 20`) D2 Hessian, which stiffens intermolecular
modes. `swartd3` / `swartd4` are xtb's Swart model with D3 / D4 instead of D2;
`fischerd3old` and `fischerd3` (two heuristics in MultiOptPy) are both Fischer plus D3(BJ).

Genuine PBE-D4 (two-body and three-body, EEQ-2019 charges) is used by the `*d4` models; its
Hessian comes from central differences of the analytic gradient, as in dftd4. D4 in periodic
cells (Ewald EEQ) is not available yet.

### Numerically safe dispersion formulas

Written literally, several dispersion formulas overflow or produce `0/0` for unusual
geometries. mopt-rs evaluates them in stable forms (`math::stable`) that are algebraically
identical where the literal forms work: the Gaussian reference weights of D3 and D4 are
normalized after shifting the largest exponent to zero (dftd4 instead switches to the reference
with the largest CN when its normalization underflows), the counting and damping functions
`1 / (1 + exp(x))` use the two-sided logistic form, the EEQ coordination-number cap uses
`softplus`, the D4 charge scaling `zeta` takes its limits explicitly when its inner exponent
under- or overflows or the charge reaches zero, and the derivative of the three-body
coefficient avoids dividing by `sqrt|C6 C6 C6|`.

Tests: `tests/dispersion_d3.rs` (simple-dftd3's PBE-D3(BJ) reference energy for MB16-43 #01
within 1e-12 Hartree, exact gradient and Hessian, periodic cells), `tests/dispersion_d4.rs`
(dftd4's PBE-D4 reference energy within 1e-11 Hartree, charge conservation, analytic gradient,
nearly coincident atoms, separated fragments, and coordination numbers far beyond the
references), `math::stable::tests`.

Tests: `tests/model_hessian.rs` (published diatomic force constants, xtb's closed-form D2
derivatives, symmetry and translation invariance in isolated and slab cells, `ts` and `clip`,
optimizations with `-modelhess`), `model_hessian::swart_o1::tests` (the reference H2O2 Hessian
of O1NumHess, continuity of the paper's angle treatment).

### O1NumHess follows the published method

`-modelhess o1numhess` is not a model: it is a Hessian from a few more gradients of the
calculator (B. Wang, S. Luo, Z. Wang, W. Liu, *J. Chem. Theory Comput.* **2025**, 21, 10893;
reference implementations `ilcpm/O1NumHess` and `ilcpm/O1NumHess_QC`). MultiOptPy's
`ModelHessian/o1numhess.py` calls itself experimental and departs from the publication; mopt-rs
implements the publication (`pes::o1numhess`).

| | MultiOptPy | mopt-rs |
|---|---|---|
| Neighbours | covalent radii (`2.5 (R_i + R_j) + 1` Bohr) with extra bonds kept | distance between van der Waals spheres (UFF) below 1 Bohr; separate parts joined along a spanning tree |
| Translations and rotations | rotations by finite differences | no gradients: translations leave the gradient unchanged, rotations turn it |
| Local mode | eigenvector of the largest absolute eigenvalue | eigenvector of the largest eigenvalue of the modified Swart model in the neighbourhood |
| Displacement | unit-norm direction times the step | largest component equal to the step (0.005 Bohr); the breathing mode on both sides |
| Fit | local fit and a low-rank loop with momentum | local fit (elements up to 5 Bohr beyond the cutoff damped) and the low-rank loop as published |
| Negative eigenvalues | none | a second round of gradients along their eigenvectors; eigenvalues below the measured curvatures are reflected (`O1NumHess_QC` version 2) |
| In an optimization | computed inside each calculator every `-mfc` iterations | the optimizer asks the calculator for a Hessian every `-mfc` iterations, and a wrapper (`O1NumHessCalculator`) answers with O1NumHess; any calculator works, in every coordinate system; `-fc` takes precedence |

Differences from the reference implementation, all in `pes::o1numhess`:

- The local fit is solved by conjugate gradients in the Frobenius inner product from zero (the
  reference stores the lower triangle without weights, which makes the same equations
  nonsymmetric, and uses GMRES from the right-hand side). Where the directions determine the
  elements the solution is the same; elements they leave open become as small as possible.
- The reference has code that scales the gradient derivatives before the low-rank loop, but it
  reads the norms from an array of zeros and never scales. mopt-rs does not scale either.
- Directions are orthogonalized twice, and a direction that loses eight digits in the
  orthogonalization ends the generation (the reference tests the norm against `1e-15` only).
- `central` (off by default) takes points on both sides of every direction.

The averaging of local modes can leave a neighbourhood incompletely spanned (seen for chains
of identical neighbourhoods). Measured on an anharmonic banded function, the error then stays
within twice that of full finite differences, because the symmetry of the Hessian supplies the
missing equations; no directions are added for it.

Measured (`tests/o1numhess.rs`, largest element error relative to the largest element):

| System | Coordinates | Gradients | Error |
|---|---|---|---|
| banded function, 60 variables | 60 | 5 | as full single-sided differences (0.9 against 1.0 x 10^-2 at a step of 0.005) |
| C14H30, GFN-FF | 132 | 56 (10 of them along negative modes) | 0.4 % |
| Ar48, Lennard-Jones | 144 | 38 | 0.5 % |
| ethane, GFN-FF | 24 | 19 | 0.6 % (the single-sided step) |

Tests: `tests/o1numhess.rs`, `pes::o1numhess::*::tests`, `tests/test_o1numhess.py`.
