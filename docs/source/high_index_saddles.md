# Saddle points of higher index

A saddle point of index `k` is a stationary point whose Hessian has `k` negative curvatures
(beyond the rigid-body motions): a minimum has index 0, a transition state index 1. Saddle
points of index 2 and higher lie above the transition states around them and organize them:
an index-2 saddle is where the ridges of two transition states meet. mopt-rs finds them with
the parts below, each usable on its own from Rust and Python; the reaction-network mapper uses
them for its experimental higher-order saddles ([mapper_design.md](mapper_design.md),
"Higher-order saddles"). The index of what is found is always checked by the Hessian of the
point: no search keeps the index of the Hessian along its way.

| Part | What it does | Rust | Python |
|---|---|---|---|
| high-index saddle dynamics | an index-`k` saddle from a point and `k` directions | `path::hisd::HighIndexSaddleSearch` | `HighIndexSaddleSearch`, `saddle_search` |
| lowest modes | the lowest curvatures at a point from gradients | `path::hisd::modes::LowestModes` | `LowestModes`, `lowest_modes` |
| upward and downward starts | the starts of the searches of the solution landscape | `path::hisd::landscape::{upward_starts, downward_starts}` | `hisd_upward_starts`, `hisd_downward_starts` |
| mode-maximized surface | the energy maximized along a fixed direction | `path::hisd::mode_max::LineMaximum` | (through `FixedRidgeBand`) |
| band on a ridge, moving frame | an index-2 saddle between two transition states | `path::band::ridge::RidgeBand` | `RidgeBand`, `ridge_band_search` |
| band on a ridge, fixed direction | the same for two transition states with similar modes | `path::band::ridge_fixed::FixedRidgeBand` | `FixedRidgeBand`, `ridge_band_search(fixed=True)` |
| `optmain -opt hisd`, `-opt ggad` | the reflected-gradient step with the optimizer's Hessian | `opt::step::hisd::ReflectedGradient` | `Optimizer(methods=["hisd"])` with `saddle_order` |

All searches are ask/tell state machines over points of `n` variables, like the dimer method
([reaction_paths.md](reaction_paths.md)): they ask for energies and gradients (or Hessians) and never
evaluate anything themselves. A search is resumed by starting a new one from its last point
and directions.

## High-index saddle dynamics

The point follows the gradient reflected across the `k` tracked directions `V`,

```text
dx/dt = -(I - 2 V V^T) g
```

which climbs along `V` and descends along the rest (J. Yin, L. Zhang, P. Zhang, SIAM J. Sci.
Comput. 2019, 41, A3576, the high-index optimization-based shrinking dimer method, HiOSD;
for `k = 1` the gentlest ascent dynamics of W. E and X. Zhou, Nonlinearity 2011, 24, 1831).
Near an index-`k` saddle with `V` spanning its unstable subspace the Jacobian of the reflected
field has the magnitudes `|lambda_i|` of the curvatures as eigenvalues: the reflected problem is
a minimization, and a linearly stable fixed point is an index-`k` saddle (the theorem of Yin,
Zhang, Zhang). Away from it nothing holds the index at `k`.

One iteration evaluates, in one batch, the gradient at the point and the products of the
Hessian with the directions, updates the directions there, tests the ends, and steps:

- **Products of the Hessian** (`curvature`): `forward`, `(g(x + l v) - g(x)) / l`, one gradient
  per direction; `central`, `(g(x + l v) - g(x - l v)) / 2 l`, two (the dimer of HiOSD); or
  `hessian`, the exact Hessian at the point. `l` is the largest displacement of an atom
  (`dimer_length`); it can shrink as `l <- max(l_min, l exp(-shrink))` per iteration (the
  shrinking dimer, `dl/dt = -l`).
- **The update of the directions** (`update`):
  - `euler`: the explicit Euler step of the direction equations of HiSD,
    `v_i <- v_i + gamma_i d_i` with `d_i = -(I - v_i v_i^T - 2 sum_(j<i) v_j v_j^T) H v_i`, then
    modified Gram-Schmidt (Yin, Zhang, Zhang); the directions become the individual
    eigenvectors, lowest first;
  - `oja`: the Oja flow of the subspace, `V <- V - gamma (I - V V^T) H V`, then a QR
    normalization (M. T. Chu, M. M. Lin, SIAM J. Numer. Anal. 2025, 63, 2343, their first and
    third systems). Only the subspace is tracked, so curvatures that are close inside it do
    not matter; only the gap between the `k`-th and the `(k+1)`-th does. Chu and Lin show that
    the projector form `dP/dt = 2 P H P - H P - P H` drifts away from a projector in
    floating point unless written as a double-bracket flow; tracking a basis and normalizing
    it, as here, keeps `P = V V^T` a projector;
  - `lobpcg`: one step of the locally optimal block preconditioned conjugate gradient method
    on `span[V, W, V_prev]` (`W` the residuals; section 7 of Yin, Zhang, Zhang; A. V. Knyazev,
    SIAM J. Sci. Comput. 2001, 23, 517), `2k` more products per iteration;
  - `ggad`: the generalized gentlest ascent dynamics of W. Quapp and J. M. Bofill (Theor.
    Chem. Acc. 2014, 133, 1510, eqs. 16-21): each direction follows the Rayleigh quotient of
    the Hessian deflated by the earlier directions,
    `d_s = -(I - v_s v_s^T) D_s H D_s v_s` with `D_s = I - sum_(r<s) v_r v_r^T`, then a Schmidt
    orthogonalization. For orthonormal directions this is the HiSD update with weight 1 instead
    of 2 on the earlier directions: it does not keep the directions orthogonal by itself
    (hence the orthogonalization), and earlier directions do not feel the later ones. With
    `ggad` the first direction starts along the gradient (their eq. 14). Quapp and Bofill
    integrate with an adaptive Runge-Kutta method of order 8 and exact Hessians; here the same
    steps as for HiSD are used;
  - `eigenvectors`: the eigenvectors of the lowest eigenvalues of an exact Hessian (the choice
    of the algorithm of iHiSD).
  The Euler steps `gamma` are Barzilai-Borwein steps from the last update (the first a
  fraction `rotation_rate` of the inverse of the largest curvature magnitude), each change of a
  direction at most `max_rotation` long.
- **The step of the point**: `x <- x - beta (I - 2 V V^T) g` with the Barzilai-Borwein length
  `beta = s^T y / y^T y` (`s`, `y`: the changes of the point and of the reflected gradient, in
  the metric of the search); the first step, and every step with `s^T y <= 0`, has the length
  `first`; no atom moves more than `max_step` in one step. These are the steps of the
  optimization-based shrinking dimer methods (L. Zhang, Q. Du, Z. Zheng, SIAM J. Sci. Comput.
  2016, 38, A528, and HiOSD).
- **iHiSD** (`flow`): `dx/dt = -(alpha (I - 2 V V^T) + (1 - alpha) s I) g`, a gradient flow
  (`s = 1` downward, `s = -1` upward) that turns into HiSD as `alpha` grows from `alpha0` along
  `d alpha/dt = 2 rate alpha (1 - alpha)` (H. Su, H. Wang, L. Zhang, J. Zhao, X. Zheng, SIAM
  2025, DOI 10.1137/25M173212X; arXiv:2502.03694). It follows a gradient path into a saddle
  before HiSD takes over, which is why it is offered for downward searches.
- **The guard** (`guard`): one direction more is tracked and not reflected; its curvature
  tells whether the `(k + 1)`-th curvature is positive.

The ends:

| End | When |
|---|---|
| `Converged` | the force criteria are met and the `k` Ritz values of the tracked directions are negative (the guard's positive) |
| `HandOff` | the RMS gradient is below `rms_gradient`, the subspace turned by less than `projector_change` (the sine of the largest principal angle, `\|P_n - P_(n-1)\|_2`), and the `k` Ritz values are negative: the point is handed to a tighter method (RS-I-RFO at order `k`, with the directions as the modes to follow) |
| `WrongIndex` | the force criteria are met, but not all `k` Ritz values are negative: a stationary point of another index |
| `LeftIndex` | after the `k` Ritz values were negative once, the highest stayed non-negative for more than `lost_after` iterations |
| `IterationLimit` | the iterations are used up |

The index is decided by the Ritz values of the tracked block (the eigenvalues of `V^T H V`),
not by the Rayleigh quotients of single directions, which can all be negative where only one
curvature is.

### Metric and units

Structures are searched in the coordinates of their atoms (`with_structure(template, rigid,
mass_weighted)`): with `mass_weighted` the dynamics run in `q = M^(1/2) x` (gradients
`M^(-1/2) g`, Hessians `M^(-1/2) H M^(-1/2)`), which changes the paths, not the stationary
points or their indices (the Hessians are congruent). With `rigid` the translations and
rotations at the point are left out of every vector (a structure that is nearly linear, with a
principal moment below `1e-3` of the largest, is treated as linear: the rotation about its axis
is a bending motion there). Fixed variables are set to zero in every vector.

| Quantity | Units |
|---|---|
| force criteria, `rms_gradient` of the hand-off | Hartree/Bohr: the Cartesian gradient without the motions left out, whatever the metric |
| `dimer_length`, `first`, `max_step` | Bohr: the largest Cartesian displacement of an atom |
| curvatures (Ritz values) | in the metric of the search: Hartree/Bohr^2, or Hartree/(amu Bohr^2) when mass-weighted |
| directions (`directions()`) | in the metric of the search, orthonormal; `point_directions()` gives them in the coordinates of the points, of unit length (not orthogonal when mass-weighted) |

## Upward and downward searches

The solution landscape of a surface is the network of its stationary points joined by the
searches between them (J. Yin, Y. Wang, J. Z. Y. Chen, P. Zhang, L. Zhang, Phys. Rev. Lett.
2020, 124, 090601; J. Yin, B. Yu, L. Zhang, Sci. China Math. 2021, 64, 1801):

- **upward** from an index-`m` point with its unstable directions `u_1..u_m` and a stable
  direction `w`: the start `x +- eps w` with the directions `[u_1, ..., u_m, w]`, a search of
  index `m + 1`. At the start the reflected gradient pushes away along `w`;
- **downward** from an index-`m` saddle: for each unstable direction `w` (or, for `m = 2`, a fan
  `cos(t) u_1 + sin(t) u_2`, `t = pi j / N`), the start `x +- eps w` with the other unstable
  directions, a search of index `m - 1` (a minimization for `m = 1`).

A search found downward is a lower neighbour in the sense of these papers; that a gradient line
joins the two points is not shown by it (HiSD trajectories are not gradient lines), which the
mapper keeps apart.

## The lowest modes at a point

`LowestModes` finds the `count` lowest curvatures and their directions at a fixed point by the
locally optimal block preconditioned conjugate gradient method on differences of gradients
(one gradient per direction and iteration; the products of the conjugate directions follow by
linearity), or from an exact Hessian. It gives the unstable directions of a saddle point and its
softest stable ones when no Hessian is at hand.

## Bands on the ridge between two transition states

Two transition states `T1`, `T2` lie on a ridge: the set of maxima along a direction across it.
On the ridge they are minima, and an index-2 saddle `S` between them is a first-order saddle:
one of its negative directions lies across the ridge (maximized at every image) and one along it
(climbed by the climbing image). `S` is not a constrained minimum and need not be; a band with a
climbing image is the tool for first-order saddles. Three forms:

### Hessian form (the chain relaxation of `nebmain`)

The chain relaxation of `nebmain` with the Hessians of its images
(`path::band::machine::ChainRelaxation` with quasi-Newton steps, [reaction_paths.md](reaction_paths.md),
`nebmain`) gets a ridge (`ChainSettings::ridge`, `path::band::rfo::ChainRidge`;
`path::band::ridge::hessian_ridge_settings` gives the settings below). Every inner image keeps
its own Hessian (exact in every iteration here; a model or an updated one works the same way)
and takes the RFO step of its model across the chain, as in every chain relaxation with
Hessians:

- the transverse mode `u` of an image is the lowest eigenvector of its model across the chain
  `P H P + Q` (`Q` the direction of the chain, `P = I - Q`; the rigid-body motions and `Q`
  shifted above every curvature), or of the two lowest the one that continues the mode of the
  last iteration;
- `u` is an eigenvector of the model, so the image function is exact there: the gradient
  `P g - 2 (P g . u) u` and the Hessian `M - 2 lambda_u u u^T`. The RFO step of its minimization
  goes to the maximum along `u` and to the minimum across the rest;
- the climbing image, whose model `P H P - Q H Q` with the gradient `g - 2 (g . t) t` already
  reverses the direction of the chain, gets the same reversal along `u`: its model has two
  directions reversed and its fixed point is the index-2 saddle;
- the forces of the step along the forces get the same reversal (`+ 2 (g . u) u`); the step of
  an inner image is half the step along the forces and half the RFO step (`ratio = 0.5`, the
  default of `nebmain`), the images are kept apart by energy-weighted springs, and the ends stay
  where they are.

With `ratio = 1` (the RFO step alone) each image is drawn to the nearest stationary point when
its tangent is off and the band does not settle (measured, below); half and half converged in
every case tried. The step for the whole chain (`global`) is not combined with a ridge.

### Moving frame (`RidgeBand`)

After J. B. Maronsson, H. Jonsson, T. Vegge (Phys. Chem. Chem. Phys. 2012, 14, 2884); the same
walk along a ridge one index lower is the ridge method of I. V. Ionova and E. A. Carter (J.
Chem. Phys. 1993, 98, 6377). The ends are fixed; at every inner image:

- the tangent `tau` towards the higher neighbour (the tangent of the nudged elastic band);
- the transverse mode `v`: the lowest eigenvector of `P Q H Q P` (`P = I - tau tau^T`), the
  minimum mode of the Hessian reduced to the directions across the band, as in the paper
  (`transverse = lowest`); or the one of the two lowest that continues the mode of the last
  iteration (`continued`). The two lowest are tracked by an Oja step of their subspace per
  iteration from two products of the Hessian, started from the imaginary modes of the ends
  interpolated along the band;
- the effective gradient `(I - tau tau^T - 2 v v^T) g - F_s` (eqs. 8-11 of the paper), with the
  spring force `F_s = k (x_(i+1) - 2 x_i + x_(i-1))` in full (`spring_force = full`, the paper;
  its part across the band cuts the corners of a curved ridge) or along the tangent only
  (`tangential`);
- the climbing image, the highest inner image once it is a maximum along the band above both
  ends (as the paper advises: not from the start), after the band has settled (`climb_rms`) or
  after `climb_after` iterations: `(I - 2 v v^T - 2 t t^T) g` with `t = unit (I - v v^T) tau`
  (eq. 13), the reflected gradient of an index-2 search with the directions `(v, t)`, without
  springs.

The band moves by the step rule of the bands of mopt-rs (`step_rule = fire`: one time step for
all images, bounded by the curvature observed) or by Barzilai-Borwein steps of each image
(`bb`). The defaults, `spring_force = full`, `step_rule = fire` and `transverse = lowest`, are the
combination that converged from a displaced start in the measurements below (with
`continued` the band once climbed to another stationary point). It has converged when the climbing image's effective gradient is below `tolerance`
(and, with `band_tolerance`, the others' below that). Its highest image and the directions
`(v, t)` there are the start of a search of index 2.

Two transition states that share a minimum have nearly orthogonal imaginary modes, and the ridge
between them turns by about 90 degrees at `S`. A straight band cuts through the basin of the
shared minimum, where there is no maximum across the band near the ends; `cornered_band` starts
over the corner where the line through `T1` along the mode of `T2` comes closest to the line
through `T2` along the mode of `T1` (on two independent double wells, exactly `S`).

### Fixed direction (`FixedRidgeBand`)

When the imaginary modes `e1`, `e2` overlap by at least `min_overlap`, the direction
`e = unit(e1 + e2)` is kept fixed and the band is the nudged elastic band of mopt-rs
(`path::band::machine::ChainRelaxation`, plain variables, the ends fixed) on the
**mode-maximized surface** `E~(y) = E(y + t*(y) e)`, `t*` the local maximum along `e` continued
from the last one, on the hyperplane through `T1` across `e`:

- its gradient is `(I - e e^T) g` at the maximum (the envelope theorem), so its stationary
  points are stationary points of `E`;
- its Hessian is the Schur complement `H_yy - H_ye H_ee^(-1) H_ey`, and with `e^T H e < 0`
  the inertia additivity of E. V. Haynsworth (Linear Algebra Appl. 1968, 1, 73) gives
  `index E = 1 + index E~`: both transition states are minima of `E~`, and the climbing image
  of the band, a first-order saddle of `E~`, is an index-2 saddle of `E`.

These hold where the maximum along `e` exists and continues; every image keeps its branch (a
jump of `t*` is reported), and an image without a maximum along `e` ends the band
(`RidgeModeLost`). `e` stays as it was taken at the transition states; it is not projected anew
at other points, which would change the surface. `LineMaximum`, one point of this surface, is
also what an exploration on the surface of a transition state needs.

### When the bands fail

- **`RidgeModeLost`**: the curvature across the band is not negative at an image for longer
  than `lost_after` iterations. This is a numerical state, not a diagnosis: the two transition
  states may share no part of a basin boundary, the ridge may fold or end, or the band may lie
  off the ridge. It is not a test for a valley-ridge inflection point.
- Sharing a minimum makes two transition states a candidate pair; it does not guarantee one
  ridge or an index-2 saddle between them. Several maxima along a converged band are several
  candidates, and minima along it are transition states of their own.
- **`Series`**: the band leaves `T1` along its own imaginary mode; the transition states lie in
  series, and the ridge picture does not apply.
- A second negative curvature across the band (a region of index 3) switches the transverse
  mode (`switches` in the report); the band can climb to an index-3 point, which the Hessian
  check shows.

## `optmain -opt hisd`, `-opt ggad`

For a standalone minimum search, `optmain input.xyz -modekill` first uses the
selected minimizer. If the Hessian at convergence has excess negative modes,
the optimizer displaces along them and follows order-zero HiSD with a
Barzilai–Borwein step in its active coordinates. It checks the Hessian again
at convergence. The downhill step itself needs only a gradient.

With `-order k`, `hisd` takes the step `-beta (I - 2 V V^T) g` with the Barzilai-Borwein length
(limited by the trust radius), `V` the eigenvectors of `k` of the lowest eigenvalues of the
optimizer's Hessian: of the `2k + 2` lowest, those with the largest projection onto the
directions of the last step (or onto the directions given to the optimizer as modes to follow),
so that a soft positive mode that dips below an unstable one does not take its place. `ggad`
updates its directions by one deflated Euler step per iteration with the same Hessian, the
first along the gradient at the start. The Hessian is the exact one with `-fc 1`, otherwise the
updated one (block Bofill unless another updater is named, `hisd_bofill`); the directions are
only as good as its curvatures. The search from gradients alone is `HighIndexSaddleSearch`.

```text
optmain guess.xyz -os gfn1xtb -opt hisd -order 2 -fc 1
```

## Measurements

All on surfaces whose stationary points are known: model surfaces in the unit tests
(`path/hisd/tests.rs`, `path/band/ridge_tests.rs`) and the seven-atom Lennard-Jones cluster on
the built-in `lj` calculator in reduced units (`tests/hisd_lj7_*.rs`; energies in epsilon,
lengths in sigma). Every point a search ends at is refined by Newton steps on the exact Hessian
at the order sought (uphill along the lowest `k` modes, downhill along the rest), and its index
is the number of curvatures below `-1e-6` without the rigid-body motions. Gradients count the
differences of gradients; Hessians count those of the searches and of the refinement.

### Model surfaces

- the index-2 saddle of the cubic of Chu and Lin (their eq. 5.3) from `(0.08, -0.05, 0.06)` with
  every update and every source of curvatures (seven combinations): all converge to the origin;
- the index-2 and index-3 saddles of four double wells in rotated coordinates (81 stationary
  points of known index), with the guard: all converge, the guard curvature positive;
- the test function `B_k` of Yin, Zhang, Zhang (as restated by Chu and Lin), `k = 2, 3`, from
  `x* + N(0, 0.1^2)`: converges to `x*`;
- upward from a transition state of the double wells to index 2 (both signs), and downward from
  an index-2 saddle along both unstable modes: the four transition states below it.

### The landscape of LJ7

From the global minimum, in rounds until nothing new appears, from every point found: downward
along plus and minus each unstable mode (and four directions of the plane of an index-2
saddle), upward along plus and minus each of the six softest stable modes, and four random
combinations each way; the default settings (`HisdSettings::hiosd(k)`, hand-off at an RMS
gradient of `1e-2` and a projector change of `0.1`, steps of at most `0.1`).

| index | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| found | 4 | 12 | 44 | 98 | 168 | 189 (+1) | 168 | 101 | 45 | 11 | 1 |
| Doye and Wales 2002 | 4 | 12 | 44 | 98 | 168 | 190 | 168 | 101 | 45 | 11 | 1 |

31 986 searches in 9 rounds: 17 772 ended at a stationary point, 59 of them (0.3 %) of another
index than sought, 16 922 at a point found before; 21 930 761 gradients (686 per search, the
differences of gradients included) and 395 293 Hessians (12.4 per search, all of the
refinement). The other searches ended without a point: the cluster fell apart (2 462), or the
search left the region of the index (6 901) or used up its 500 iterations (727). 19 s on 16
threads.

- The one point of undecided index (at `-10.22982`) has four negative curvatures and a fifth of
  magnitude `2.7e-7` (`1e-8` of the spectrum), negative or positive depending on where it is
  refined: along that flat direction the point is not located (copies of it with the same
  energy to `1e-7` differ by more than `1e-2` in their interatomic distances). Counted with
  index 5, the counts are those of Doye and Wales at every index.
- The point that Quapp and Bofill (2014) report as an index-4 saddle at `-13.708` is a
  stationary point of index 3 by the Hessian here (`-13.70794`); their index-2 and index-3
  saddles at `-14.723` and `-14.348` are found with those indices.
- Points are the same when the energies (`1e-6`), the sorted interatomic distances (`1e-3`) and
  the curvature spectra (`1e-3` of their size) agree. Without the spectra two points of index 3
  fall together; with distances to `1e-6` one point is counted several times.
- Which point a search from a stationary point reaches depends on the orientation and the
  numbering of the atoms in which that point was found (the same start, rotated, follows another
  path near a basin boundary). The random combinations make the count independent of that here;
  with two random combinations instead of four one point of index 6 is missing.
- The rigid-body motions of a nearly linear structure are five: the linear chain is the one
  point of index 10 (its ten bending modes are unstable), and near it the normalized rotation
  about the axis is a bending motion.

### The settings of high-index saddle dynamics on LJ7

Upward searches from all 12 transition states (the six softest stable modes, both signs: 144
searches) to index 2, and from the 33 index-2 saddles of the default to index 3 (198 searches):

| setting | index | of the index sought | of another index | failed | distinct | gradients per search | Hessians per search |
|---|---|---|---|---|---|---|---|
| HiOSD: euler, forward (default) | 2 | 110 (76 %) | 1 | 33 | 33 | 231 | 12.7 |
| HiOSD: euler, forward (default) | 3 | 142 (72 %) | 1 | 55 | 49 | 291 | 10.0 |
| euler, central | 2 | 97 (67 %) | 11 | 36 | 35 | 351 | 9.8 |
| euler, central | 3 | 137 (69 %) | 14 | 47 | 49 | 481 | 9.0 |
| euler, forward, guard | 2 | 73 (51 %) | 0 | 71 | 27 | 225 | 2.9 |
| euler, forward, guard | 3 | 118 (60 %) | 0 | 80 | 45 | 268 | 2.8 |
| oja, forward | 2 | 108 (75 %) | 2 | 34 | 33 | 230 | 12.1 |
| oja, forward | 3 | 25 (13 %) | 0 | 173 | 16 | 658 | 1.4 |
| lobpcg, forward | 2 | 117 (81 %) | 1 | 26 | 25 | 393 | 6.4 |
| lobpcg, forward | 3 | 177 (89 %) | 0 | 21 | 46 | 585 | 6.1 |
| GGAD, forward | 2 | 99 (69 %) | 3 | 42 | 36 | 264 | 11.6 |
| GGAD, forward | 3 | 150 (76 %) | 1 | 47 | 54 | 319 | 9.9 |
| eigenvectors, exact Hessian | 2 | 110 (76 %) | 14 | 20 | 16 | 38 | 44.6 |
| eigenvectors, exact Hessian | 3 | 173 (87 %) | 10 | 15 | 37 | 46 | 50.8 |
| iHiSD upward | 2 | 81 (56 %) | 2 | 61 | 30 | 297 | 6.5 |
| iHiSD upward | 3 | 124 (63 %) | 1 | 73 | 51 | 390 | 4.7 |
| hand-off at RMS 3e-3 | 2 | 108 (75 %) | 3 | 33 | 35 | 259 | 11.7 |
| hand-off at RMS 3e-3 | 3 | 142 (72 %) | 1 | 55 | 48 | 348 | 8.6 |
| dimer length 1e-2 | 2 | 106 (74 %) | 3 | 35 | 31 | 227 | 11.9 |
| dimer length 1e-2 | 3 | 151 (76 %) | 1 | 46 | 52 | 313 | 8.4 |

- HiOSD (Euler update, forward differences) is the default. GGAD found 10 % more distinct points
  of index 2 and 3 together (90 against 82) with 11 % more gradients; LOBPCG reached the index sought most often but
  found fewer distinct index-2 saddles (its searches converge to the same ones) at 1.7 to 2
  times the gradients; the eigenvectors of exact Hessians cost the fewest gradients (but 45 to
  51 Hessians per search) and found the fewest distinct points. The guard ends searches whose `(k + 1)`-th curvature is not yet positive
  and so finds less; the Oja update without the order of HiSD loses its way at index 3; iHiSD
  upward climbs the stiffest modes first and finds less.
- Wrong index: the default ends at another index in 1 of 111 and 1 of 143 converged searches;
  central differences in about 10 % of theirs.
- The defaults of `HandOff` (`1e-3` Hartree/Bohr, `0.1`) and of the step lengths are for
  molecules and are not measured on them yet; the LJ7 searches above use `1e-2` epsilon/sigma
  and steps of `0.1` sigma.

### The bands on the ridge between two transition states

Two independent double wells in rotated coordinates (four variables), the inner images of the
first chain displaced by `N(0, 0.1^2)`, the climbing image judged by its distance to the index-2
saddle:

| form | step rule, springs | side by side (2 pairs) | sharing a minimum (1 pair) |
|---|---|---|---|
| moving frame | Barzilai-Borwein per image, full springs | no convergence (forces 0.03 to 0.6), climbing image at the saddle | the same |
| moving frame | Barzilai-Borwein per image, tangential springs | images leave the ridge (`RidgeModeLost`) | the same |
| moving frame | FIRE of the band, full springs | converged, 88 to 377 iterations | converged, 89 iterations (from a corner) |
| moving frame | FIRE of the band, tangential springs | images leave the ridge | the same |
| Hessian form | `ratio = 0.5`, springs 0.1/0.01 to 3/1, inner steps 0.05 and 0.1 | 12/12 converged, 73 to 133 iterations | 6/6, 106 to 114 iterations |
| Hessian form | `ratio = 1` | 2 of 12 converged | 6/6 |
| Hessian form | `ratio = 0.3` | 12/12 | 5/6 (one diverged) |
| fixed direction | nudged elastic band of mopt-rs | converged | not applicable (modes orthogonal) |

The moving frame therefore uses the full spring of the paper and the step rule of the bands
(FIRE), and the Hessian form `ratio = 0.5`.

LJ7: the transition states below 14 index-2 saddles (downward along plus and minus both unstable
modes), every pair of them: side by side (from the same unstable mode) and sharing a minimum
(from different modes, a band over a corner). A method succeeds on a pair when its point, refined
at order 2, is the saddle the pair came from:

| method | pairs side by side: at the saddle of the pair | gradients + Hessians per pair | pairs sharing a minimum: at the saddle of the pair | gradients + Hessians per pair |
|---|---|---|---|---|
| Hessian form (chain relaxation of `nebmain`) | 10 of 14 (3 at another index-2 saddle) | 1381 + 1381 | 16 of 29 (5 at another) | 1584 + 1584 |
| moving frame from gradients | 14 of 14 | 678 + 0 | 18 of 29 (6 at another) | 691 + 0 |
| fixed direction | 9 of 9 that it accepts (5 refused: modes apart) | 491 + 0 | (not applicable) | |
| HiSD (index 2) from the middle of the first chain | 14 of 14 | 80 + 3 | 21 of 29 (5 at another) | 230 + 8 |

The Hessian form here uses springs of 1 and 0.1 epsilon/sigma^2, inner steps of 0.03 sigma and
`ratio = 0.3`, which found the most saddles of the settings tried (with the defaults of `nebmain`:
7 of 14 side by side, 8 of 29 sharing a minimum; with 500 iterations no more; with the mode
continued from the last iteration 1 of 14). Most of its bands end at the iteration limit: each
image is drawn towards the nearest stationary point whenever its tangent is off, which the
springs only partly hold.

- On LJ7 an index-2 search of high-index saddle dynamics started at the middle of the first
  chain of the band, with the interpolated mode and the direction of the chain as its two
  directions, reaches the saddle of the pair most often and costs the least; a band is the more
  expensive way to the same point. The bands remain the tools that follow the ridge itself: they
  show whether the two transition states lie on one ridge, and the maxima and minima along it.
- The moving frame from gradients ends with `RidgeModeLost` at an image near an end in most
  pairs on LJ7 (the transverse curvature there turns positive after a few iterations), yet its
  highest image is at the saddle: the climbing image settles before the band does.
- The fixed direction needs modes that overlap by 0.7 or more (5 of the 14 pairs side by side
  are refused); where it runs, it found the saddle of the pair in every case.

## Limits

- The index of what a search finds is decided by the Hessian of the point, not by the search:
  HiSD reflects `k` directions whatever their curvatures are, and its stable fixed points are
  index-`k` saddles only near one. A point whose curvature is near zero has no decided index;
  it is reported apart.
- A point found by a downward search is a lower neighbour in the sense of the solution
  landscape; that a gradient line joins it to the saddle above is not shown (HiSD trajectories
  are not gradient lines).
- Which points a set of searches finds depends on the orientation and the numbering of the
  atoms of the point it starts from; more start directions (random combinations of modes) make
  the result less dependent on that.
- The bands need two transition states that share a ridge; sharing a minimum makes them
  candidates, not certainly that. `RidgeModeLost` and the iteration limit say that a band did not
  settle, not why.
- The mode-maximized surface and the fixed-direction band hold only where the maximum along `e`
  exists and continues; `e` is not projected anew at other points.
- With ONIOM and microiterations the surface searched is `E_eff(x) = min_y E(x, y)`; its
  products of the Hessian must be those of that surface (the Schur complement
  `H_xx - H_xy H_yy^(-1) H_yx` with an exact Hessian, or differences of gradients of the
  re-relaxed environment), not those of `H_xx`. Not built here.
- `optmain -opt hisd` takes its directions from the optimizer's Hessian: an updated Hessian
  may place its negative curvatures wrongly, so `-fc 1` (or a few exact Hessians) is the safe
  use.
