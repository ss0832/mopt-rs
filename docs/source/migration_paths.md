# Differences from MultiOptPy: reaction paths

Chains of images (`nebmain`), the IRC, BITSS, and `ieipmain`.

## Chain relaxation (the parts of `nebmain`)

What MultiOptPy calls NEB is, read from its code, a chain of images without springs: the
component of the gradient along the chain is removed atom by atom, the images move along
what is left (and, with Hessians, by a quasi-Newton step), and the images are put back
along the chain every few iterations. mopt-rs has these parts in `path::band` in two forms
(`path::band::Compatibility`):

- `MultiOptPy`: what MultiOptPy does, step by step. `tests/golden_neb.rs` compares seven
  runs of MultiOptPy 1.21.2 (`tools/golden/run_neb_reference.py`, seven argon atoms with the
  Lennard-Jones calculator: default, fixed ends, climbing, steepest descent, equal spacing,
  the settings of the articles with and without redistribution) iteration by iteration;
  the chains agree within `1e-7` Bohr.
- `Consistent` (the default): the same procedure with the changes below. Each of them
  removes something that kept a chain from coming to rest; they were found on the
  Muller-Brown surface (17 images between its outer minima, energies in units of `1e-3`)
  and are described where they are implemented.

| | MultiOptPy | mopt-rs (default) |
|---|---|---|
| Force of an image at an extremum of the energy profile | shares `a` and `b` of the gradient components along the directions to the two neighbours are removed one after the other; where the chain has a kink the image keeps a force (`1.5e-5` however long it is relaxed) | the component along the one direction `a (x(i+1) - x(i)) + b (x(i) - x(i-1))` is removed (the weighted tangent of Henkelman and Jonsson) |
| Climbing image (`-cineb`) | `F = -2 R` with the removed component `R`: the image moves down along the chain and not across it; the forces of its neighbours are multiplied by 0.001 | `F = -g + 2 (g . t) t` with the unit tangent `t` of the whole image, as in ase_qsm; the neighbours keep their forces |
| Time step of the step rule along the forces ("FIRE"; the velocity is the force) | grows by a tenth per iteration until the steps overshoot, is halved, and grows again: the forces of an almost relaxed chain rise by orders of magnitude every few dozen iterations | bounded by the largest curvature met along the last steps, `dt (2 + dt) k <= 1` |
| Step of an image away from both neighbours | not limited (a climbing image went to `1e104`) | at most the larger of the distances to the neighbours |
| Quasi-Newton step of an inner image (`-modelhess`, `-fc`) | a minimization with the full gradient: the image slides down the valley and only the redistribution brings it back; `gamma t t^T` is added to its Hessian in every iteration on top of what was added before; the trust radius of the minimizer has no effect (its solver returns the unrestricted step) | a step across the chain: gradient `P g`, Hessian `P H P + Q` (`Q`: projection on the directions along the chain that the force rule removes, `P = 1 - Q`); a climbing image gets the model with its tangent reversed; nothing is added to the Hessians |
| Direction of the chain in the quasi-Newton model | not there (the minimization does not know of the chain) | the force of an image depends on its direction along the chain, which turns when the image moves: a step `d` across the chain changes the force by `(G / h) d` (`G`: slope of the energy along the chain, `h`: distance to the neighbour uphill). This is added to the model Hessian, `P H P + sum_k (G_k / h_k) P_k + Q`. Where the energy rises steeply along the chain it is as large as the curvature of the surface; without it neighbouring images swing against each other in a cycle of two iterations (forces of 0.03 to 0.07). The Hessians themselves stay Hessians of the surface, updated with the full gradients |
| Pairs of the block updates of these Hessians | one (the block updates drop every pair after using it) | the window of pairs of mopt-rs's block updates (`ChainRfoSettings::pairs`; one pair can be chosen). Both converge alike (table below) since the block updates screen their pairs (see "Block updates scale and screen their pairs") |
| Where redistribution puts the images (`-ad`, `-adpred`) | on the straight lines from image to image, which pushes them off a path that bends by about `k h^2 / 8` every time | on a cubic curve through the images (`redistribute::Curve::Smooth`); images that are where they should be stay exactly there |
| Units in `-adpred` | arc length in Angstrom, gradients per Bohr | both in Bohr |
| Equal spacing with climbing images | all images are spread evenly, which takes a climbing image off its saddle | the images are spread between the ends and the climbing images |
| Step rule along the forces when quasi-Newton steps are mixed in | started anew in every iteration | keeps its time step, with the bound above |
| Direction that is removed from the gradient of a structure | one per atom: every atom loses the part of its gradient along its own displacement to the next image. An atom that hardly moves along the path has a direction that is noise, and the direction of every atom turns with the smallest step of that atom | one for the whole structure, the tangent of the chain (as in ase_qsm; `-neb_opt projection=atom` and `ChainSettings::block` choose MultiOptPy's rule). HCN to HNC converged in 18 iterations instead of 42, CH3NC to CH3CN in 46 instead of not within 60 |
| Orientation of the structures | every structure is centred and turned onto the next one as that one lay before its own turn, so the chain keeps turning; the Hessian, the last gradient, and the last step of an image stay in the old orientation | every structure is centred and turned onto the one before it as that one lies after its turn (an aligned chain stays as it is), and what is kept for an image is turned with it. With MultiOptPy's way the chain for HCN to HNC below did not come to rest (the free end image, a plain minimization of HCN, kept forces of 0.1 to 0.4 for 60 iterations), and with redistribution its energies went to thousands of kcal/mol |
| Images between the ends and the predicted maxima (`-adpred`) | spread evenly | gathered around the maxima: twice as dense there, over a width of one average distance of the images (`redistribute::Gathering`; the settings of the articles) |

| Image at a maximum of the profile in the redistribution towards the predicted maxima | moved to where the polynomial through it and its neighbours has its maximum | the same, but a move that reverses the last one is at most half of it, and the limit grows by half with a move in the same direction (`path::band::peaks`). The prediction is as good as the energies and slopes it is made from; with images that are not on the path yet (model Hessians) it sent the image back and forth between two places for ever (below) |

Iterations until the largest force of the inner images is below `1e-6` on the Muller-Brown
surface (17 images, fixed ends, exact Hessians of the surface at iterations 0 and 50, half
a quasi-Newton step; `examples/chain_muller_brown.rs`):

| Redistribution | pairs of the block updates | one pair |
|---|---|---|
| towards the predicted maxima every second iteration, images spread evenly | 27 | 26 |
| the same, images gathered around the maxima (the settings of the articles) | 32 | |
| equal spacing every second iteration | 30 | 29 |
| none | 26 | 26 |

With the steps along the forces alone (equal spacing every tenth iteration, climbing
images) it takes 165 iterations. In MultiOptPy's form with the settings of the articles
the largest force stays between 0.03 and 0.1 (the first chain has 0.16). With the
redistribution towards the predicted maxima an image sits on each of the two saddles;
without redistribution the images are where their steps left them and none is near the
higher saddle. Tests: `path::band::machine::tests`
(`the_way_of_the_articles_converges_with_steps_across_the_chain` shows both forms,
`end_images_are_minimized_unless_they_are_fixed` every setting of `-fe` with and without
Hessians), `path::band::rfo::tests`, `path::band::redistribute::tests`,
`path::band::fire::tests`.

The Hessians of the images are kept in files between the iterations when the chain is
given a directory (`ChainRfoSettings::directory`): `tmp_hessian_<i>.npy`, the names and
the format of MultiOptPy (`io::npy`; they can be read with `numpy.load`), one Hessian in
memory at a time. Without a directory they stay in memory. Test:
`path::band::rfo::tests::hessians_kept_in_files_give_the_same_steps`.

A relaxation ends when the forces of the images that move are below a tolerance, if one
is set (MultiOptPy runs its number of iterations): the inner images, and the end images
that are not fixed (`-fe`), which are minimized.

The command `nebmain` runs these parts (`cli::nebmain`; `tests/cli_nebmain.rs`). Its NEB iteration
limit defaults to 30. Image spacing and redistribution follow the articles of MultiOptPy's
author (`-nd 0.5 -adpred 2`), with a model Hessian in place of `o1numhess`
(`-modelhess lindh2007d3`; see "The Hessians of the images without `-modelhess`" below), not
MultiOptPy's defaults (`-ns 10 -modelhess none -nd 0 -adpred 0`, which can be given);
figures are written by default, a deliberate difference (MultiOptPy 1.21.2 writes every
figure only with `-spng`, which is off by default); `-spng` alone keeps meaning on, and
`-spng off` or `"save_pict": false` in a settings file writes no figures and the CSV data
as before; the run ends
when the largest force of the images that move is below `3e-4` Hartree/Bohr (the largest
force at which an optimization of `optmain` is converged; `-neb_opt tolerance=0` runs all
iterations, as MultiOptPy does), and a run that used up `-ns` before says how large the
force still is; `-gfix` and `-uc` work besides `-fix` and `-pc`,
and every image holds its own starting values of the constraints; the run directory is
`<name>_NEB_<time>/` beside the input, as for the other commands (MultiOptPy:
`<input>_NEB_<calculator>_<time>/`).

Errors of MultiOptPy's `nebmain` that were found while the reference data were made (they
concern the input and are not reproduced): a file of several structures loses its last
structure unless a line that is not an atom follows it; the first structure enters the
chain twice; without `-modelhess` every image raises and swallows an exception
(`config.model_hessian.lower()`), which resets the step rule every 50 iterations; the block
updates fail on NumPy 2.5 (see "Block Bofill on current NumPy").

Molecules (`examples/chain_molecule.rs`, built-in GFN0-xTB, three rough structures per
reaction made into a chain of images 0.5 Angstrom apart, free ends, Hessians of the
calculator at the start, until the largest force is below `1e-4` or 60 iterations; the
first chains have barriers of 521 and 288 kcal/mol because atoms pass close to each
other). The barrier is the energy of the highest image above the first one:

| Chain | iterations | barrier (kcal/mol) |
|---|---|---|
| HCN to HNC, 8 images, one direction per atom (MultiOptPy's rule) | 42 | 68.285 |
| the same, images spread evenly (`gathering: None`) | 41 | 68.285 |
| the same, equal spacing instead of the predicted maxima | 48 | 66.388 |
| the same, no redistribution | 37 | 67.885 |
| the same, one direction for the whole structure (the default since) | 18 | 68.285 |
| the same, MultiOptPy's form | not in 60 | 70.1 (moving) |
| HCN to HNC, 12 images, one direction for the whole structure | 18 | 68.285 |
| CH3NC to CH3CN, 6 images, one direction per atom, without gathering | not in 60 (force `5e-3`, falling slowly) | 70.73 |
| the same, one direction for the whole structure, without gathering | 46 | 68.906 |
| the same, MultiOptPy's form | not in 60 | 76.2 (moving) |
| CH3NC to CH3CN, 10 images, one direction for the whole structure | 58 | 66.983 |

The same chains with `nebmain` (its defaults: Hessians from `-modelhess`, images gathered
around the predicted maxima every second iteration), until the largest force is below
`1e-4`, with the Hessians of the images from different sources:

| Hessians of the images | HCN to HNC, 8 images: iterations, barrier | CH3NC to CH3CN, 6 images: iterations, barrier |
|---|---|---|
| `-fc 50` (the calculator's) | 19, 68.285 | 41, 68.906 |
| `-modelhess gfn0xtb` (the calculator's own Hessian as the model) | 18, 68.285 | 41, 68.906 |
| `-modelhess o1numhess` (the default when this was measured) | 19, 68.278 | 50, 66.983 |
| `-modelhess lindh2007d3` | 21, 68.264 | 49, 66.983 |
| `-modelhess fischerd3` | 21, 68.284 | 55, 66.983 |
| `-modelhess schlegeld3` | 19, 68.285 | |
| `-modelhess swart` | 21, 68.285 | |
| `-modelhess gfnff` | 19, 68.285 | 61, 66.983 |
| `-rrs 1` (quasi-Newton steps alone, `o1numhess`) | 17, 68.285 | 72, 66.973 |
| `-modelhess none` (steps along the forces alone) | 60, 68.285 | 138, 66.983 |

CH3NC has two saddles (68.906 kcal/mol, 329i cm^-1, and 66.983, 701i; both were optimized
and followed by `ircmain`); the chain with the calculator's Hessians ends on the path over
the first, the chains with approximate Hessians on the path over the second.

Before the moves of the image at a maximum were limited, none of the approximate Hessians
brought the CH3NC chain to rest (largest force between 4e-3 and 1.5e-2 after 150
iterations, the barrier going back and forth between 66.4 and 67.2 with a period of four
iterations, the positions of all inner images changing between two patterns at every
redistribution), and for HCN `lindh2007d3` did not converge and `fischerd3` took 57
iterations. The Hessians were not the cause: the same happened with steps along the forces
alone and with quasi-Newton steps alone. `tests/cli_nebmain.rs::
the_chain_comes_to_rest_with_a_model_hessian_and_with_either_kind_of_step_alone`.

Allyl bromide (bromine from one end of the allyl group to the other, 10 images, the
defaults): the chain did not come to rest in 80 iterations before (forces of 1e-3 to 1e-2
after every redistribution) and is relaxed after 121 iterations now, with a symmetric
profile of two maxima of 20.006 kcal/mol and a plateau between them.

The largest force at which a run ends, with the defaults: for HCN 16, 17, 18, 19, and 20
iterations at `1e-3`, `4.5e-4`, `2e-4`, `1e-4`, and `3e-5` (the barrier is 68.2853 at
each); for CH3NC 43, 45, 48, 50, and 54 iterations with barriers of 66.9843, 66.9832,
66.9830, 66.9830, and 66.9830. The default of `nebmain` is `3e-4`.

A DIIS over the chains of the last iterations (`path::band::diis`, `nebmain -neb_opt
diis=N diis_start=F`: the chain as one vector, the forces across it as the error vector,
and the steps of the step rule as the relaxation, with the rules of acceptance of the
optimizer's GDIIS) is off by default: for CH3NC it changed the iterations from 49 to 47
(`lindh2007d3`), from 50 to 59 (`o1numhess`), and from 41 to 53 (`-fc 50`) with five chains
from a force of `5e-3` on; from a force of `1e-3` on (three or five chains) it took 59 instead of 50 (`o1numhess`). The steps of the chain are quasi-Newton
steps with updated Hessians already, which leaves the extrapolation little to add.

Two further forms of the quasi-Newton steps count on what the images do together. In the
default form every image gets its step alone, and the turning of its direction along the
chain enters as a stiffening `G / h` of the image itself, as if its neighbours stood still.

- `nebmain -neb_opt sweep=on` (`ChainRfoSettings::sweep`): the images get their steps from
  the highest one down, and the step of the neighbour uphill enters the force of an image
  (`G_i - (G / h) P s_u`). With the direction of an image pointing to its neighbour uphill
  the Jacobian of the forces of a chain is block triangular, so this is the Newton step of
  the equations of the chain at the cost of the default.
- `nebmain -gqnt` or `-neb_opt global=on` (`path::band::global`, experimental): one
  rational-function step for all images as one variable, with the model `sum_i [P H P + Q]_i
  + sum_i (G_i / h_i) |P_i (dx_i - dx_u)|^2 / 2`. MultiOptPy reads `-gqnt` ("use global
  quasi-Newton method") and does nothing with it. The Hessian of the chain (images times
  atoms times three squared) is never formed: the lowest eigenvector of the augmented
  matrix is found by Davidson's method (`math::davidson`) from the action of the Hessian
  on a vector, which is computed image by image from the Hessians of the images. Those
  stay in their files; as many as fit into `-neb_opt global_memory=MB` (256) are held in
  memory during a solve, the others are read for every application. An effective Hessian
  of an image that takes in the curvatures of its neighbours (the Schur complement of the
  neighbouring blocks) is the elimination of this block-tridiagonal model image by image;
  the sweep is its limit for the forces as they are.

Steps that are found for several images together may go against the force an image has
while its neighbours stand; the rule that holds such an image (MultiOptPy's, kept for the
steps of single images) is not applied to them. Before that the two forms were slower
than the default (CH3NC: 68 and 65 iterations against 50).

Iterations until the largest force is below `3e-4` (the default of `nebmain`), GFN0-xTB,
Hessians from `-modelhess o1numhess`:

| Steps | HCN to HNC, 8 images | CH3NC to CH3CN, 6 images | allyl bromide, 10 images |
|---|---|---|---|
| default (redistribution towards the predicted maxima every second iteration) | 17 | 47 | 121 |
| the same, `sweep=on` | 17 | 43 | 103 |
| the same, `global=on` | 23 | 53 | 117 |
| `-adpred 0 -cineb 3` (no redistribution; the image at the maximum climbs) | 17 | 38 | not in 200 |
| the same, `sweep=on` | 17 | 35 | not in 200 |
| the same, `global=on` | 20 | 32 | 110 |

The step for the whole chain is at its best without the redistribution: between two
redistributions it converges fast, and every redistribution sets the forces back by an
order of magnitude (CH3NC, redistribution every eighth iteration: `7e-4` before one and
`6e-3` after it). With a climbing image in its place it took the fewest iterations for
CH3NC and was the only form that relaxed the chain of allyl bromide, whose top is a plateau
with two maxima 0.006 kcal/mol above it; with the redistribution of the default it is no
faster than the default. The sweep is as fast as the default or up to 15 % faster. Neither
is the default: the differences are within what the iterations of one chain vary by with
the tolerance (CH3NC at `1e-4`: 50, 53, and 57 for default, sweep, and whole chain).

Tried on the way and left out: a longest step of an image (and of the chain) that follows
how well the model foresaw the gradient, halved below a ratio of 1/4 of found to foreseen
reduction and doubled above 3/4. The model foresees the gradients of the middle of a
relaxation badly (the steps along the forces are mixed in, and the directions turn), so
the radius shrank and stayed small (CH3NC: 97 iterations instead of 50, 70 instead of 36
for the whole chain). Quasi-Newton steps alone (`-rrs 1`) were unstable in every form for
CH3NC (72 to 131 iterations, or none in 150); the steps along the forces that are mixed
in by default are what keeps the relaxation steady.

### The Hessians of the images without `-modelhess`

Without `-modelhess`, `nebmain` used `o1numhess` at first, as the articles do: a Hessian
of every image from further gradients at the start and again every `-mfc`
iterations. It now uses the model Hessian `lindh2007d3`, which costs no calculation;
`-modelhess o1numhess` gives the articles' setting. Measured (GFN0-xTB, the defaults
otherwise, until the largest force is below `3e-4`; iterations, and in brackets the
height of the highest image above the first in kcal/mol, where the transition states
optimized from these images are 68.285, 66.983, and 20.006):

| `-modelhess` | HCN to HNC, 8 images | CH3NC to CH3CN, 6 images | allyl bromide, 10 images | Sum |
|---|---|---|---|---|
| `o1numhess` | 17 | 47 | 121 | 185 |
| `lindh2007d3` | 19 (68.250) | 31 (66.953) | 117 (20.006) | 167 |
| `fischerd3` | 19 (68.285) | 37 (66.977) | 119 (20.006) | 175 |
| `schlegeld3` | 17 (68.285) | 41 (66.981) | 147 (20.006) | 205 |
| `swartd3` | 16 (68.100) | 33 (66.983) | 79 (20.042) | 128 |
| `lindh2007d3 -mfc 1000` | 19 (68.250) | 31 (66.953) | 81 (20.033) | 131 |
| `fischerd3 -mfc 1000` | 19 (68.285) | 37 (66.977) | 186 (20.006) | 242 |

Gradient calculations are images times iterations for the models (152, 186, and 1170
for `lindh2007d3`); `o1numhess` adds those of its Hessians. Why `lindh2007d3`: with it the
chains came to rest in fewer iterations than with `o1numhess` in sum, and its highest
images are within 0.04 kcal/mol of the saddles. `swartd3` took the fewest iterations but
stopped with the highest image of HCN 0.19 kcal/mol below the saddle. What the numbers do
not decide: this is three small molecules with one calculator; the allyl bromide chain
(a flat top with two maxima) takes between 79 and 186 iterations with settings that
differ little, so the differences between `lindh2007d3`, `fischerd3`, and `swartd3` there
are within what a small change of the settings does; and a model Hessian that is not
computed again every `-mfc` iterations (the last two rows) changed nothing where the run
was shorter than 50 iterations and went either way for allyl bromide, so `-mfc` was left
as it is.

### The first chain

MultiOptPy puts the images between the structures it reads on straight lines (`-p`),
optionally minimizes each on the image dependent pair potential (`-idpp`), and has a copy
of the program geodesic-interpolate for a redistribution (`-adg`). mopt-rs makes the
images between two structures by one of five methods (`path::interp`, `nebmain -neb_opt
path=`):

- `linear`: the straight line.
- `lst`: the linear synchronous transit of Halgren and Lipscomb; new. Every image
  minimizes `sum (r - r(f))^2 / r(f)^4 + 1e-6 |x - x(f)|^2` on its own (the formula as
  later papers quote it; the original was not at hand).
- `idpp`: Smidstrup, Pedersen, Stokbro, Jonsson. MultiOptPy minimizes every image on its
  own function with a limited-memory update whose gradient differences have the wrong
  sign for one of the two gradients; here the chain is relaxed across the path on the same
  functions with the relaxation of `path::band`, the ends held. The publication's springs
  are left out (the images keep their places along the chain, and `-nd` spaces them
  afterwards).
- `sidpp`: the sequential variant of Schmerwitz, Asgeirsson, Jonsson (arXiv:2310.04531);
  new. The two sides take turns here (there an image is added where the last one has
  converged), and there are no springs.
- `geodesic`: Zhu, Thompson, Martinez; after their program (MIT licence,
  `third_party/GEODESIC_INTERPOLATE.md`). The program gives SciPy a dense Jacobian of
  (images x pairs) rows and (images x coordinates) columns; here the least-squares
  problems are solved with products of that Jacobian only (`path::interp::least_squares`:
  Levenberg-Marquardt with conjugate gradients), so memory and work grow with images
  times pairs. The two starts of the search for a new image are shifted by a fixed
  pattern instead of random numbers, the structures are not turned onto each other, and
  the better start is chosen without the program's sub-division.

- `fbenm`, `cfbenm`: the flat-bottom elastic network model of Koda and Saito and its
  correlated form, with the definitions of their program PyDMF (MIT licence,
  `third_party/PYDMF.md`; the papers were not accessible). MultiOptPy's `-cfbenm` has the
  correlated term alone and minimizes every image on its own; here the model is whole
  (the limits of all pairs, the planes, the correlated term; the van der Waals radii of
  Alvarez were added for it, `parameters::vdw`). The authors find the path with their
  direct MaxFlux method, which also shortens it; that method is not in mopt-rs yet, and
  the straight line is relaxed across the path on the model instead, so the images are
  where the model costs nothing but the path is not the one of the publications.

All start from structures shifted by at most 0.01 Angstrom in a fixed pattern, so that a
straight line with a symmetry (three atoms on one line) does not hold the images.

Chains relaxed from the first chains (GFN0-xTB, the defaults otherwise, until the largest
force is below `3e-4`; gradient calculations, in brackets the iterations):

| Structures read | `linear` | `lst` | `idpp` | `sidpp` | `geodesic` |
|---|---|---|---|---|---|
| HCN and HNC, both on one line | 553 (79) | 351 (27) | 806 (62) | 403 (31) | 140 (14) |
| CH3NC and CH3CN | 970 (194) | 728 (104) | 819 (117) | 833 (119) | 576 (72) |
| allyl bromide and its mirror image | 1120 (112) | 636 (53) | 1287 (117) | 1419 (129) | 852 (71) |
| HCN, a bridged structure, HNC | 152 (19) | 99 (11) | 99 (11) | 90 (10) | 99 (11) |
| CH3NC, a structure between, CH3CN | 186 (31) | 182 (26) | 259 (37) | 196 (28) | 216 (27) |

`fbenm` and `cfbenm` (the same numbers for both: the correlated term costs nothing on
these paths) in the order of the rows: 135 (15), 426 (71), 1507 (137), 135 (15), 203
(29); highest images 68.28, 68.91, 20.01, 68.23, 66.98 kcal/mol. Fewer gradients than
the geodesic for the two small rearrangements from two structures, more for allyl
bromide.

The highest images: HCN 68.22, 68.29, 68.12, 68.21, 68.28 kcal/mol in the first row
(the saddle is 68.285); CH3NC from the two ends 68.91 for `linear`, `lst`, `idpp` and
66.98 for `sidpp` and `geodesic` (the two saddles of this reaction); allyl bromide 20.01
for `linear`, `idpp`, `sidpp` and 20.06 and 20.05 for `lst` and `geodesic` (a flat top).
On the straight line of HCN and HNC the hydrogen passes through the other atoms; `idpp`
relaxed from it converges with carbon and nitrogen pulled apart in one image
(`path::interp::tests`), which is what the sequential variant was made for.

Time of the interpolation alone, 8 images, atoms on a cubic lattice of which eight are
turned (release build): 64, 216, and 512 atoms take 0.03, 0.17, and 0.34 s with
`geodesic`; 0.01, 0.08, 0.40 s with `lst`; 0.04, 0.41, 2.2 s with `idpp`; 0.07, 0.81,
4.3 s with `sidpp` (the last three use all pairs of atoms). A dense Jacobian of the
geodesic for 512 atoms would have about 10^5 x 10^4 elements.

The default is `geodesic`: over the three cases from two structures it needs the fewest
gradient calculations in sum (1568; `lst` 1715, `cfbenm` 2068, `linear` 2643, `sidpp`
2655, `idpp` 2912), it is never the worst, it is close to the best with a structure
between, and it is the cheapest but for the straight line. What this does not show: three small molecules
and one calculator; no comparison with the program itself (not installed), so the paths
are those of this implementation; nothing larger than the timing case was relaxed.

The other tables of this chapter (the Hessians of the images, the steps, the
optimizations after the chain, the chain in a periodic cell) were measured before this,
with the images between the structures on the straight line (`path=linear` now).

### Chains with a Gaussian process

MultiOptPy has an unfinished file for chains on a Gaussian-process model
(`MEP/pathopt_gpneb_force.py`: not called, with names that are not defined; all images are
calculated, the model is fitted, and the chain takes 50 steps on it). mopt-rs has the
model as a surface between the relaxation and the calculator (`path::band::surrogate`,
`workflow::chain_surrogate`, `nebmain -neb_opt gp=on`), so that it works with every force,
step, and redistribution, and with a memory that does not grow with the run: a prediction
uses the 12 nearest calculated points (the factorization of `math::gp` that never forms
the covariance matrix), a new point replaces one at nearly the same place, and 30 points
per image are kept.

What was measured first is how good such a model is (`path::band::surrogate::tests`,
HCN, GFN0-xTB): with twelve calculated points within 0.05 to 0.3 Bohr of a structure, the
predicted gradient is off by `1e-3` to `5e-2` Hartree/Bohr for every length scale and
noise tried, where a chain is relaxed at `3e-4`; the posterior variance says `1e-9` to
`1e-3` and is no measure of that error. A first form that believed the model wherever
the variance was small left the HCN chain wandering for 300 iterations. The form that is
in the code measures the error of the model against every calculated value and lets the
model answer up to `gp_between` iterations between two calculated ones only while that
error is below `gp_error` of the largest force; a chain is relaxed only on calculated
values, and the last chain is calculated.

Gradient calculations until the largest force is below `3e-4` (GFN0-xTB, the defaults
otherwise; "not" = not relaxed in 300 iterations):

| Structures read | without | `gp_error=0.1` (default) | `gp_error=0.3` | `0.3`, `gp_between=4` | `gp_error=1.0`, `gp_between=3` |
|---|---|---|---|---|---|
| HCN, a bridged structure, HNC | 99 | 99 | 99 | 99 | 188 |
| CH3NC, a structure between, CH3CN | 216 | 216 | 601 | 600 | not (926) |
| CH3NC and CH3CN | 576 | 576 | 488 | 509 | not (849) |
| allyl bromide and its mirror image | 852 | 960 | not (2932) | 2534 | not (1302) |

On the Muller-Brown surface (9 images, a tolerance of `1e-3` in its units,
`gp_error=0.3`) the chain needs 81 calculations instead of 225
(`workflow::chain_surrogate::tests`).

What this shows: the model saves calculations where the surface is simple (two
variables) and in one of the molecular cases (15 %); where it is used on the other
molecules it costs more than it saves, because the steps taken on predicted gradients
are worse than the calculated steps they replace by more than they are cheaper (the
Hessians of the images are updated with them as well). With the default the model is
almost never used for these molecules (it is as without, but for allyl bromide, 13 %
more). It is therefore off by default. What was not tried and may change the picture:
a model of the difference from the quadratic models of the images (their Hessians as
the prior), a kernel in internal coordinates, leaving the Hessians of the images alone
in model iterations, and calculating one image per iteration instead of all (the way of
Koistinen et al. and of Garrido Torres et al., whose savings are against chains relaxed
by steps along the forces alone, which take many more iterations than the chains here).

### The direct MaxFlux method

`nebmain -dmf` (`path::dmf`, `workflow::dmf`, `cli::nebmain_dmf`) is the method of Koda
and Saito (J. Chem. Theory Comput. 2024, 20, 2798) with the definitions of their program
PyDMF (`third_party/PYDMF.md`; the text of the paper was not at hand): the path is a
B-spline of degree 3 in 4 segments fitted to the first chain, the objective is
`ln(integral exp(beta E) ds) / beta` with `beta` 10 per eV taken by the trapezoidal rule
over a few points, energies from the lower end, and the points move towards the top of
the energy profile with PyDMF's gates and constants. PyDMF hands the problem to the
interior-point program IPOPT; here it is a limited-memory quasi-Newton method with
IPOPT's settings from PyDMF (the first curvature, every step accepted, the tolerance on
the largest component of the gradient, 0.04 eV/Angstrom).

MultiOptPy's `-dmf` (`MEP/pathopt_dmf_force.py`) is not ported: it fits a spline through
the images of the chain anew in every iteration, takes the speed only at the images,
keeps them at equal steps of the parameter, applies `beta = 10` to energies in Hartree
(the method means 10 per eV), and hands the gradient of the objective back as forces on
the images, which the optimizer of the chain then moves; nothing keeps the pace of the
path.

What differs from PyDMF, and why (measured on the Müller–Brown surface, energies × 1e-3,
from the minimum at (-0.558, 1.442) to the one at (-0.050, 0.467), saddle at
(-0.822, 0.624); `path::dmf::tests`):

- PyDMF takes the speed at a point from the lengths of the pieces of the path next to it
  and bounds those lengths to 1 % of each other. With two variables the spline has ten
  numbers between its ends and cannot meet sixteen such bounds; with the speed taken at
  the points alone the path went 0.48 Hartree high between them. Here the length is
  integrated along the whole spline (eight-point Gauss–Legendre rule on every segment),
  and the points sit at given shares of it (`I = L integral_0^1 exp(beta E(tau)) dtau`):
  no bounds are needed, and a path that goes round is longer.
- The objective does not change when the path is run through at another pace, so the
  control points move across the path only (their gradients and steps without the part
  along the tangent at their Greville abscissae, as images of a nudged elastic band).
  Without that the pace drifted, the spline had to be fitted anew most iterations (147 in
  200), and the path still climbed 14 mHartree above the saddle where no point was. The
  spline is still fitted anew to its own curve when its fastest part runs three times as
  fast as its slowest.
- With the points at equal shares of the length (PyDMF's `update_teval=False`, here
  `-neb_opt dmf_moves=off`) the objective went below the energy of the saddle: the path
  crosses the ridge between two points and keeps the points in the valleys. It did not
  come to rest in 200 iterations. The points move towards the top by default.
- One point between each end and the points near the top (`dmf_guards`, none in PyDMF)
  keeps the rest of the path in view.
- PyDMF has IPOPT accept every step. For HCN to HNC (GFN0-xTB) the top of the path came
  to the saddle in six iterations and the gradient nearly to the tolerance; then single
  steps threw the top 40 kcal/mol up, and the sum over the points went on falling below
  its value at the saddle path while the top rose and the path grew from 8.3 to 11.6
  Bohr: the path crossed the ridge between two points and lengthened elsewhere so that
  the points slid off the top. At large `beta` the method minimizes the highest energy of
  the path, so a step after which the top of the energy profile (interpolated with the
  slopes of the energy, which see between the points) is higher than the lowest top so
  far by more than 1 % of the barrier is taken back: the path returns to the one with the
  lowest top and the step is halved until the top comes lower. From there the gradient
  can point the same way again (HCN went round three iterations, the same three, for
  170 iterations, its top 0.003 kcal/mol above the saddle and the gradient five times the
  tolerance), so the run also ends when the top has not come down by 0.1 % of the
  barrier in eight iterations (`DmfEnd::Settled`; `nebmain` says "settled"), with the
  path of the lowest top. On the Müller–Brown surface every setting then ends in 14 to 22
  iterations with the path 0.007 to 0.021 from the saddle.
- The method measures from minima at the ends. `nebmain` minimizes the ends that `-fe`
  leaves free before the method starts (on the surface of the path, from the model
  Hessian of the surface, with the coordinates of `-fix` held); PyDMF takes the ends as
  given.

Measured (GFN0-xTB; `nebmain path.xyz -os gfn0xtb -dmf -neb_opt optimize=saddles
optimize_hessian=exact`, the three structures of each reaction as for the chains;
iterations of the method, gradient calculations of the whole run with the minimization
of the ends and the optimization of the saddle from the estimate, and the estimate of the
transition state in kcal/mol above the first minimum; the saddles are 68.285, 66.983, and
20.006, and every run below reached its saddle):

| Setting | HCN, a bridged structure, HNC | CH3NC, a structure between, CH3CN | allyl bromide, [1,3]-shift | Gradients |
|---|---|---|---|---|
| 3 points to the top, 1 guard each side (default) | 12, 71, 68.289 | 18, 111, 66.520 | 28, 242, 20.555 | 424 |
| 3 points to the top, no guards | 12, 48, 68.624 | 16, 78, 71.177 | 42, 292, 20.321 | 418 |
| 5 points to the top, no guards (PyDMF's count) | 12, 72, 68.290 | 18, 111, 66.574 | 37, 706, 21.010 | 889 |
| 5 points to the top, 1 guard each side | 24, 184, 67.903 | 23, 178, 66.914 | 30, 349, 20.617 | 711 |
| the chain (default, `-ns 300`) | 11, 101, 68.285 | 27, 217, 66.983 | 83, 1284, 20.040 | 1602 |

Nine of the twelve runs of the method ended settled, three on the tolerance of the
gradient. Without guards the estimate for CH3NC was 4.2 kcal/mol too high (the
optimization of the saddle still found it); with five points to the top the estimate for
allyl bromide was 1 kcal/mol too high and the optimization from it went 2.5 Bohr and took
517 gradients. The default is three points and one guard: about as cheap as without
guards, with estimates within 0.6 kcal/mol of the saddles. The method took a quarter of
the gradients of the chain to the same saddles here; three molecules and one level of
theory, so this is not a general ranking.

#### Against the other path methods

Gradient calculations of the whole run (the path, the minimization of the ends, and the
optimization of the saddle from the estimate with an exact Hessian at its start; GFN0-xTB,
the vacancy of copper with EAM), and where it did not end at the right saddle, the saddle
it found (kcal/mol above the first minimum). "nc": not relaxed in 300 iterations (the
saddle optimization still ran from the highest image). The chain at `tolerance=1e-3` is
the two-stage mode (`-neb_opt tolerance=1e-3 optimize=saddles`); BITSS is
`optmain A -mf bitss B` followed by `optmain -order 1` from its estimate.

From three structures (a rough one between the ends):

| Method | HCN | CH3NC | allyl bromide |
|---|---|---|---|
| direct MaxFlux (`-dmf`) | **71** | **111** | **242** |
| chain (default) | 101 | 217 | 1284 |
| chain to 1e-3, then optmain | 83 | 209 | 1174 |
| chain, Global RFO (`global=on`) | 83 | 202 | 1658 |
| chain, GP-NEB (`gp=on`) | 101 | 217 | 1768 |
| chain, forces only (`-modelhess none`) | 443 | 666 | 3740 nc |
| springs and a climbing image (`-ewbneb -cineb 3`) | 172 | 2402 nc | 4865 nc |

From the two ends only (the first images by geodesic interpolation):

| Method | HCN | CH3NC | allyl bromide | vacancy of copper (EAM) |
|---|---|---|---|---|
| direct MaxFlux, one image between the ends (before the fix below) | 173, saddle 111.2 | 1192, saddle 46.9 | 304 | **44** |
| chain (default) | 142 | 578 | 969 | 170 |
| chain to 1e-3, then optmain | **122** | **442** | 942 | 121 |
| chain, Global RFO | 181 | 730 | 1124 | 170 |
| chain, GP-NEB | 142 | 578 | 1289 | 173 |
| chain, forces only | 452 | 2407 nc | 3730 nc (saddle 19.995) | 310 |
| springs and a climbing image | 3003 nc | 521 | 4448 nc (saddle 20.054) | 2104 nc |
| BITSS | 399 | 528, saddle 69.8 | 355 | |

The direct MaxFlux method was the cheapest wherever its first path was near a good one
(from three structures, and the hop of the vacancy, a quarter to a fifth of the chain),
and its estimate of the top was within 0.6 kcal/mol, so that the optimization of the
saddle finishes it. It stays over the pass its first path crosses: the curve has few
points and its moves are taken back when the top rises, so a first path over a high pass
does not slide over to a lower one as the images of a chain do. With two ends only,
`nebmain -dmf` put one image between them (to have a chain to fit the curve to) instead of
the images the spacing asks for, which a chain gets (five for HCN): its first path went
40 kcal/mol higher than that of the chain (120 against 81 kcal/mol for HCN), and it
settled over higher passes. It now starts from the same first chain as a chain; the runs
with that are measured below. Forces only and the springs were the most expensive and
often did not relax in 300 iterations; GP-NEB gained nothing; Global RFO was about as
cheap as the default.

### Springs

`nebmain -ewbneb` (`path::band::springs`): the chains of mopt-rs keep their images apart
by redistribution; with this option they get springs instead, with constants that grow
with the energy (Asgeirsson et al., J. Chem. Theory Comput. 2021, 17, 4929; the formula
and the constants 0.1 and 0.01 Hartree/Bohr^2 as later papers and the manual of ORCA give
them, the publication itself was not at hand). An inner image gets `(k_i d_i - k_(i-1)
d_(i-1)) t` along its tangent besides its force across the chain; the quasi-Newton share
of the step gets the step that balances the two springs; an image that climbs has no
springs. It is there for comparison.

MultiOptPy's `-ewbneb` (`MEP/pathopt_ewbneb_force.py`) is not that: its constant belongs
to an image and falls from the upper one to the lower one towards the top (the other
way round), it multiplies unit vectors to the neighbours and not distances (nothing
pulls harder when stretched), and the term is added to the gradient, so that the force
pushes an image away from the next one. That form is kept with `-neb_opt
form=multioptpy` (with MultiOptPy's constants 0.005 and 0.0001) and is not the default.

Measured (GFN0-xTB, until the largest force is below `3e-4`; gradient calculations, the
height of the highest image in kcal/mol; the saddles are 68.285 and 66.983):

| Setting | HCN, a bridged structure, HNC | CH3NC, a structure between, CH3CN |
|---|---|---|
| the default (no springs) | 99, 68.285 | 216, 66.983 |
| `-ewbneb` | 135, 67.318 | 360, 66.721 |
| `-ewbneb -cineb 3` | 171, 68.285 | not in 300 iterations (67.04) |
| `-ewbneb -cineb 3 -modelhess none` (steps along the forces only) | 711, 68.285 | 624, 66.984 |
| `-ewbneb -cineb 3`, constants 0.05 and 0.005 | 153, 68.285 | not in 300 iterations (67.02) |

Without an image that climbs the highest image lies below the saddle (by 1.0 and 0.3
kcal/mol here), as expected of an elastic band. With the climbing image and steps along
the forces alone, the way the method is published, both chains reach their saddles. With
the quasi-Newton share of the step the CH3NC chain does not come to rest in 300
iterations; why was not found out (the default without springs has no such problem), so
`-modelhess none` is what to use with `-ewbneb -cineb`. In no case were the springs
cheaper than the default. `path::band::springs::tests`, `tests/cli_nebmain_springs.rs`.

### Chains in internal coordinates

`nebmain -coord dic|tric|tric-cart` (`workflow::chain_internal`): MultiOptPy's chains are
in Cartesian coordinates. Here a surface between the relaxation and the calculator turns
the images into vectors of independent internal coordinates relative to a reference
structure (the image in the middle of the first chain); the relaxation works on these
vectors as on plain variables, the structure of a vector is found by the iterative
back-transformation from the nearest structure known, and the gradient and Hessian of the
calculator are taken to the coordinates with the Wilson matrix. Coordinates of the
position and orientation of the whole structure (in `tric`) are made stiff in the
Hessians. The transformation is exact for a structure without straight angles (the
gradient in the coordinates against differences of the energy along them: `2e-9`,
`workflow::chain_internal::tests`).

Measured (GFN0-xTB, gradient calculations until the largest force is below `3e-4`;
"not" = not in 300 iterations):

| Structures read | Cartesian | `-coord dic` | `-coord tric-cart` |
|---|---|---|---|
| HCN, a bridged structure, HNC | 99 | 81 | (falls back to `dic`: 81) |
| CH3NC, a structure between, CH3CN | 216 | not | not |
| CH3NC and CH3CN | 576 | not | not |
| allyl bromide and its mirror image | 852 | 1020 | a chain that collapsed (804) |

What this shows: the chain in internal coordinates is better for HCN and worse or
useless for the others. Coordinates made of angles break down where an angle comes near
180 degrees (the C-N-C of methyl isocyanide): the back-transformation ends 1 to 4 away
from what was asked for, the image counts as failed, its step is taken again shorter,
and the chain never comes to rest. One set of coordinates for a whole chain cannot be
set up anew on the way as an optimization of one structure does. `tric` and `tric-cart`
cannot be set up at all where a structure of the chain is a linear molecule; then `dic`
takes their place, and Cartesian coordinates if nothing can be set up. Cartesian
coordinates stay the default. Not tried: coordinates without angles (interatomic
distances with weighted Cartesian coordinates), which have no such places.

The Gaussian-process model can see structures by their inverse interatomic distances
(`-neb_opt gp_features=distances`, `path::band::surrogate::Features`; after Koistinen et
al. 2019, with the gradient taken to the distances by least squares). For HCN the error
of its gradients is 5 to 10 times smaller than with Cartesian points (for example
`5.4e-4` instead of `6.6e-3` Hartree/Bohr with points 0.15 Bohr around, length scale 0.1
per Bohr; `path::band::surrogate::tests`). In chains it did not help: CH3NC with a
structure between needs 216 gradient calculations with either at `gp_error=0.1` and
1035 (distances) against 601 (points) at `0.3`; CH3NC from two structures 609 against
576 at `0.1`, 1079 against 488 at `0.3`; HCN 99 with all. The better model does not make
the steps taken on it good enough. It is an option, not the default.

### Optimizations after the chain

`nebmain -neb_opt optimize=on|saddles|minima` (`cli::nebmain_followup`; off by default)
runs `optmain` from the images of the relaxed chain: the images at the maxima of the
profile with `-order 1`, the end images and the images at minima as minimizations, each
on the surface of the chain. MultiOptPy has the steps as separate runs (`run_autots`
chains them from a settings file). The first Hessian of an optimization is, by default,
the one the relaxation kept for the image (`optimize_hessian=chain`). For a saddle point
`H + (c - t.H.t) t t^T` is used, with `t` the direction of the chain at the image (the
difference of its neighbours without rigid motion) and `c` the second difference of the
energies along the chain: the images move across the chain, so the updates teach the
Hessian little about the direction along it, where a model Hessian is positive, while
the profile has a maximum there. Not handed over, with the calculator's Hessian at the
first point instead: when the relaxation kept no Hessians, when the chain did not come to
rest (the kept Hessians then belong to the next chain), and with bias potentials (the
kept Hessian contains their curvature, which `optmain` adds again).
`tests/cli_nebmain.rs::the_candidates_of_a_chain_go_on_to_optmain`.

Every optimization says how far it went from its image (`moved_bohr` in
`optimized_images.csv`, the distance of the two structures at their centroids). When that
is further than the nearest image of the chain is away, a warning says that what was
found need not be the stationary point of the chain (`left_its_place`).

A converged optimization to a transition state need not end at one: on the aldol
reaction of MultiOptPy's test (formaldehyde and vinyl alcohol, GFN0-xTB), the
optimizations from the tops of chains of 300 iterations that had not come to rest found
thirteen distinct "transition states". Seven had no imaginary frequency (minima, among
them the separated reactants and hydrogen-bonded complexes), two only a flat one (-10 and
-14 cm^-1), and four one each: the aldol transition state (502i cm^-1, its IRC ends at
3-hydroxypropanal and at a complex of the reactants), a rearrangement of a complex (both
ends at the same complex), the step from the separated reactants to a complex, and one
between two structures far above the reactants. So the transition states get a
vibrational analysis by default (`optimize_check=saddles`; `all` for the minima too,
`off` for none; one exact Hessian each), `optimized_images.csv` gives the number of
imaginary frequencies, and an optimization that ended with the wrong number is reported
with a warning.

A chain that is stopped early and `optmain` after it (`-neb_opt tolerance=T
optimize=saddles`), measured with the defaults otherwise (GFN0-xTB; copper: the built-in
embedded-atom potential, 31 atoms in a periodic cell). Gradient calculations of the
chain plus those of the optimizations of the highest images; in brackets the iterations
of the chain and of the optimizations:

| Tolerance | HCN to HNC | CH3NC to CH3CN | vacancy in copper | allyl bromide (two maxima) |
|---|---|---|---|---|
| `3e-4` (the default), `optimize_hessian=chain` | 155 (19; 3) | 189 (31; 3) | 169 (24; 1) | 1187 (117; 7, 10) |
| `3e-4`, `exact` | 154 (19; 2) | 188 (31; 2) | 170 (24; 2) | 1188 (117; 13, 5) |
| `1e-3`, `chain` | 122 (15; 2) | 178 (29; 4) | 142 (19; 9) | 1076 (102; 52, 4) |
| `1e-3`, `exact` | 122 (15; 2) | 177 (29; 3) | 135 (19; 2) | 1071 (102; 9, 42) |
| `3e-3`, `chain` | 107 (13; 3) | 154 (25; 4) | 120 (14; 22) | 540 (45; 90), warning |
| `3e-3`, `exact` | 107 (13; 3) | 152 (25; 2) | 103 (14; 5) | 628 (45; 178) |
| `1e-2`, `chain` | 99 (12; 3) | 93 (14; 9) | 67 (7; 18), warning | 674 (34; 234, 100), warnings |
| `1e-2`, `exact` | 99 (12; 3) | 90 (14; 6) | 57 (7; 8) | 707 (34; 137, 230) |

`exact` costs one Hessian of the calculator per optimization besides (not in the
numbers). What this shows:

- For the three barriers with one clear top, the looser the chain the cheaper the whole:
  a third to two thirds fewer gradients at `1e-2` than at the default, and the same
  transition states (68.285, 66.983, and 12.18 kcal/mol above the first image).
- Allyl bromide has a flat top with two maxima. From chains stopped at `3e-3` and `1e-2`
  the optimizations take 90 to 234 iterations, go 1.2 to 2.2 Bohr from their images
  (the images are 0.6 to 0.75 Bohr apart), and end 20.02 and 20.04 kcal/mol above the
  first image where the chain relaxed to the default has 20.006: other stationary points
  than the chain passes, which the warning reports. At `1e-3` both are found (20.006).
  Whether the points found from the loose chains are saddle points of first order was
  not checked.
- The Hessian of the chain is as good a start as the calculator's when the chain is
  relaxed to the default tolerance (one iteration more or fewer) and costs nothing; from
  a loose chain the calculator's Hessian takes fewer iterations (copper at `3e-3`: 5
  against 22).
- The default tolerance was left as it is. `tolerance=1e-3` was safe in all four cases
  and saves 6 to 21 % of the gradients; larger values save more where the barrier has
  one top, and the warning tells when the chain was stopped too early. Nothing goes back
  to the chain by itself in that case.

### Chains in periodic cells

MultiOptPy's `nebmain` has no periodic cells. Here a chain of structures with a cell
(`workflow::chain_input::join_in_the_cell`, `ChainSettings::turn`,
`path::band::align::centre_chain_at`): the structures are joined by the nearest periodic
images of every atom and shifted to the centroid of the first; after every step the
images are shifted to that centroid again and are not turned (the cell does not turn
with them; with `-fix` nothing is shifted, as without a cell); the model Hessians, the
directions of rigid motion (shifts only), and the calculators already used the minimum
image. Tested with the built-in embedded-atom potential: the hop of a vacancy in copper
(31 atoms in 2 x 2 x 2 cells, the second structure given with the atom at another
periodic image and all atoms shifted) comes to rest in 24 iterations with
7 images, the highest image 0.5282 eV above the ends, and a transition-state
optimization from that image ends at 0.5281 eV
(`tests/cli_nebmain_periodic.rs`). Not covered: a cell that changes along the chain;
wires, where a turn about the axis is a rigid motion that is not removed; slabs were not
run.

### The quadratic string options (`-qsm`, `-qsmv2`)

In MultiOptPy `-qsm` and `-qsmv2` choose a force (`MEP/pathopt_qsm_force.py`: the gradient
without its part along the displacement of every atom to the neighbouring image, the rule
of the default force; `MEP/pathopt_qsmv2_force.py`: the gradient without its part along
one tangent of the whole structure, carried outwards from the highest image by the arcs
and parabolas of Ayala and Schlegel) and, with Hessians, a step rule of their own
(`Optimizer/rfo_neb.py:RFOQSMOptimizer`, quasi-Newton steps on the projected Hessians of
the images). The chain of mopt-rs has that structure already, so the two options select
the direction on it and nothing else: `-qsm` the direction atom by atom, `-qsmv2` the
carried tangents (`path::band::tangents`). MultiOptPy's step rule for them is not
reproduced, and there are no reference data for them; with `form=multioptpy` they are
refused.

One point is changed in the carried tangents. MultiOptPy (and ase_qsm) carry the tangent of
the highest image, which points to the next image, down both sides. Towards the first
image the chord then points against the tangent, the angle between them is near 180
degrees, and the parabola is used even for a nearly straight chain, where it agrees with
the arc only to first order in the bend. mopt-rs carries the tangent down each side
pointing away from the highest image, so that both sides use the same formulas and the arc
is exact on both for a circular chain (unit test). The paper of Ayala and Schlegel was not
at hand; the formulas are those of the two programs. With several maxima of the profile
the direction towards the higher neighbour is used. The directions are given the length of
the distance to the higher neighbour, as the directions of the default rule have it (the
quasi-Newton steps read the coupling of neighbouring images from that length).

Measured on HCN to HNC (9 images, GFN0-xTB, tolerance 1e-4 Hartree/Bohr; iterations and
the highest image above HCN, the saddle being 68.285 kcal/mol):

| Direction | With `-adpred 2` (default) | `-adpred 0` | `-adpred 0 -cineb 3` |
|---|---|---|---|
| towards the higher neighbour (default) | 13, 68.284 | 10, 67.651 | 12, 68.285 |
| `-qsm` | 21, 68.277 | | |
| `-qsmv2` | not within 80, 68.285 | not within 80, 67.608 | not within 80, 68.285 |

With `-qsmv2` the largest force stays near 1e-3 Hartree/Bohr at the image next to an end.
The carried tangents are exact for a circular chain, but on a path whose curvature changes
each arc passes its error on to the next image, and at the far images the tangent is not
the direction of the chain; what is left of the gradient along the chain then counts as a
force, which the redistribution undoes (or, without it, lets the image creep down the
path). A rule that also removes the gradient along the chord would stop that; it was not
tried. One molecule and one engine.

### Chains on surfaces of two electronic states

`nebmain -mf NAME [CHARGE MULTIPLICITY]` is new (MultiOptPy has `-mf` for `optmain` and
`ieipmain` only). The surface of `optmain -mf` (`pes::combine`, `cli::optmain::
state_surface`) is made once per image (`workflow::chain::CalculatorSurface::with_states`),
so that a function with a memory (the second direction of the branching plane of
`opt_meci`, which is updated from evaluation to evaluation) keeps it image by image; when
images are taken out the surfaces follow, and when images are added (`-aneb`) all start
anew. Tested: the seam function of the ground state and the first excited singlet of the
built-in GFN0-xTB on three images, whose energies are those of the function of the two
states calculated one by one (`tests/cli_nebmain.rs::
a_chain_on_a_model_function_of_two_states`). Not covered: a relaxed chain between two
crossing points of a molecule has not been run to convergence and looked at; with
`-modelhess o1numhess` and `-fc` the Hessians of both states are asked for and combined as
in `optmain`, which is untested for chains; the projections (`opt_mesx`, `opt_meci`) give a
gradient that is not the gradient of a function, so the energies of the profile (the mean
of the states) and the forces do not belong to one surface, and the redistribution towards
predicted maxima (`-adpred`) uses both.

### Images added and taken out

`nebmain -aneb N FREQ` (`path::band::images`, MultiOptPy's adaptive chain after Maragakis
et al.): after every `FREQ` iterations `N` images are added on either side of every image at
a maximum of the profile. MultiOptPy puts them on the straight lines between the images
and computes all Hessians anew; here they lie on the curve through the images, an image
that was there keeps its Hessian and the pairs of its update (under its new number in the
files), and a new image starts with a copy of the Hessian of the image nearest to it
(MultiOptPy's way with `form=multioptpy`). CH3NC with images 1 Angstrom apart (4 images):
without it 140 iterations to a chain whose highest image is 61.26 kcal/mol above the first
(the top of the barrier lies between two images); with `-aneb 1 10` 54 iterations, 12
images, 66.983 kcal/mol (the saddle), 636 gradients against 360 for the default chain of 6
images. Not together with `-pc`, `-gfix`, `-uc` yet (those are kept image by image).

An image whose calculation fails (no convergence of the wavefunction, a program that
stopped) is left out by MultiOptPy, and the run goes on with the others. By default
mopt-rs first takes the step of that image again, at half, at a quarter, and at none of
its length (where the image was calculated in the iteration before), and takes the image
out only when that fails too or there is no such point; `-neb_opt failed=drop` is
MultiOptPy's way (also what `form=multioptpy` does), `failed=stop` ends the run with the
error. What happened is written to the log and is in `ChainReport::failures`. Other errors
(a wrong setting, a file that cannot be written) end the run as before.
`workflow::chain::tests::an_image_whose_calculation_fails_is_put_nearer_or_taken_out`,
`path::band::machine::tests`, `tests/cli_nebmain.rs`.

What these show:

- redistribution is not needed for a chain to come to rest, but only the redistribution
  towards the predicted maxima puts an image on the top of the barrier (the same barrier
  with 8 and with 12 images); without it the highest image is where its steps left it;
- gathering with the weight 1 does not change the barrier or the number of iterations
  much; with the weight 3 a chain of 8 images did not converge in 60 iterations (too few
  images are left for the rest of the path), a chain of 12 did;
- removing the gradient along one direction of the whole structure (the tangent of the
  chain, as in ase_qsm) converges in half the iterations or fewer than removing it atom by
  atom (MultiOptPy's rule, the default). Atom by atom, an atom that hardly moves along the
  path (the hydrogens of the methyl group) has a direction that is noise, and the
  direction of every atom turns with the smallest step of that atom, which makes the
  model stiff and the steps short. The direction of the whole structure is therefore
  the default; one direction per atom is a setting (`ChainSettings::block`,
  `nebmain -neb_opt projection=atom`);
- the image on a predicted maximum is on the saddle: a transition-state optimization from
  that image (`optmain -opt rsirfo_bofill -order 1 -fc 1`) ends at the same energy within
  0.001 kcal/mol in all three cases tried (HCN to HNC: 68.285 kcal/mol, imaginary wave
  number 1286i; CH3NC to CH3CN from the chain of 6 images: 68.906, 329i; from the chain of
  10 images: 66.983, 701i), and the IRC from these transition states (`ircmain`, LQA, step
  0.1) ends in the two isomers, with the barrier of the chain as the highest energy of its
  profile. The chains of 6 and of 10 images of CH3NC went over two different saddles: a
  chain finds a path near the one it is started on, and how it is started (here, how many
  images are put on the rough first path) decides which.

Two molecules and one calculator are little; the numbers say what was seen, not what
holds in general.

### The first chain on a Bezier curve (`-ndb`)

`-ndb DISTANCE` is MultiOptPy's placement
(`Interpolation/binomial_interpolation.py:distribute_geometry_by_length_bernstein`): the
images of the first chain are the control points of one Bezier curve, and the new images
are the points of the curve at the parameters `s / L` for `s = 0, DISTANCE, 2 DISTANCE,
...` and at its end (`L` is the length of the chain). Two things differ:

- MultiOptPy sums the Bernstein polynomials with their binomial coefficients, which for
  a chain of some hundred images (the trajectory of a biased optimization) are numbers
  far beyond the precision of the sum; mopt-rs evaluates the same curve by the steps of
  de Casteljau (weighted means of neighbouring points), which lose nothing. For short
  chains the two agree;
- nebmain of mopt-rs has a default distance for `-nd` (0.5 Angstrom) that MultiOptPy
  has not: with `-ndb` that default is not applied, so that `-ndb` alone does what it
  does in MultiOptPy. `-nd` given on the command line is applied before `-ndb`, as in
  MultiOptPy.

The curve does not pass through the images between the ends. It is a way to smooth a
rough first path and not a way to space images: the points at equal steps of the
parameter are not at equal distances.

### The path of a search in a periodic cell

The frames that the searches between two structures write (their trajectories and
`<stem>_path.xyz`) did not carry the lattice of a periodic cell. A path of a search in a
cell that was given to `nebmain`, as step 2 of `run_autots` does, was therefore read as a
path of clusters, and the steps after it ran without the cell, without a word. The frames
have the lattice now; `tests/cli_autots_periodic.rs` runs the nine searches as step 1 in
a cell and one of them through step 4.

### Images calculated at the same time

MultiOptPy calculates the images of a chain one after the other. mopt-rs calculates the
images of an iteration at the same time where they are calculations that do not depend on
each other (a chain of structures in Cartesian coordinates on one calculator): the built-in
calculators in parallel threads, the engines of the Python side in worker processes when
`-parallel N` is given (`calculators.md`, "Several structures at once"; `-core` stays the
threads of one calculation, as in MultiOptPy). The numbers
are the same as one after the other for calculators without a memory; a program that
starts its SCF from the orbitals of the structure before may differ in the last digits of
its convergence. Chains whose images depend on their order (followed excited states,
ONIOM with microiterations, internal coordinates, a Gaussian-process surface, functions
of several states) are calculated one image after the other, as before.

## IRC

`optmain -irc STEP_SIZE MAX_STEP METHOD` runs as in MultiOptPy (the same methods, step
rules, stopping rules, files, and columns), and `ircmain` runs the same calculation for a
given structure; [irc.md](irc.md) describes both. The integrators are ask/tell state
machines (`irc::Irc`), so they work with every calculator and from Python. What differs:

### Constraints and combined surfaces (new)

MultiOptPy's IRC after an optimization ignores the constraints of the optimization
(`-fix`, `-pc`, ...) and the model functions (`-mf`): the path is the one of the free
surface of one state, although the structure it starts from is a saddle of the
constrained or combined one. mopt-rs follows the path within the constraints (projected
mass-weighted gradient and Hessian, points brought back onto the constraints in the
mass-weighted metric; [irc.md](irc.md), "Constraints and combined surfaces") and on the
combined surface of `-mf` (functions of the states; the projections for crossings are
refused), in `optmain -irc` and in `ircmain`. `tests/irc_constraints.rs`. `optmain -vc`
followed by `-irc` was not possible in MultiOptPy (no variable-cell optimization); here
the IRC of the atoms and the cell together follows (`ircmain -vc` as well;
`tests/irc_variable_cell.rs`).

### Errors of MultiOptPy's IRC that are not repeated

Each entry names the regression test that shows the difference.

- **Arrival test** (`IRC/converge_criteria.py`): `abs(grad.max())` is the largest component
  with its sign, so a gradient with only negative components passes however large they are.
  The largest absolute component is tested. `irc::convergence::tests`.
- **Displacement from the transition state** (`irc.py:saddle_check`): the eigenvector of
  the mass-weighted Hessian is added to the Cartesian coordinates without the division by
  the square roots of the masses; for a meta-IRC the Cartesian gradient is normalized and
  then divided by them. Both displacements are the mass-weighted ones (`dx = M^(-1/2) dq`).
  `irc::start::tests`.
- **Is the start a transition state** (`irc.py:saddle_check`): the gradient without the
  bias potentials is tested, so a transition state found with a bias potential always gave
  a meta-IRC; and the RMS threshold (0.0001) is below that of the optimizer (0.0002), so
  `optmain -irc` gave a meta-IRC from many converged transition states. The gradient of the
  surface that is followed is tested, and a structure the optimization converged to passes.
  `irc::start::tests`, `tests/cli_ircmain.rs::optmain_irc_follows_the_optimized_saddle`.
- **Hessian along the path** (`IRC/lqa.py`, `rk4.py`, `dvv.py`): the starting Hessian
  already holds the Hessian of the bias potentials, which is added again at every step;
  `iter % FC_count == 0` is true for the default `-fc -1` (Python's `%`), so the Hessian is
  replaced by that of the transition state at every step and updated once; and an exact
  Hessian is updated with the step that led to its point. Here the Hessian without bias
  potentials is updated from point to point, replaced by exact ones every `-fc N` points
  (never without `-fc`), and the Hessian of the bias potentials is added once.
  `irc::history::tests`, `irc::tests::exact_hessians_at_an_interval_are_asked_for`.
- **HPC** (`IRC/hpc.py`): the interpolant is built and the corrector integrated in
  Cartesian coordinates with gradients that are not mass-weighted (the corrected path is
  not the IRC), and the starting Hessian, already mass-weighted, is mass-weighted twice
  more. Everything is done in mass-weighted coordinates. The corrector (Euler, 100 steps,
  as MultiOptPy) halves a step that would raise the energy, so the path ends in a minimum
  instead of stepping across it until `MAX_STEP`. `tests/irc_muller_brown.rs`,
  `irc::tests::every_integrator_reaches_both_minima_from_both_saddles`; the interpolant
  and the corrector away from minima agree with MultiOptPy (`tests/golden_irc.rs`).
- **DVV** (`IRC/dvv.py`): every point is moved with the acceleration of the point before it,
  the first point is evaluated twice, and the error estimate mixes points of different
  steps. The velocity Verlet equations and the error estimate are those of Hratchian and
  Schlegel 2002. `irc::tests::dvv_evaluates_every_point_once_and_euler_goes_past_its_second_point`,
  `tests/irc_muller_brown.rs` (1.3e-2 from the reference path).
- **Euler** (`IRC/euler.py`): stops with an `AttributeError` at its second point (it reads
  a Hessian it never set). It runs; the Hessian is kept as in RK4. Same test.
- **Curvature of Euler and RK4** (`PESAnalyzer/calc_irc_curvature.py`): the "curvature" is
  the change of the gradient per step size (of the gradient with bias potentials against
  the one without), not of the unit tangent. `irc_curvature_properties.csv` holds the change
  of the unit tangent per mass-weighted length and its components along the modes of
  positive curvature. `irc::tests::the_curvature_of_a_circular_valley_is_the_inverse_of_its_radius`.
  (The full analysis of a path is `freqmain -path`.)
- **ModeKill** (`IRC/modekill.py`): its BFGS update is skipped along negative curvature, so
  an imaginary mode could never turn real between exact Hessians; it worked because the
  default `-fc -1` computes the exact Hessian at every cycle (`cur_cycle % -1 == 0`). The
  update is Bofill's, exact Hessians come with `-fc N` or `modekill:update=off`, the modes
  are those of the surface with the bias potentials, and the last row of the table has the
  values of the last structure (MultiOptPy writes 0.0 for the energy with bias potentials).
  `irc::tests::modekill_leaves_a_maximum_along_its_second_mode`.
- **Unknown method**: MultiOptPy runs LQA for a name it does not know; here it is an error.
  `irc::registry::tests`.
- **Bending angle**: `arccos(cos + 1e-8)` is NaN for parallel vectors and 179.992 degrees
  for a straight path; the cosine is limited to [-1, 1]. `irc::history::tests`.
- **Files**: the plots of a side are written into its directory (`irc_forward/
  plot_IRC_bending_0.png`); MultiOptPy writes `irc_forwardplot_IRC_bending_0.png` into the
  run directory (a missing path separator, also in the name of its `end.txt` stop file;
  mopt-rs stops on the `STOP` file of every command). `tests/cli_ircmain.rs`.
- **Frame of the starting structure**: every path point is moved to its centre of mass;
  MultiOptPy copies the starting structure into `irc_combined_path.xyz` as the input has
  it, so the frames of the combined path are not in one frame. The start is moved as well.
  `tests/irc_hcn.rs::optmain_irc_from_a_guess_reaches_hcn_and_hnc`.

### Defaults that differ: failed steps and the end of a side

MultiOptPy keeps a point of the path whose energy is above that of the point before it, and
ends a side at the first point whose gradient is below 0.0004 / 0.0001, before the minimum.
By default mopt-rs (LQA, Euler; RK4: the last item only) tests the first point of each side
and corrects it on a hypersphere when it fails (`first_step=check`), takes a step again at
half its length when it raised the energy or went past the floor of the valley
(`energy_rise=retry`, `overshoot=retry`), halves the step length where the path turns back,
and goes on below MultiOptPy's thresholds until the gradient is below those of `optmain
-tcc` (`approach=on`), so that the end point of a side is the minimum. `-irc_opt
first_step=off energy_rise=off approach=off` gives MultiOptPy's behaviour (with it every
step has the step size, and the files agree with MultiOptPy as before). Measured on HCN,
CH3NC, and allyl bromide with GFN0-xTB: no point of higher energy against one to five per
path, and end points from which an optimization with tight thresholds has nothing to do
against 19 to 86 iterations, at 8 to 16 % more evaluations; [irc.md](irc.md), "Steps that fail, and where a side ends",
has the table and what else was tried. `tests/irc_flat_valley.rs`, `tests/test_irc.py`.

### Masses

MultiOptPy's IRC weights with the masses of the most abundant isotopes
(`Parameters/parameter.py:atomic_mass`: C 12.0, N 14.003074, H 1.007825). mopt-rs weights
with the standard atomic weights (C 12.0107, N 14.0067, H 1.00794), the same as its
vibrational analysis, so that the IRC and the analysis of its path (`freqmain -path`) use
the same coordinates. `-irc_opt masses=isotope` gives MultiOptPy's masses; the comparisons
with MultiOptPy use them (`tests/golden_irc.rs`: LQA step, RK4 run).

### Kept as in MultiOptPy

The step counting (`range(1, max_step)`: `MAX_STEP - 1` points for LQA, HPC, DVV, RK4;
`MAX_STEP` for Euler), the first displacement of 0.1 of the step size and the first step of
0.05 of it straight down the gradient (LQA, RK4), the 20000 steps in which LQA sums the
length of its step and its limit of the gradient norm (1e-8; 1e-6 in the predictor of HPC),
the modes with `|eigenvalue| < 1e-8` that LQA leaves out, `expm1` with its limit for small
arguments, the FSB update of LQA and the BFGS update with `s.y > 1e-10` of RK4 and HPC, the
100 Euler steps of the HPC corrector, DVV's `v0`, `dt0`, tolerance, and limits of the time
step (0.025 to 3 fs), the oscillation rule (5 reversals in a row), the arrival thresholds
and the 10 points before they are tested, ModeKill's overlap threshold 0.3 (its threshold on the modes is now the one criterion of the
crate, a mass-weighted eigenvalue below -1e-8, about -0.51 cm^-1; `nu_thresh=-5` gives
MultiOptPy's -5 cm^-1) (its
factor from eigenvalue to wavenumber, 5140.48678, is replaced by the CODATA 2018 5140.48714), and that DVV does not end at a minimum by itself (nor RK4 with
`approach=off`; `-irc_opt energy_rise=stop` ends them there). The tables show the values of the surface at
the predicted points of HPC and the structures of its corrected points, as MultiOptPy's do.

### Added

- `ircmain` (the IRC of a given structure or variables file), with `--config`,
  `--dump-config`, `--restart`, and the `STOP` file.
- GS2 (`-irc ... gs2`), with a step that is halved where the energy rises.
- The test and correction of the first step (`-irc_opt first_step=check`, the default);
  its limits and the measurements behind them are in [irc.md](irc.md).
- The step control (`energy_rise`, `overshoot`, `bend`, `approach` and its thresholds) and
  its counts in the log, in `BranchSummary`, and in the result of `mopt_rs.irc_path`.
- `-irc_opt` (`initial`, `sides`, `masses`, `energy_rise`, thresholds), options of the
  methods after colons (`dvv:v0=...`), plain variables, `irc_profile.csv`,
  `irc_first_step.csv`, `-path_analysis`.
- Observers in Rust (`workflow::irc::IrcObserver`) and a callback in Python
  (`mopt_rs.irc_path(..., callback=...)`) that see every step and can end the run.

Tests: `crates/mopt-rs/src/irc/`, `tests/irc_muller_brown.rs`, `tests/irc_hcn.rs`,
`tests/cli_ircmain.rs`, `tests/golden_irc.rs`, `tests/test_irc.py`.

## Transition state between two structures (`-mf bitss`)

### BITSS follows the published method

Reference: Avis, Panter, Kusumaatmaja, J. Chem. Phys. 2022, 157, 124107 (arXiv:2209.02541).
MultiOptPy's `-mf bitss` (`BITSSModelFunction` and the BITSS branch of
`ModelFunctionHandler`) has the function, its gradient, and its Hessian of the paper, but
drives them differently. mopt-rs (`path::bitss`, `optmain -mf bitss REFERENCE`) follows the
paper:

| | paper and mopt-rs | MultiOptPy 1.21.2 |
|---|---|---|
| separation asked for | `d_i = (1 - f) d_(i-1)`, reduced when the minimization of a stage has converged (Eq. (1) and the three steps of section II) | multiplied by `1 - f` in *every* iteration, and set to the current distance every 500 iterations; after a few iterations the target is zero and the images are simply pulled together |
| barrier estimate `E_B` | highest energy on the straight line between the images less the mean of their energies | `abs(E1 - E2)`: the difference of the two starting energies (zero for equivalent minima) and, when it is set again, close to zero once the energy constraint is met, so that `kappa_e = alpha / (2 E_B)` is as large as the guard `1e-10` allows |
| coefficients | set at the start of every minimization and every 100 iterations of it | every 500 iterations of the run |
| `beta` | 0.1 | 0.02 (changed "for stronger constraint") |
| end | separation, gradient at the midpoint, or shift of the midpoint | the convergence test of the optimizer on the `6N` gradient |
| bias potentials | part of the surface of each image, so they enter `E1 - E2` | added to the function afterwards |

What mopt-rs adds, with the reason:

- **The length of a step is limited to half the separation asked for** (`step_share`).
  Without it a long step of a quasi-Newton minimizer carries an image over the ridge, and
  the search then ends at another transition state or none.
- **A pair that left the ridge is recognized**: when the search would end and the midpoint
  is not higher in energy than both images, the result is "no barrier", not a transition
  state. (The gradient at the midpoint also vanishes at a minimum.)
- The images of a converged minimization are not evaluated again for the survey of the
  line between them, and the minimization of each stage starts from a Hessian that has
  the curvature of the two penalty terms, which is known exactly.
- The midpoint of the final images is written as the estimate of the transition state
  (`<stem>_ts_guess.xyz`); MultiOptPy leaves the `2N`-atom structure only. The reference is
  laid onto the input first; MultiOptPy takes the distance of the two files as they are.

Measurements on the Muller-Brown surface (`path::bitss::tests::bitss_measurement`; minima A,
B, C with C between A and B; from B to C, from A to C, and from A to B across C, where the
higher transition state is the one sought). Iterations of the minimizer; "other" is an end
at another transition state, "-" the iteration limit of 3000:

| setting | `lbfgs` | `rsirfo_bfgs` | `rsirfo_fsb` | `fire` |
|---|---|---|---|---|
| no step limit | 72 / 91 / 107 | 62 / other / 87 | - / other / 85 | 406 / 590 / 860 |
| `step_share` 1.0 | 67 / 76 / other | 58 / other / 88 | other / other / 84 | 406 / 590 / 860 |
| `step_share` 0.5 (default) | 62 / 75 / 86 | 48 / 66 / 87 | 48 / 62 / 86 | 406 / 590 / 860 |
| `step_share` 0.25 | 65 / 72 / 95 | 50 / 69 / 87 | 50 / 64 / 89 | 406 / 590 / 860 |
| `step_share` 0.1 | 93 / 111 / 140 | 74 / 99 / 120 | 76 / 97 / 117 | 359 / 482 / 900 |
| `beta` 0.02 (MultiOptPy's) | 72 / 94 / 118 | 52 / 75 / 100 | 53 / 72 / 77 | 608 / 842 / 1237 |
| `f` 0.25 | 151 / 173 / 216 | 95 / 119 / 152 | 97 / 110 / 136 | 809 / 1418 / 1656 |
| `f` 0.75 | 36 / 73 / other | 45 / 53 / 72 | 34 / 50 / 62 | 315 / 465 / - |
| coefficients every 20 iterations | 76 / 75 / 86 | 48 / 66 / 92 | 48 / 62 / 90 | 688 / 893 / 1353 |

The rows from `beta` on use the default step limit. One, three, or five points between
the images for the barrier estimate give the same counts within three iterations (the
midpoint is the highest point on these lines); three is kept so that a line across an
intermediate basin is not judged by its midpoint alone, at the cost of two energies per
stage. The curvature of the penalty terms in the starting Hessian saves 2 to 5 % of the
iterations of the two RS-I-RFO methods (201 and 196 against 206 and 206 in the sum of the
three searches). Adam-type methods without a line search do not converge the first stage
within 3000 iterations.

Not done: no comparison with the authors' own program, and no molecular benchmark beyond
the two tests (an adatom on four copper atoms with EMT, 33 iterations; the inversion of
ammonia with GFN-FF, 31 iterations). The distance is the Euclidean one of the paper; a
mass-weighted or internal-coordinate metric, which the paper allows, is not offered. A
variable cell (`-vc`) and constraints other than fixed atoms cannot be combined with it.

Tests: `path::bitss::tests`, `tests/cli_bitss.rs`, `tests/test_bitss.py`.

## Transition states from two structures or one (`ieipmain`)

`ieipmain` runs the improved elastic image pair method (iEIP, the default), the two-point
scaled hypersphere search (`-2pshs`), the growing Newton trajectory (`-gnt`), the
anharmonic downward distortion following (`-addf`), the dimer method (`-use_dimer`), the
spring pair method (`-use_spm`), or a model function of two states (`-mf`), chosen in
MultiOptPy's order of precedence (`-mf`, `-gnt`, `-addf`, `-2pshs`, `-use_dimer`,
`-use_spm`); the methods that are not run are named in the log. The options keep their
names and meanings; the publications are in `third_party/TWO_STRUCTURE_SEARCHES.md`.

### iEIP keeps MultiOptPy's procedure

The iterations of iEIP are MultiOptPy's: the terms of the displacement and their order of
summation, the DS-AFIR weights, AdaBelief without bias correction (0.9, 0.999, rate 0.01),
the limit of 0.04 Bohr per displacement, the separation `max(0.9 L, 0.02)` asked for, the
growth of `-beta` by 2 % whenever the images moved apart, and the end below 0.15 Bohr.
`tests/golden/ieip_lj_ar7_{default,beta,bias,micro}.json` (MultiOptPy 1.21.2 on Ar7 with
the Lennard-Jones calculator; `tools/golden/run_ieip_reference.py`) are reproduced to
1e-11 Bohr in every image of every iteration (`crates/mopt-rs/tests/golden_ieip.rs`), and
so are the files: `approx_TS.txt`, `approx_EQ.txt`, `local_min_grad.txt`,
`<stem>_traj.xyz`, the four `plot_*.png`, and `input.txt`. The directories
`samples_<stem>_<k>` are not written: every iteration appends the two images to a trajectory
each (`<stem A>_image_1_traj.xyz`, `<stem B>_image_2_traj.xyz`), which become
`<stem>_traj.xyz` at the end, and shows them in Cartesian coordinates on the screen and in
`mopt.log` (`Image 1`, `Image 2`; MultiOptPy shows no structure). The other methods of
`ieipmain` follow the same rule (the table in [reaction_paths.md](reaction_paths.md)): the dimer writes
`dimer_trajectory.xyz`, `dimer_eigenmodes.xyz`, and `dimer_saddle_point.xyz` instead of a
file per iteration in `dimer_structures/`, ADDF writes no `optimized_structures/ADD_<k>/`
(its points are in `ADD_<k>_trajectory.xyz`), and `spm_trajectory.xyz` holds the first
particle in order and then the second backwards (it held the two particles in turn). What
is corrected:

- The first structure is laid onto the second before the first evaluation. MultiOptPy turns
  the first evaluated structure onto the second without turning its gradient, so the first
  displacement mixes two orientations (from the second iteration on the two agree).
- `|x - x0| = 0` in the DS-AFIR weights (an image back at its start) gives weights from the
  limit instead of NaN.
- The inputs of a directory are taken in the order of their letters (`_A` before `_B`);
  MultiOptPy takes the order of `glob`, which the file system decides.
- `-gfix` (fixed Cartesian components) is applied; MultiOptPy parses it and drops it.
- `raw_bias_grad.npy` and `raw_bias_hessian.npy` (arrays of the last evaluation of
  MultiOptPy's calculator object) are not written.

### iEIP microiterations (`-mi`)

The relaxation of the images on hyperspheres about where they were keeps MultiOptPy's
constants (trust radius 0.015 between 0.01 and 0.1, the ratios 0.2 and 0.85, the weights 1
and 0.85, the limit 0.01 of the force) and its coordinates (the polar coordinates of the
displacement, with the radius kept), as a coordinate system of `coord::system` given to the
relaxation. MultiOptPy's version pairs Cartesian steps with differences of polar forces of
the opposite sign of a gradient, takes the polar gradient at the coordinates of the
structure about the origin (with the Jacobian of the inverse map), compares the change of a
step with the prediction of the next step, keeps every third coordinate within `[0, pi]` as
if it were the polar angle of an atom, and returns the structure after its last step with
the energy and gradient of the structure before it. Here the L-BFGS pairs are steps and
gradient differences in the same coordinates, the gradient is that of the coordinates, a
step is judged by its own prediction, the angles keep the ranges of the coordinate system,
and the relaxation ends with a structure that was evaluated. On the golden case
`lj_ar7_micro` (`-mi 3`) both programs converge in 28 iterations; mopt-rs makes no
relaxation step there, since the force in the hyperspherical coordinates is below 0.01 at
the radii of about 0.04 Bohr the images move per iteration.

### The other methods follow their publications

MultiOptPy's versions of these methods do not follow the publications they name; mopt-rs
implements the methods as published and keeps MultiOptPy's options and files.

- Dimer (`-use_dimer`): MultiOptPy treats the gradients of its calculators as forces (every
  sign of the method is inverted), ignores `-dimer_sep`, `-dimer_trial_angle`, and
  `-dimer_maxiter`, starts from a random axis, evaluates both end points, and translates by
  a fixed 0.1 Bohr in every iteration, so it does not settle on the saddle. Here: rotation
  from one gradient with the Fourier fit of the curvature and the extrapolated gradient
  (Heyden, Bell, Keil; Kastner, Sherwood), evaluated anew after a rotation by more than three
  trial angles, the axis towards the second structure (or a seeded random axis without
  one), the rigid-body motions taken out of the axis, the L-BFGS translation on the
  effective gradient while the curvature is negative and a step of `step` (0.1 Bohr)
  uphill along the axis while it is positive, and the end on the force thresholds with
  negative curvature. The options are used.
- Spring pair (`-use_spm`): MultiOptPy uses a Hooke spring `10 (d - l)` along the unit
  vector, momentum and an adaptive step in the drift, a random displacement of every atom
  for the second particle (the second structure is not used), half the parallel gradient in
  the climb, at least six cycles, and writes the average of the two particles to the
  current directory. Here: Algorithm 1 of Cui and Jiang (drift with the force across the
  line and the spring force `(d - l)(r2 - r1)`, climb along the line, the end when one
  particle meets the force thresholds), the second particle towards the second structure,
  with the rates measured in `reaction_paths.md`.
- Two-point SHS (`-2pshs`): MultiOptPy's radius is the Cartesian distance of the
  structures, its maps between scaled and Cartesian coordinates are not inverse to each
  other, it minimizes the ADD of the second structure measured against the energy of the
  first with steepest-descent steps biased towards the first, reads `-addf_step` and
  `-addf_num` instead of `-2pshs_step` and `-2pshs_num`, and reports the structure of the
  lowest ADD. Here: contracting spheres in the scaled normal coordinates of the first
  minimum from the radius of the second structure, the energy minimized on each sphere, the
  highest point of the path as the estimate; `-2pshs_step` is the length along the stiffest
  mode per sphere (Angstrom).
- ADDF (`-addf`): MultiOptPy maps scaled coordinates back with `l^(1/2)` instead of
  `l^(-1/2)` (its spheres are not levels of the harmonic energy), does not minimize on the
  first sphere, minimizes on the others with a few steepest-descent steps with random
  kicks, grows the radius in the square root of the energy, declares a transition state
  where the gradient along the radius turns negative after the fifth sphere, and takes the
  elimination functions from differences. Here: the ADDs of the first sphere along both
  directions of every mode, the `-addf_num` largest followed on spheres of radius `p` times
  that of `-addf_step` along the stiffest mode, iterative optimization-elimination with
  analytic `cos^3` functions, paths that run into each other merged, paths whose structure
  falls apart into fragments (bonds longer than 1.5 times the sum of the covalent radii)
  ended as dissociation channels, and a transition state where the energy along a path
  falls. The state machine lives in `workflow::explore::addf` for the reaction-network
  searches; `ieipmain -addf` runs it on the first structure.
- Growing Newton trajectory (`-gnt`): MultiOptPy stops with an error at the first growth
  without `-gnt_vec`, since it takes a new direction from the current structure alone. Here
  the direction is taken towards the final structure (as pysisyphus does) or along the
  bonds of `-gnt_vec`, the rigid-body motions are taken out of the direction and of the
  projected gradient (the corrector cannot meet the hyperplane by turning the molecule),
  and the corrector is the minimizer of `-opt` (MultiOptPy: a BFGS step within 0.02 Bohr).
- The methods that take a second structure only as a start or a direction (`-use_dimer`,
  `-use_spm`, `-gnt`, `-2pshs`) displace it by 0.05 Bohr along a seeded random direction
  (`-ieip_opt symmetry_noise=`), so that a linear or otherwise symmetric start can leave its
  symmetry. Without it, the dimer started on linear HCN rotated into the dissociation of the
  hydrogen atom and the 2PSHS path stayed linear.

### Methods that MultiOptPy does not have (`-gsm`, `-se_gsm`, `-fsm`, `-dsafir`)

New in mopt-rs (MultiOptPy 1.21.2 has none of them), from their publications:

- `-gsm`: the double-ended growing string (Peters, Heyden, Bell, Chakraborty, J. Chem.
  Phys. 2004, 120, 7877; Zimmerman, J. Chem. Phys. 2013, 138, 184102 and J. Chem. Theory
  Comput. 2013, 9, 3043): nodes added from both ends when the frontier has settled, the
  string spread evenly, the highest node climbing. Zimmerman's strings move in delocalized
  internal coordinates; here they move in Cartesian coordinates (rigid-body motions taken
  out), as in Peters et al., or in the coordinates of `-coord`, and the exact search for the
  transition state at the end is left to `optmain -order 1`.
- `-se_gsm`: the single-ended growing string (Zimmerman, J. Comput. Chem. 2015, 36, 601)
  with the driving coordinates `ADD`, `BREAK`, `ANGLE`, `TORSION` (and `VAR` for plain
  variables). The driving direction is the sum of the unit Cartesian gradients of the
  coordinates not yet made; a bond counts as formed below 1.1 and as broken beyond 2.0 times
  the sum of the covalent radii; the growth ends when they are made, when the string has
  passed over a maximum, or at its largest number of nodes; the last node is minimized and
  the string optimized as in DE-GSM.
- `-fsm`: the freezing string (Behn, Zimmerman, Bell, Head-Gordon, J. Chem. Phys. 2011,
  135, 224108): nodes from both sides in turn along the linear synchronous transit between
  the frontiers, each optimized across the path by three gradients and frozen.
- `-dsafir`: the double-sphere AFIR (Maeda et al., J. Comput. Chem. 2018, 39, 233, eqs.
  6-10). `dsafir_variant=paper` is the paper's procedure with the constants of its authors'
  public manual; the default adds the superposition of the ends before every step, a
  `delta` of 0.1 Hartree/Bohr, and the relaxation of the path the paper's default procedure
  runs, the only setting measured that reached the right saddle on the three test
  reactions. What is the paper's and what is mopt-rs's is listed with the measurements in
  `reaction_paths.md`, "DS-AFIR: the paper and the additions".
- `-lmm` and `-ieip_opt dimer_translation=plain|optimistic|extragradient` (experimental):
  the local minimax method (Li, Zhou, SIAM J. Sci. Comput. 2001, 23, 840) and
  descent-ascent steps of the dimer translation (Korpelevich 1976; optimistic steps), from
  other fields; measured and documented in `reaction_paths.md`, off by default.

### `-coord`

MultiOptPy's `ieipmain` has no `-coord`. In mopt-rs every method runs with it: iEIP and
DS-AFIR take their steps through the coordinates and keep their Cartesian distances; the
other methods move in the coordinates (`reaction_paths.md`, "Coordinates (`-coord`)").

### Model functions (`-mf`)

MultiOptPy moves both structures by the step of the first (so the second only gives its
charge and multiplicity). `ieipmain -mf NAME` runs `optmain -mf NAME` at the first structure
with the charges, multiplicities, and `-es` of the two inputs as the two states (a third
number in `-mf NAME CHARGE MULT STATE` of optmain gives the excited state of the second).

In MultiOptPy `-mf` comes before every other method. That order still holds among
MultiOptPy's methods. Together with a method that only mopt-rs has (`-dsafir`, `-gsm`,
`-se_gsm`, `-fsm`, `-lmm`), `-mf NAME` instead names the surface of that method: the search
runs on the function of the two states, and the low points of its path are written as
candidates for a crossing (`reaction_paths.md`, "A two-point method on a function of two states").
The functions that project gradients (`opt_mesx`, `opt_meci`) are refused there.

### Added

- Every method on objectives of plain variables (`-os mb2d`, `-os expr:...`; variables files
  as inputs; bias potentials and constraints are refused there), and on structures in cells
  (the second structure is taken to the periodic images next to the first).
- `-ieip_opt KEY=VALUE` with the settings of the methods, `--config`, `--dump-config`,
  `--restart` (the evaluations are taken from the journal of the run), the `STOP` file, the
  log `mopt.log` with what the calculators say, figures of the progress after every
  iteration, `<stem>_ts_guess.xyz`, and the counts of the evaluations at the end.
- Ask/tell state machines in Rust (`path::ieip`, `path::dimer`, `path::spring_pair`,
  `path::twopshs`, `path::newton_trajectory`, `workflow::explore::addf`) and Python
  (`mopt_rs.ElasticImagePair`, `Dimer`, `SpringPair`, `TwoPointShs`, `NewtonTrajectory`,
  `Addf`, the functions `ieip_search`, ..., and `IeipJob` in the manner of MultiOptPy's
  `IEIPJob`).

Tests: `crates/mopt-rs/src/path/ieip/tests.rs` and the tests of the other modules,
`crates/mopt-rs/tests/golden_ieip.rs`, `crates/mopt-rs/tests/cli_ieipmain.rs`,
`tests/test_ieipmain.py`.
