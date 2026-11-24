# Reaction paths and transition states

Paths between structures and the transition states on them: the IRC (with a page of its own,
[IRC](irc.md)), chains of images (`nebmain`), BITSS, `ieipmain`, and the automated search of
`run_autots`. Saddle points of higher order are on [High-index saddle
points](high_index_saddles.md).

## Intrinsic reaction coordinate (`ircmain`, `optmain -irc`)

`ircmain ts.xyz -os ...` follows the steepest-descent path in mass-weighted coordinates
from a transition state to the two minima it connects, and `optmain ... -irc STEP_SIZE
MAX_STEP METHOD` does the same from the optimized structure (methods `lqa`, `hpc`, `dvv`,
`euler`, `rk4`, `gs2`, `modekill`). `ircmain ... -path_analysis` hands the path to the
analysis of `freqmain -path` above. The constraints of `optmain` (`-fix`, `-pc`, `-gfix`,
`-uc`) give the path within them (`optmain -irc` keeps those of its optimization),
`-mf seam|conical|avoiding` a path on a surface that combines two states, and `-vc` the
path of the atoms and the cell of a periodic structure together (also after `optmain
-vc`). Commands,
methods, options, files, and the measurements behind the defaults are in
[irc.md](irc.md).

## Reaction path between structures (`nebmain`)

```text
nebmain path.xyz -os gfn0xtb                    # a file of several structures
nebmain start.xyz end.xyz -os gfn0xtb -p 10     # two structures, 10 images between them
nebmain folder/ -os ...                         # files folder/<name>_<number>.xyz
nebmain start.txt end.txt -os mb2d -p 15        # points of an objective of plain variables
```

A chain of images between the structures is relaxed towards a reaction path. The default
limit is 30 NEB iterations. Image spacing and redistribution follow the articles of
MultiOptPy's author (`-nd 0.5 -adpred 2`). The default model Hessian
(`-modelhess lindh2007d3`) requires no calculator evaluation.

- the structures that are read become images 0.5 Angstrom apart (`-nd`; `-p N` puts `N`
  images between every two structures first). Where two structures that follow each
  other are further apart than that, the images between them are made on the geodesic in
  scaled interatomic distances, not on the straight line (`-neb_opt path=`, below).
  `-ndb DISTANCE` puts the first chain on the Bezier curve of its images instead (the
  curve keeps the two ends and passes near the images between them, which smooths a
  rough first path such as the trajectory of a biased optimization; it also cuts corners
  that the path needs, so look at the first chain in the figures). The number of images
  is what `DISTANCE` (Angstrom) holds in the length of the chain; the images are not that
  far apart on the curve, and `-nd` is applied (before it) only when it is given;
- every image gets a model Hessian at the start (`-modelhess lindh2007d3`; another model
  by its name, `-modelhess o1numhess` for a Hessian from further gradients of every
  image as in the articles, exact ones with `-fc N` every `N` iterations), which is
  updated from iteration to iteration and kept in `tmp_hessian_<image>.npy` in the run
  directory, one in memory at a time. A chain of plain variables uses the second
  derivatives of its objective;
- in each iteration (up to the `-ns` limit) an image moves half by a step along its force and half
  by a quasi-Newton step (`-rrs`), both across the chain: the component of the gradient
  along the chain is removed. There are no springs;
- every second iteration (`-adpred`) the images are put back along the chain, the image at
  a maximum of the energy profile on the predicted maximum and the others gathered around
  it (`-neb_opt gather=`);
- the end images are minimized (`-fe 0`); `-fe 1`, `2`, `3` fix the first, the last, or
  both where they are read.

MultiOptPy's own defaults are `-ns 10 -modelhess none -nd 0 -adpred 0` (steps along the
forces alone, the images as they are read, no redistribution). Its other options that
work: `-ad N` (equal spacing every `N` iterations), `-cineb N` (images at maxima climb
after iteration `N`), `-sd N` (steepest descent from iteration `N`), `-aconv` (images that
are converged are held), `-mfc`, the calculator options, the bias potentials, and `-es`.
The constraints of `optmain` are hard constraints of every image, held at every iteration
to the rounding floor of the constraints:

- `-fix`: the atoms stay exactly where the first structure has them, in every image (the
  later structures are put there, and the images are neither centred nor turned); their
  forces are constraint forces and are left out of the forces of the chain (the test of
  the end, the extrapolation, `perp_*_gradient.csv`), and their rows and columns out of
  the Hessians of the images;
- `-pc`, `-gfix`, `-uc`: an explicit value (`-pc manual`, `-uc "... = VALUE"`) applies to
  every image; a value "at the start" is the value of each image in the first chain,
  resolved once. An image that the chain puts elsewhere (the redistributions of `-ad` and
  `-adpred`, the images that `-aneb` adds, a chain replaced by an observer) holds the
  values interpolated at its fraction of the length of the chain it came from (targets
  linearly, angle-like ones the shorter way round). The directions of the constraints are
  taken out of the gradient of every image, its Hessian is projected as `optmain`
  projects it (`P (H - C) P + s Q Q^T`), and the images are put back onto them after every
  iteration; the first chain is put onto them before it is evaluated. The projection of
  the Hessians matters for the quasi-Newton steps of the default chain: a five-atom
  Lennard-Jones cluster (`-neb_opt path=linear tolerance=1e-4`) came to rest in 110, 118,
  125, and 66 iterations with `-pc bond 1,2`, `-gfix 1,2,3`, `-pc angle 3,1,2`, and `-pc
  bond 4,5` (a bond that changes along the path), against 118, 128, 131, and 88 with the
  gradients projected and the Hessians as they were;
- the optimizations after the chain (`-neb_opt optimize=`) hold the constraints of their
  image, resolved at it (`-gfix` among them).

`-dmf` keeps `-fix` only (it moves a fitted path, whose points cannot be put back onto
constraints), `-coord` with internal coordinates none, and `-pc eigvec` is refused (the
images have no Hessian to resolve the direction at).

Bias potentials on a chain: every image evaluates its own copy of every term, so that the
terms that keep a warm start (the auxiliary variables of `-smp`, `-aerp`, `-aerpv2`) keep
one per image and the chain is the same whatever the threads. A term that changes with the
simulated time (`-nrp`, `t` in `-ubias`) or keeps state that changes its value (`-metad`,
which deposits at accepted steps; a chain has none) is refused with the reason; a term
ramped over the iterations (`it` in `-ubias`, the ramps of the constants) follows the
iteration of the chain. A term that sees the position or the orientation in space (a point
of `-vpp`, `x()` in `-ubias`) turns the alignment of the chain off (and, in a periodic
cell, the shift of the structures to one centroid). `energy_plot.csv` has the energies
with the bias potentials, which the chain is relaxed on and which choose the maxima, the
candidates for transition states, and the images of the optimizations after the chain;
`energy_plot_without_bias.csv` and the last column of the table in the log have them
without.

What is not implemented yet (the other forces such as `-dneb`, `-om`, the other
redistributions, `-ci`, `-cg`, `-lbfgs`) is refused with a message.

### The locally updated planes

`-lup` relaxes the chain by the locally updated planes (R. Elber, I. Ulitsky, J. Chem.
Phys. 1990, 92, 1510): every inner image moves in the hyperplane through it perpendicular to
the path, down the true force across the path, with no spring force and no climbing image;
the planes are taken anew in every iteration (`path::band::lup`). On the chain machinery this
is the force without springs and without climbing, with the projection of whole images; the
steps, the Hessians of the images, the constraints (every image held on them), the bias
potentials (a copy per image), ONIOM, and periodic cells are those of every chain. `-ewbneb`,
`-cineb`, `-qsm`/`-qsmv2`, and `projection=atom` are refused with `-lup` by name. MultiOptPy
declares `-lup` without implementing it.

The original method has no springs, and an image only moves in its plane, so that images
can slide along a curved path. `nebmain -lup` keeps `nebmain`'s redistribution towards the
predicted maxima every 2 iterations unless `-adpred`/`-ad` are given (`-adpred 0` turns it
off: the original method). Measured on the test paths of `nebmain` (iterations to the
tolerance; the highest image above the first, kcal/mol for HCN):

| setting | Muller-Brown (`-p 15 -fe 3`, tolerance 1e-6) | HCN -> HNC (GFN0-xTB, `-fe 3`) |
|---|---|---|
| `-lup` (redistribution towards the predicted maxima every 2 iterations) | 34 iterations | 10 iterations; 67.98 |
| `-lup -adpred 0` (no redistribution, the original method) | 26 iterations | 9 iterations; 67.35 |
| `-lup -adpred 0 -ad 5` (equal spacing every 5) | 39 iterations | 11 iterations; 67.19 |

(The saddle of HCN -> HNC is at 68.285 kcal/mol.) Without redistribution the images slide
off the top of the barrier and the highest image underestimates it by 0.9 kcal/mol; with
the redistribution towards the predicted maxima an image sits within 0.3 kcal/mol of the
saddle, for one more iteration. That redistribution is the default of `-lup`. On these paths
`-lup` with its default gives the same chain as `nebmain`'s default chain, which has the same
force (no springs, no climbing).

LUP is a relaxation of any path; it is not a part of a two-point method. The path of a
two-point method is relaxed by it (or by `nebmain`'s default chain) with the step
`method_relax` after the method ("Relaxing the path of a two-point method").

`-qsm` and `-qsmv2`, MultiOptPy's two quadratic string forms, run on the chain of mopt-rs,
which is a quadratic string already (a Hessian per image, quasi-Newton steps across the
chain, no springs). What they choose is the direction of the chain at an image:

- `-qsm`: the direction atom by atom (the same as `-neb_opt projection=atom`).
- `-qsmv2`: one direction for the whole structure, carried outwards from the highest
  image by arcs and parabolas (Ayala and Schlegel; `-neb_opt tangent=top`), in place of
  the direction towards the higher neighbour. The rule assumes one maximum of the energy
  profile; while the profile has several, the direction towards the higher neighbour is
  used (as ase_qsm does by default).

Both go with every other setting of the chain; neither is available with
`form=multioptpy` (MultiOptPy's own step rule for them is not reproduced).

Measured on HCN to HNC (9 images, GFN0-xTB, tolerance 1e-4 Hartree/Bohr): the default
direction comes to rest in 13 iterations and `-qsm` in 21, both with the highest image at
the saddle (68.28 kcal/mol). With `-qsmv2` the highest image reaches the saddle as well
(68.285), but the chain does not come to rest: the image next to an end keeps a force of
about 1e-3 Hartree/Bohr for as long as it was run (80 iterations), with and without the
redistribution. The carried tangent of an image far from the top is not the direction of
the chain there, so a part of the gradient along the chain counts as a force that the
image cannot follow. For a relaxed chain use the default; `-qsmv2` is there so that
MultiOptPy's settings run (they run a fixed number of iterations).

`-mf NAME [CHARGE MULTIPLICITY]` relaxes the chain on a surface that combines two
electronic states, with the names and the meaning of `optmain -mf` (`seam`, `avoid`,
`conical`, `opt_mesx`, `opt_meci`, ...): every image is calculated in both states (state
`-es N` and the next one, or the second state with the charge and multiplicity given), and
the chain sees the value and the gradient of the function. Bias potentials, constraints,
`-fe`, and the redistributions work as without it. For paths on or towards a crossing
seam, for example between two minimum-energy crossing points
(`nebmain a.xyz b.xyz -os gfn0xtb -es 1 -mf seam -p 8`). The Hessians of the images come
from a model Hessian or from updates (`-modelhess NAME`, `-modelhess none`); see
[migration_paths.md](migration_paths.md), "Chain relaxation", for what is not covered yet.

Structures in a periodic cell (`Lattice="..."` in the comment line of the first
structure, as for `optmain`) give a chain in that cell: from structure to structure every
atom follows its nearest periodic image, so an atom written at the other side of the cell
does not cross it; the structures are shifted to one centroid and are not turned onto
each other; the files of the chain have the lattice. Without `-vc` all structures need the
same cell. An atom that moves further than half a cell between two structures needs a
structure in between.

### A chain of the atoms and the cell (`nebmain -vc`)

With `-vc` every image has a cell of its own: the images are points of the variables of
`optmain -vc` made from the first structure (quasi-Cartesian coordinates of the atoms and
scaled lattice vectors, `-vc_weight`, Gubler et al. 2023), relaxed as a chain of plain
variables on the enthalpy `E + P V` (`-vc_pressure` in GPa, 0 without it) with the bias
potentials. Distances and tangents along the chain are Euclidean in these variables (the
solid-state NEB of Sheppard et al. 2012 in the variables of Gubler et al.). This is the
chain that `run_mapper` relaxes on a variable cell (`variable_cell` in the step settings):
the same vector, the same metric, the same values and Hessians of the images. The
structures that are read may have different cells; every atom is taken to the periodic
image nearest to it in the structure before. The files of every iteration have the lattice
of every image (`path_ITR_<k>_<name>/`, extended XYZ); `cell_volume.csv` has the volume of
every image (Angstrom^3) per iteration and the log the lattice parameters; the profile in
`energy_plot.csv` is the enthalpy. `-nd` is a distance in these variables (0.5 Angstrom of
them unless given), the first chain is on the straight line between the structures, and
where the steps use Hessians they are those of the enthalpy in the variables (the model
Hessians of `-modelhess` are of the atoms alone; `-modelhess none` gives steps along the
forces). Constraints are those of `optmain -vc` (`-fix` keeps the fractional coordinates of
an atom, `-pc`, `-gfix`, `-uc` with the functions of the cell, `-vc_fix`), held by every
image at its values of the first chain to 1e-10. `-neb_opt optimize=` runs `optmain -vc` on
the candidates; `-cineb`, `-sd`, `-qsm` (the lattice vectors are blocks of three as the
atoms), `-qsmv2`, `-lup`, `-ewbneb`, `-aneb`, `-ad`, `-adpred`, `-gqnt`, `gp=on`, `-mf`, and
the restart work as for a chain of structures. Refused, with the reason: `-dmf`, `-coord`
other than `cart`, `path=` other than `linear`, `gp_features=distances`, `-vc_afir`,
`-oniom`, plain variables, and the constraints that `optmain -vc` refuses (a Cartesian
component, `-pc rot`, a cell fixed completely by `-vc_fix`: then the cell is not variable,
and the chain without `-vc` is the one to run). The Python `relax_band(..., options=["-vc"])`
relaxes the same chain and returns the cells of the images (`cells_bohr`).

`-neb_opt path=NAME` chooses how the images between two structures that follow each
other are made (`-p N` of them, or as many as the distance of `-nd` asks for; a file of
many structures that are nearer than that stays as it is):

| `path=` | Images |
|---|---|
| `geodesic` (the default) | on the shortest path in scaled interatomic distances (Zhu, Thompson, Martinez 2019) |
| `linear` | on the straight line (MultiOptPy's `-p`; the default with `form=multioptpy` and for plain variables) |
| `lst` | linear synchronous transit (Halgren, Lipscomb 1977): every image fits the linearly interpolated distances |
| `idpp` (also `-idpp`) | the straight line relaxed on the image dependent pair potential (Smidstrup et al. 2014) |
| `sidpp` | sequential IDPP (Schmerwitz, Asgeirsson, Jonsson 2024): the images are put one after the other from both ends |
| `fbenm` | the straight line relaxed on the flat-bottom elastic network model (Koda, Saito 2024): bonds that stay keep their lengths, atoms that are not bonded do not collide, planar groups stay planar |
| `cfbenm` (also `-cfbenm`) | the same with the correlated term (Koda, Saito 2025): an atom that changes one bond for another is not far from both partners at once |

On the straight line atoms run into each other and groups that turn shrink; the others
keep the distances between the atoms sensible, and none of them calculates an energy.
What some programs call LST is `linear` here. `fbenm` and `cfbenm` need the elements
(bonds are found from covalent radii). In a periodic cell the distances
are those of the nearest images; atoms of `-fix` stay. The measurements behind the
default are in [migration_paths.md](migration_paths.md), "The first chain". From Rust:
`path::interp::images_between`.

`-ewbneb` gives the chain springs between its images, stiffer where the energy is
high (the nudged elastic band with energy-weighted springs, for comparison with the
default, which has no springs): the images are then kept apart by the springs, closer
together near the top of the barrier, and no redistribution is done unless one is asked
for (`-adpred`, `-ad`). `-neb_opt spring_upper=` and `spring_lower=` set the constants at
the highest image and at the energy of the higher end (0.1 and 0.01 Hartree/Bohr^2).
Without an image that climbs (`-cineb N`) no image sits on the saddle, as in every
elastic band. See [migration_paths.md](migration_paths.md), "Springs", for what was measured.

`-dmf` finds the path by the direct MaxFlux method (Koda and Saito, J. Chem. Theory
Comput. 2024) instead of relaxing a chain: the first chain (read and made as for a chain)
becomes a smooth curve, a B-spline, and the curve is moved until
`ln(integral exp(beta E) ds) / beta` is smallest, with `beta` 10 per eV. That integral is
smallest for a path over the lowest pass, and its top is then the transition state. In
every iteration the energies and gradients are calculated at a few points only: three
that move towards the top of the energy profile and one between each end and them (the
ends once). The log has a line and a table of the points for every iteration; the run
directory has the points of every iteration as a chain (`path_ITR_<k>_<name>/`,
`energy_plot.csv`), the estimate of the transition state of every iteration
(`<name>_dmf_highest.xyz`), and at the end the path (`<name>_dmf_path.xyz`, 41 structures
at equal shares of its length) and the estimate (`<name>_ts_guess.xyz`). `-neb_opt
optimize=saddles` optimizes the highest point with `optmain -order 1` as it does an image.
`-neb_opt dmf_points=`, `dmf_guards=`, `dmf_moves=off`, `dmf_beta=`, `dmf_segments=`,
`dmf_tolerance=`, and `dmf_step=` change the method; `-ns` is its largest number of
iterations (60 without it). The calculator options, the bias potentials, `-mf`, `-es`,
`-fix`, and periodic cells work as for a chain; `-coord`, `gp=on`, `-aneb`, `-cineb`, and
`-ewbneb` belong to chains and are refused. See [migration_paths.md](migration_paths.md), "The direct
MaxFlux method", for how it is solved and what was measured.

From Python, `mopt_rs.dmf_search(objective, chain)` runs the method on a function of plain
variables (a Python function returning the value and the gradient, or a built-in objective
such as `"mb2d"`) and returns the estimate, the path, and its energy profile;
`mopt_rs.DirectMaxFlux` is the ask/tell state machine underneath (`ask()` gives the points
to evaluate, `tell()` takes their energies and gradients), for any other way of evaluating
the points, structures with a calculator among them. The parts are there on their own as
well: `mopt_rs.DmfPath` is the path (the spline through given points or control points,
its arc length, the points at shares of the length, and the MaxFlux objective with its
gradient for given energies), and `mopt_rs.dmf.dmf_profile` the energy profile through
energies and slopes with its top. The defaults are those of the method for Hartree and
Bohr.

`-coord dic` (or `tric`, `tric-cart[:WEIGHT]`) relaxes a chain of structures in
internal coordinates: the images are vectors of the coordinates, set up once at the image
in the middle of the first chain, and distances along the chain, forces, steps, and the
Hessians of the images are in them; the calculator still gets Cartesian structures, and
the files have structures. Coordinates that cannot be set up for the chain give way to
others with a warning (`tric` and `tric-cart` to `dic` where an end is a linear molecule,
whose orientation is not defined; at last to Cartesian coordinates). Read
[migration_paths.md](migration_paths.md), "Chains in internal coordinates", first: where an angle
comes near 180 degrees on the way the coordinates break down, and the chain does not come
to rest. Not together with `-fix`, `-pc`, `-gfix`, `-uc`, `-aneb`, or plain variables.

Every iteration prints its line (the highest image, the largest force, the maxima of the
profile) and a table of the images (energy above the first image, energy, largest and
root-mean-square force, distance to the next image, and whether the image is a maximum
or minimum of the profile, climbs, or is held), on the screen and in `mopt.log` alike.
What the calculator says while the images are calculated is there too: its warnings,
`<S**2>` of unrestricted calculations, and the output of the program itself where the
engine hands it on, each with the number of its image.

`-neb_opt gp=on` puts a Gaussian-process model of the calculated points between the
relaxation and the surface (off without it). Between two iterations that are calculated,
up to `gp_between` (2) iterations are answered from the model, as long as the error of
its gradients, which is measured at every calculated iteration, is below `gp_error` (0.1)
of the largest force of the chain; a chain counts as relaxed only with calculated values.
It works with every other setting of the chain (steps, redistributions, `-mf`, `-aneb`,
a periodic cell, plain variables) and keeps little in memory (`gp_points`, 12 points per
prediction; `gp_memory`, 30 points per image). `gp_features=distances` lets the model
see a structure by its inverse interatomic distances instead of its coordinates. It is not on by default because it saved
calculations in some of the measured cases only and cost many in others; read
[migration_paths.md](migration_paths.md), "Chains with a Gaussian process", before using it.

`-neb_opt optimize=on` goes on to `optmain` when the chain is relaxed (off without it):
the image at every maximum of the profile is optimized to a transition state (`optmain
-order 1`), the two end images and the images at minima of the profile to minima
(`optimize=saddles` and `optimize=minima` do one kind only). The optimizations use what
`optmain` uses without options (`rsirfo_block_bofill` and `rsirfo_block_fsb`, its
criteria) on the surface of the chain: the calculator and its settings, charge and
multiplicity, `-es` or `-mf`, the bias potentials, and the constraints are handed on;
`optimize_settings=FILE` names a settings file of `optmain` with anything else (for
example `{"tight_convergence_criteria": true}`). An optimization starts from the Hessian
the relaxation kept for its image (`optimize_hessian=chain`, the default; for a saddle
point the curvature along the chain is replaced by that of the energy profile, which is
negative there), from the calculator's Hessian at the first point (`exact`, also used
where the chain has no Hessian to give), or from none (`none`). Each run is in a directory
next to the file of its image (in the folder of the last chain), `optimized_images.csv` in
the run directory of the chain lists them (image, kind, converged, iterations, energy,
energy above the lowest minimum, the number of imaginary frequencies, and the optimized
structure's file relative to that run directory), and the log has a line for each. The
transition states get a vibrational analysis (`optmain -freq`, one exact Hessian each;
`optimize_check=all` for the minima too, `off` for none), and an optimization to a
transition state that ended at a minimum or at a saddle point of higher order is reported
with a warning: on a flat surface the optimization from the top of the chain can travel
far and fall into a minimum (on the aldol reaction of MultiOptPy's test with GFN0-xTB,
seven of thirteen distinct "transition states" were minima). With a looser tolerance the chain is a cheap first stage: `nebmain path.xyz -os ...
-neb_opt tolerance=2e-3 optimize=on` stops the chain early and leaves the rest to
`optmain` (what that saves is measured in [migration_paths.md](migration_paths.md), "Chain
relaxation"). Chains of plain variables have no such step yet.

`-aneb N FREQ` adds `N` images on either side of every image at a maximum of the profile
after every `FREQ` iterations, so that a chain can start coarse and resolve the top of
the barrier as it goes (`nebmain path.xyz -nd 1.0 -aneb 1 10`). `-gqnt` takes one
quasi-Newton step for the whole chain instead of one per image (experimental; best with
`-adpred 0 -cineb 3`, see [migration_paths.md](migration_paths.md), "Chain relaxation").

An image whose calculation fails does not end the run: its step is taken again at half
and at a quarter of its length, then the image is put back where it was calculated before,
and only if that fails as well it is taken out of the chain (`-neb_opt failed=drop` takes
it out at once, as MultiOptPy does; `failed=stop` ends the run). The log says what was
done.

The run ends when the largest force of the images that move is below `3e-4` Hartree/Bohr
(`-neb_opt tolerance=`; `-ns` is the most iterations that are run, and a run that used
them up says how large the force still is and how to go on). Either kind of step works
alone: `-rrs 1` takes quasi-Newton steps only, `-modelhess none` steps along the forces
only (slow: use it where no Hessian of any kind is wanted). Any model Hessian of
`optmain` can be given to `-modelhess`; the iterations they take are compared in
[migration_paths.md](migration_paths.md) ("Chain relaxation").

`-neb_opt KEY=VALUE ...` has what MultiOptPy has no option for: `tolerance` (`0`: all
iterations are run, as in MultiOptPy), `diis` and `diis_start` (an extrapolation over the
last chains, off by default), `sweep` (`on`: the quasi-Newton steps are found from the
highest image down, each with the step of its neighbour uphill counted in), `global` (`on`:
one quasi-Newton step for all images at once, experimental; `global_memory` is the
megabytes of image Hessians it holds in memory, the others stay in their files), `failed`
(`retry`, `drop`, `stop`: what is done with an image whose calculation fails), `gather`
and `gather_width`, `pairs`, `projection` (`whole`, the
default: the gradient along the one direction of the whole structure is removed; `atom`:
along the direction of every atom, MultiOptPy's rule), `form` (`multioptpy`: MultiOptPy's
steps, alignment, and redistribution as they are), `figures` (`off`).

The run directory has MultiOptPy's files (`energy_plot.csv`, `path_length.csv`,
`bias_force_rms.csv`, `orthogonality.csv`, `perp_rms_gradient.csv`,
`perp_max_gradient.csv`, one line per iteration; `path_ITR_<iteration>_<name>/` with the
structures of the chain, `<name>_path.xyz`, and the images at maxima and minima of the
profile), the figures `energy_profile`, `image_forces`, `convergence` (and
`chain_on_surface` for two variables) as PNG and SVG, drawn anew after every iteration (by
default, a deliberate difference from MultiOptPy 1.21.2, which writes every figure only with
`-spng`; here `-spng` alone keeps meaning on, and `-spng off`, `"save_pict": false` in a
settings file or in `step2_settings` of `run_autots`, or `-neb_opt figures=off` writes no
figures and the CSV files as before),
`mopt.log`, `settings.json`, and the journal of the evaluations. At the end the images at
the maxima of the profile are named as candidates for transition states; a
transition-state optimization from such an image (`optmain -opt rsirfo_bofill -order 1
-fc 1`) and `ircmain` follow.

A file `STOP` in the run directory ends the run after the iteration at hand;
`nebmain --restart RUN_DIR` continues it (the evaluations done are taken from the
journal). From Rust, `cli::nebmain::run_with_observers` calls observers after every
iteration, which may do anything with the chain or end the run
(`workflow::chain::ChainObserver`).

### What the method is called

The names of chain methods are used loosely, and MultiOptPy calls all of its variants
NEB. What the default does, in the usual terms:

| Name | Images are kept apart by | Images move by |
|---|---|---|
| nudged elastic band (NEB) | springs along the chain | the force across the chain plus the spring force along it |
| locally updated planes (LUP; `-lup`) | nothing in the method (`nebmain -lup` puts them back along the chain by default) | the force across the chain |
| string method | putting them back along the chain (reparametrization) | the force across the chain |
| quadratic string method (QSM) | putting them back along the chain | steps on a quadratic model of the surface at every image (a Hessian per image, updated) |
| the default of `nebmain` | putting them back along the chain, an image onto every maximum of the profile | half a step along the force across the chain, half a quasi-Newton step across it (a Hessian per image, updated) |

So the default is a string method with quasi-Newton steps, closest to QSM; the command
keeps MultiOptPy's name. See `docs/source/migration_paths.md`, "Chain relaxation", for what was
changed against MultiOptPy and what was measured.

## Transition state between two structures (`-mf bitss`)

`optmain start.xyz -mf bitss other.xyz` searches the transition state between the basins
of two structures with the binary-image transition state search (BITSS; Avis, Panter,
Kusumaatmaja, J. Chem. Phys. 2022, 157, 124107). The two structures (same atoms in the
same order; they need not be minima) are two images that are minimized together on

`E1 + E2 + kappa_e (E1 - E2)^2 + kappa_d (d - d_i)^2`,

where `d` is their distance in Cartesian coordinates and `d_i` the separation asked for.
`d_i` is halved from stage to stage; within a stage the function is minimized with the
method of `-opt` (L-BFGS in the paper; every minimizer works, the quasi-Newton ones need
far fewer iterations than FIRE). The energy term keeps either image from being pulled
over the ridge, so the images meet at the lowest point of the ridge between them. Between
two basins with an intermediate, that is the highest transition state on the way.

- Every calculator of `-os` and every bias potential can be used; the bias potentials are
  part of the surface of each image.
- `-fix` keeps atoms where they are in both images (the two files must agree in them);
  other constraints are not available. Without fixed atoms the second structure is laid
  onto the first before the search (in a cell: its atoms are taken to the periodic images
  next to those of the first).
- With `-fc N` the Hessians of both images are computed every `N` iterations and the
  Hessian of the function is built from them. With `-modelhess` the model Hessians of the
  images start each minimization.
- The search ends when the forces at the midpoint of the images are below the thresholds
  of the run, or when the images are closer than 0.01 Bohr. If the midpoint is then not
  higher in energy than both images, the pair has slipped into one basin, and the run
  says so instead of reporting a transition state.
- `-bitss KEY=VALUE ...` changes the settings: `f` (share by which the separation shrinks
  per stage, 0.5), `alpha` (10) and `beta` (0.1) of the two coefficients, `interval`
  (iterations after which the coefficients are set again, 100), `points` (points between
  the images whose energies give the barrier estimate, 3), `min_separation` (0.01 Bohr),
  `stages` (30), `step_share` (longest step as a share of the separation asked for, 0.5;
  `none` removes the limit), `midpoint_shift` (end when the midpoint moved by less than
  this share of the reduction of a stage; the paper uses 0.1 before handing over to
  another search), `midpoint_gradient` (`false`: do not end on the forces at the midpoint).
  If the pair leaves the ridge, `f=0.25` or `step_share=0.25` keep it there at the price
  of more iterations.

Output: every iteration shows the two images in Cartesian coordinates (Angstrom), the
first under `State1` and the second under `State2`, on the screen and in `mopt.log`, with
the energies of the two images (`State1`, `State2`), the separation, and at every stage the
barrier estimate and the coefficients; `<stem>_optimized.xyz` holds the pair as one
structure of `2N` atoms (the first image, then the second).
`<stem>_ts_guess.xyz` is the midpoint of the final images, the estimate of the transition
state; `<stem>_bitss_image_1.xyz` and `_2.xyz` are the images, which lie on either side of
it along the reaction coordinate. While the search runs, every iteration appends each image
to a trajectory of its own (`<stem>_image_1_traj.xyz`, `<stem>_image_2_traj.xyz`); at the
end the two become `<stem>_traj.xyz`, the first image from start to end followed by the
second from end to start, as MultiOptPy writes it, and the two files are removed. The
estimate is
as good as the last separation allows: refine it with `optmain <stem>_ts_guess.xyz -order
1 ...` and check it with `-freq`.

In Python: `mopt_rs.bitss_search(function, first, second)` for a function of plain
variables (or `"mb2d"`, `"expr:..."`), `mopt_rs.Bitss` for the ask/tell state machine
(points to evaluate out, energies and gradients in), `mopt_rs.bitss.bitss_function` for
the function alone, and `mopt_rs.run_optmain([..., "-mf", "bitss", "other.xyz"])` for the
command. From Rust: `path::bitss::{Bitss, BitssSettings, penalized, penalized_hessian,
Coefficients}`.

## Transition states from two structures or one (`ieipmain`)

`ieipmain DIR -os NAME` searches a transition state between the structures `DIR/*_A.xyz`
and `DIR/*_B.xyz` (MultiOptPy's input; one or two structure files work too, and variables
files for objectives of plain variables such as `-os mb2d`). The method:

| option | method | inputs | estimate of the transition state |
|---|---|---|---|
| (none) | improved elastic image pair (iEIP; Liu, Qi, Lei 2023) | two minima | midpoint of the images when they meet |
| `-2pshs` | two-point scaled hypersphere search (Maeda, Ohno 2005) | two minima | highest point of the path into the first |
| `-gnt` | growing Newton trajectory (Quapp 2005) | a minimum and a direction (the second structure, or `-gnt_vec` bonds) | highest point when the trajectory has passed it |
| `-addf` | anharmonic downward distortion following (Ohno, Maeda 2004) | one minimum | the highest point of every path that passes one |
| `-use_dimer` | dimer method (Henkelman, Jonsson 1999; Kastner, Sherwood 2008) | a structure and an axis (towards the second structure) | the converged dimer |
| `-use_spm` | spring pair method (Cui, Jiang 2024) | a minimum and a direction (towards the second structure) | the particle at the saddle |
| `-mf NAME` | `optmain -mf NAME` at the first structure | two states (charges, multiplicities, `-es` of the two inputs) | the optimized point |
| `-gsm` (mopt-rs) | double-ended growing string (Peters, Heyden, Bell, Chakraborty 2004; Zimmerman 2013) | two minima | the climbing node |
| `-se_gsm 'ADD i j' ...` (mopt-rs) | single-ended growing string along driving coordinates (Zimmerman 2015) | one minimum | the climbing node |
| `-fsm` (mopt-rs) | freezing string (Behn, Zimmerman, Bell, Head-Gordon 2011) | two minima | the highest frozen node |
| `-dsafir` (mopt-rs) | double-sphere AFIR (Maeda et al. 2018) | two structures | the highest point of the path |
| `-lmm` (mopt-rs, experimental) | local minimax method (Li, Zhou 2001) | a minimum and a direction (towards the second structure) | the peak of the last ray |

The methods marked mopt-rs are not in MultiOptPy 1.21.2. Their settings are the keys of the
registry of two-point methods (below) with the prefixes `gsm_` (both growing strings),
`fsm_`, `dsafir_`, and `lmm_` in `-ieip_opt` (`-ieip_opt gsm_nodes=11 dsafir_variant=paper`).

Every calculator of `-os`, the bias potentials, the constraints (`-fix`, `-pc`, `-gfix`,
`-uc`), `-calc_opt`, `--config`, `--dump-config`, `--restart`, and the `STOP` file work as
in the other commands. The constraints are hard: every point of a search holds the values
of the first structure (explicit values as given; the second structure is put onto them,
and the log says so), the gradient and the Hessian lose their directions (`P (H - C) P +
s Q Q^T`, so that the methods that follow modes see them as stiff), and every method holds
every kind (`-fix`, `-pc` bond, fbond, angle, dihedral, `-gfix`, `-uc`) at every point it
evaluates or writes (to the floor of the restore, at most 1e-10), each in the way its own
equations allow on the manifold `M` of the constraints (new in mopt-rs; the derivations are
in the module documentation, `path::manifold` and the modules of the methods):

- iEIP (with the estimate between its images), DS-AFIR, the dimer method, and the spring
  pair method put their points back onto `M` after every step;
- the growing and the freezing string (GSM, SE-GSM, FSM) take tangents, steps, and the
  growth direction into the tangent space `T_x M` and put every node that moved, was
  added, or was spread back onto `M`; a driving coordinate of SE-GSM that the constraints
  hold is refused by name;
- LMM follows the curves `R(A + t v)` on `M` (`R` the restore, `v` in `T_A M`);
- GNT takes its direction into `T_x M`, and its corrector minimizes on `M` and its plane
  together (both held exactly by the optimizer);
- 2PSHS and ADDF use the normal coordinates of the projected Hessian of the Lagrangian at
  the constrained minimum (the normals left out), and a point of a sphere is the point of
  `M` with those coordinates (`M` as a graph over its tangent space at the minimum: the
  radius and the constraints hold together); the ADD is measured on `M`.

The optimizations after the search (`-ieip_opt optimize=on`) get the same constraints,
resolved at the first structure. Bias potentials that change with the time or keep state that changes their
value (`-nrp`, `-metad`) are refused, since a search is never told which evaluation is
accepted; a term with a warm start is copied for every evaluation, so the values do not
depend on the order or the threads; a term that sees the position in space keeps the
structures from being laid onto each other. The estimate is written as `<stem>_ts_guess.xyz` (`_ts_guess_<k>.xyz` for several
ADDF paths); refine it with `optmain <stem>_ts_guess.xyz -order 1 -fc 1` and check it with
`freqmain`. The run directory has MultiOptPy's files (iEIP: `approx_TS.txt`, `approx_EQ.txt`,
`local_min_grad.txt`, `<stem>_traj.xyz`, `plot_*.png`; ADDF: `add_energy_list.csv`,
`ADD_<k>_trajectory.xyz`, `TSpoints.xyz`; 2PSHS: `direction_info.csv`,
`ADD_0_distances.csv`, `ADD_0_trajectory.xyz`; dimer: `dimer_trajectory.xyz`), a table and
a figure of the progress redrawn after every iteration (`ieip_energies.csv`,
`dimer_results.csv`, `spm_cycles.csv`, `nt_points.csv`, `addf_paths.csv`; `-ieip_opt
figures=off` leaves the figures out), and `mopt.log` with what the calculators say and the
counts of the evaluations at the end.

The structures of every iteration are shown in Cartesian coordinates (Angstrom) on the
screen and in `mopt.log` and appended to trajectory files, so that they are there while a
run goes on and after it was interrupted; no directory per iteration is written
(MultiOptPy: `samples_<stem>_<k>/`, `dimer_structures/`, `optimized_structures/ADD_<k>/`).
A method with two structures keeps one trajectory per structure while it runs and leaves one
file at its end:

| method | shown every iteration | files while it runs | at the end |
|---|---|---|---|
| iEIP | `Image 1`, `Image 2` | `<stem A>_image_1_traj.xyz`, `<stem B>_image_2_traj.xyz` | `<stem A>_traj.xyz`: the first image in order, then the second backwards, without the two inputs (MultiOptPy's layout); the two files removed |
| `-use_spm` | `Particle 1`, `Particle 2` (every cycle) | `spm_particle_1_traj.xyz`, `spm_particle_2_traj.xyz` | `spm_trajectory.xyz`: the first particle in order, then the second backwards; the two files removed |
| `-dsafir` | the end that moved (both at the start) | `dsafir_side_1_traj.xyz`, `dsafir_side_2_traj.xyz` | `dsafir_path.xyz` (and `dsafir_relaxed_path.xyz`); the two files removed |
| `-fsm` | every frozen node | `fsm_side_1_traj.xyz`, `fsm_side_2_traj.xyz` (reactant and product side) | `fsm_path.xyz`; the two files removed |
| `-gsm`, `-se_gsm` | the nodes at the frontiers while the string grows, then its highest node | `gsm_side_1_traj.xyz`, `gsm_side_2_traj.xyz` while it grows; `gsm_path.xyz` rewritten in every iteration of the optimization | `gsm_path.xyz`; the two files removed |
| `-use_dimer` | the centre | `dimer_trajectory.xyz`, `dimer_eigenmodes.xyz` (appended) | the same, and `dimer_saddle_point.xyz` when it converged |
| `-2pshs` | the point on the sphere | `ADD_0_trajectory.xyz` (rewritten) | the same |
| `-addf` | every point of a path | `ADD_<k>_trajectory.xyz` (rewritten) | the same, and `TSpoints.xyz` |
| `-gnt` | every point of the trajectory | `newton_trajectory.xyz` (rewritten) | the same |
| `-lmm` | every peak | `lmm_peaks_path.xyz` (rewritten) | the same |

Plain variables are shown as their values and written to the tables of the methods.

`-ieip_opt KEY=VALUE ...` holds the settings of the methods (`docs/source/cli_reference.md`
lists them); among them `criteria=` (force thresholds of the dimer and the spring pair),
`step=` (largest step of the dimer and the spring pair, Bohr), `spm_drift=`, `spm_climb=`,
`sphere_iterations=`, and `symmetry_noise=` (a seeded random displacement of 0.05 Bohr of
the second structure where it only gives a start or a direction, so that a linear molecule
can leave its line; 0 switches it off).

In Python: `mopt_rs.ieip_search`, `two_point_shs`, `newton_trajectory`, `addf_search`,
`dimer_search`, `spring_pair_search` for functions of plain variables (or `"mb2d"`); the
ask/tell classes `mopt_rs.ElasticImagePair`, `TwoPointShs`, `NewtonTrajectory`, `Addf`,
`Dimer`, `SpringPair` for any other evaluation; `mopt_rs.IeipJob(folder)` with
`set_option`, `run`, and `get_results` for the command, in the manner of MultiOptPy's
`IEIPJob`. From Rust: `path::{ieip, twopshs, newton_trajectory, dimer, spring_pair}`,
`workflow::explore::addf`, and the drivers in `workflow::pair_search`.

### The path of a search (`<stem>_path.xyz`)

Every search between structures writes the points it went through as one file,
`<stem>_path.xyz` in the run directory (`<stem>`: the name of the first input), and the log
names it (`Path file path: ...`). The points are ordered from the first structure to the
second and have the two input structures as their ends, whatever the method does inside:

| Method | Where the points come from |
|---|---|
| `-gsm`, `-fsm`, `-dsafir` | the path of the string (also `<method>_path.xyz`) |
| `-lmm` | its estimate of the transition state between the two structures (the method moves a direction, not a chain of structures) |
| iEIP (no option) | the histories of the two images (`<stem>_traj.xyz`) |
| `-2pshs` | the walk on the shrinking spheres (`ADD_0_trajectory.xyz`), turned round |
| `-gnt` | the points of the Newton trajectory (`newton_trajectory.xyz`) |
| `-use_dimer` | the trajectory of the centre of the dimer, then the second structure |
| `-use_spm` | the forward history of the particle selected as the saddle estimate, then the second input. Both particles start near the first input. Their diagnostic histories are in `spm_trajectory.xyz`. |

BITSS and iEIP retain the inward walks of both images in computation order on the
reactant side and reverse order on the product side. This also applies to mapper and
plain-variable searches when intermediate files are disabled. GSM, FSM and DS-AFIR
supply their ordered strings. The dimer, spring pair and directed Newton trajectory use
the second input as a direction or target. A final segment anchored by that input can
still require relaxation. The presence of both inputs in the initial band does not
verify a chemical connection. Check the saddle order and both IRC endpoints.

The files of the methods stay as they are. The path is what a relaxation takes as its
input (`nebmain <stem>_path.xyz ...`), and what `run_autots` takes when a search is its
step 1. `-addf` goes out from one structure in several directions and writes no such file.
The points are those the method evaluated or moved through; they are not a relaxed path,
and their energies are in the file only where the method returns them with the points.
In a periodic cell every frame of the path, and of the trajectories of the methods, has
the lattice in its second line (extended XYZ), so that what reads the file reads
structures in that cell.

### The searches in the variables of the atoms and the cell (`ieipmain -vc`)

With `-vc` the points of every method are points of the variables of `optmain -vc` made
from the first structure (quasi-Cartesian coordinates of the atoms and scaled lattice
vectors), on the enthalpy `E + P V` (`-vc_pressure` in GPa) with the bias potentials: the
vector and the metric of the two-point searches of `run_mapper` on a variable cell. The two
structures may have different cells (every atom of the second is taken to the periodic
image nearest to it in the first). The estimates, the path, and the trajectories of the
methods have the structures of the points with their own cells. `-fix` holds the fractional
coordinates of an atom in every method but ADDF and 2PSHS, whose scaled coordinates leave
fixed atoms out only for structures in Cartesian coordinates (refused there); `-pc`,
`-gfix`, `-uc`, and `-vc_fix` are held at the values of the first structure by the methods
made of steps (iEIP, the dimer, the spring pair, DS-AFIR) and refused by the others, which
hold constraints through a manifold of structures in one cell. `-ieip_opt optimize=on` runs
`optmain -vc` from the estimates; `-mf NAME` alone runs `optmain -vc -mf`. Refused, with the
reason: `-vc_afir`, `-oniom`, `-coord`, `-mf` together with a method of mopt-rs, two
structures in different states, plain variables. On the hop of a vacancy in copper (31
atoms, EAM; `tests/cli_ieipmain_variable_cell.rs`, short runs) iEIP, the dimer, the spring
pair, GSM, FSM, DS-AFIR, and LMM give estimates whose cells differ from that of the first
structure (by up to 35 Bohr^3 of 2527); 2PSHS, GNT, and ADDF run within the few spheres or
steps of the test without an estimate. The energy written with an estimate is that of the
surface at its structure to 1e-12 Hartree, but for the dimer (that of its last evaluated
centre, 6e-6 Hartree off) and GSM after four iterations (4e-3 Hartree off).

### The registry of two-point methods

`workflow::pair_search::PairMethodRegistry` holds every method that looks for a
transition state from two structures (or from one), so that a workflow (the automated
transition-state search, the mapper) picks one by name with `KEY=VALUE` settings. A method
takes a `PairInput` (the end points, or one point with driving coordinates, bonds, or a
direction), the surface (`SearchSurface`: structures on a calculator of the calculator
registry, structures in the coordinates of `-coord`, or an objective of plain variables),
and `PairSettings`; it returns a `PairResult` with the estimates of transition states
ranked by energy (the lowest first), the path where it makes one, the end points, and the
counts of gradients, Hessians, and energies. Unknown keys are refused before anything is
calculated. In Python: `mopt_rs.pair_methods()` lists them, and
`mopt_rs.run_pair_method(name, "mb2d", points, settings={...})` runs one on a built-in
objective.

| name | ends | keys (defaults; lengths in Angstrom for structures where marked) |
|---|---|---|
| `ieip` | two | `iterations` (999), `beta` (1.0), `micro` (0), `micro_tolerance` (0.01) |
| `gsm` | two | `nodes` (9), `add_tolerance` (5e-3), `node_steps` (5), `climb_tolerance` (2e-3), `iterations` (300), `max_step` (0.1 Bohr), `first_scale` (1.0), `interpolation` (linear, lst), `criteria` (normal) |
| `se_gsm` | one | the keys of `gsm` (`nodes` 20), `driving` (`ADD i j; BREAK i j; ANGLE i j k deg; TORSION i j k l deg; VAR k value`), `drive_step` (0.16 Angstrom), `drop` (2e-3 Hartree), `final_steps` (20) |
| `fsm` | two | `nodes` (12), `gradients` (3), `max_component` (0.3 Angstrom), `first_scale` (1.0), `interpolation` (lst; plain variables: linear), `images` (10) |
| `dsafir` | two | `variant` (`default` or `paper`), `force` (0.1; paper 0.03 Hartree/Bohr), `step` (0.1 Angstrom), `separation` (0.12 Angstrom), `steps` (2000), `align` (on; paper off), `alternate` (off), `quasi_newton` (off), `relax` (0: steps of the moving end across its drive), `relax_tolerance` (1e-3); `lup=N` and its `lup_*` keys name the relaxation of the path after the method (below) |
| `lmm` (experimental) | one or two | `iterations` (200), `first_step` (0.1), `sigma` (1e-4), `rho` (0.2), `trials` (8), `bb` (on), `first_distance` (half way to the second point), `ray_step` (0.15 Angstrom), `ray_tolerance` (1e-4), `ray_evaluations` (30), `criteria`, `noise`, `seed` |
| `2pshs` | two | `step` (0.05 Angstrom), `spheres` (300), `sphere_iterations` (30), `sphere_gradient` (1e-4), `noise` (0.05 Bohr), `seed` (1) |
| `gnt` | one or two | `step` (0.5 Bohr), `tolerance` (1.7e-3), `corrector` (25), `iterations` (999), `bonds` (one-ended: `i-j,k-l`), `noise`, `seed` |
| `dimer` | one or two | `separation` (1e-4 Bohr), `trial_angle` (pi/32), `iterations` (1000), `rotations` (8), `rotation_angle` (1 degree), `step` (0.1 Bohr), `criteria`, `noise`, `seed`, `translation` (lbfgs; experimental: plain, optimistic, extragradient), `rate_across`, `rate_along` (0.5) |
| `spm` | one or two | `length` (0.1), `offset` (0.1), `drift` (0.5), `spring` (1), `climb` (5), `tolerance` (2e-3), `drift_steps` (200), `cycles` (999), `step` (0.1), `criteria`, `noise`, `seed` |
| `addf` | one | `initial` (0.03 Angstrom), `step` (0.1 Angstrom), `spheres` (300), `paths` (5; 0: all), `angle` (0.01 rad), `gradient` (1e-3), `sphere_iterations`, `sphere_gradient` |
| `neb` | two | `images` (10), `iterations` (30), `path` (nebmain's first path), `options` (further `nebmain` options, as on its command line), `settings` (nebmain's settings as a JSON object, as in its settings file; the other keys go over them); nebmain's defaults, the Hessian-based steps from the model Hessian `lindh2007d3` included (`-modelhess none` in `options`: steps along the forces alone) |
| `dmf` | two | `images` (1), `iterations` (60), `path`, `options` (`nebmain -dmf` options) |
| `bitss` | two | `f`, `alpha`, `beta`, `interval`, `points`, `min_separation`, `stages`, `step_share`, `midpoint_shift`, `midpoint_gradient` (those of `optmain -bitss`) |

The one-ended methods (`addf`, `se_gsm`, `gnt` with `bonds`, and `dimer` and `spm` with
one point) start from one minimum, as the mapper needs. Every name runs through the
registry on the Muller-Brown surface in `crates/mopt-rs/tests/pair_registry.rs`.

### Relaxing the path of a two-point method

Any method that makes a path between the two structures can have it relaxed afterwards by
a chain of `nebmain` (`workflow::pair_search::relax_path`), the same vocabulary as the
mapper's relaxation of the AFIR path: `method_relax` (`lup`: the locally updated planes,
`nebmain -lup`; `neb`: `nebmain`'s default chain; `off`, the default), with
`method_relax_iterations` (5), `method_relax_nodes` (0: by the spacing),
`method_relax_spacing` (0.5 Angstrom), `method_relax_max_step` (the longest quasi-Newton step
of an image, Bohr; the chain's own without it), and `method_relax_options` (further options
of `nebmain`). Images are put evenly along the path of the method, its ends fixed, and
relaxed with `nebmain`'s settings; the maxima of the relaxed chain replace the estimates of
the method, and the optimizations after the search (`-ieip_opt optimize=on`) start from
them. In the registry these are keys of every method; in `ieipmain` they are `-ieip_opt`
keys of their own (`-ieip_opt method_relax=lup method_relax_iterations=20`). Methods
without a path between the structures (the dimer, ADDF, GNT) refuse it by name, and two
structures in different states refuse it (the path has no single energy profile). Hard
constraints hold at every image (restored after every iteration).

DS-AFIR's keys of earlier versions, `lup=N` with `lup_nodes`, `lup_spacing`,
`lup_max_step` (`-ieip_opt dsafir_lup=5`), are this step with the LUP chain (`lup_max_step`
0.1 Angstrom by default, as before); the log says so in one line, and giving both is
refused. DS-AFIR itself ends with its own path and its highest point.

### Coordinates (`-coord`)

`-coord NAME` (any system of the coordinate registry, as in `optmain`) changes what the
searches move in. iEIP and DS-AFIR keep their distances, attractions, and forces in
Cartesian coordinates, as their papers define them, and take only their steps through the
coordinates (the step `dx` from `x` becomes the structure with the coordinates
`q(x) + B(x) dx`); every other method moves in the coordinates themselves: its points are
the coordinates of structures relative to the midpoint of the inputs, so its steps,
tangents, distances, and Hessians are in them (Zimmerman's growing string lives in
delocalized internal coordinates in the same way). The coordinates are set up at the
midpoint of the inputs. Where they cannot be set up, or break down during the run (a
back-transformation that fails, or the system asking to be set up anew), they are set up
anew at the structure reached and the search starts again from the inputs; where they break
down again, the next system takes over (`tric` -> `tric-cart` -> `dic` -> Cartesian; `dic`
-> Cartesian), with a warning each time; the log ends with a line `Coordinates: ...` on what
happened. Refused, with exit code 2: `-fix` and `-gfix`, periodic cells, plain variables,
`-se_gsm` and `-gnt_vec` (their coordinates are Cartesian here), and redundant systems
(`ric`, `tric-p`) for the methods that move in the coordinates.

Measured (GFN0-xTB, defaults; gradients, then the barrier of the saddle `optmain -order 1`
reaches from the estimate, a check mark where it is the right one, 68.285 or 66.983
kcal/mol; "fell back": the coordinates were set up anew, broke down again, and the next
ones took over):

| method | HCN, `tric-cart` | CH3NC, `tric-cart` | `dic` |
|---|---|---|---|
| iEIP | 26, right | 34, no saddle | fell back to Cartesian on both (set up anew, then Cartesian); the results are then those of Cartesian coordinates |
| `-gsm` | 519, right | fell back to `dic`, then Cartesian; 2229, 68.906 | the same |
| `-use_dimer` | 2134 (used up), right | 2745 (used up), no saddle | the same |
| `-2pshs` | 302, right | fell back to Cartesian; 500, right | the same |
| `-fsm` | 69, right | fell back to Cartesian; 26, right | the same |
| `-gnt` | 73, right | fell back to Cartesian; 205, no saddle | the same |
| `-addf` | 151, right (4 of 5 estimates) | 944 (spheres used up), no saddle | the same |
| `-use_spm` | 2200 (used up), right | 2280 (used up), no saddle | the same |
| `-dsafir` | 2002, right | 1001, right | the same |

`dic`, set up at the midpoint of the inputs, broke down on both reactions before the
first steps (set up anew at the structure reached, it broke down again), and `tric-cart`
broke down twice on CH3NC for four of the methods that move in it; the fallbacks kept every
run going (before them, such runs ended with an error). In Cartesian coordinates
the same methods need 26 (iEIP), 295 (GSM), 117 (dimer), 130 (2PSHS), 27 (FSM), 116 (GNT),
102 (ADDF), 256 (SPM) gradients on HCN: the internal coordinates did not save gradients on
these small molecules.

### DS-AFIR: the paper and the additions

From the paper (Maeda et al., J. Comput. Chem. 2018, 39, 233, eqs. 6-10): the function,
`X`, `u`, `Y`, `Z`; the lower end moves; the reference of an end is the latest minimum of
the energy along its path, and the second term is left out while an end is at its
reference; the ends meet below 0.12 Angstrom; the highest point is the estimate; the path is
integrated by the local quadratic approximation ("the steepest descent path is integrated by
the LQA method"), in coordinates that are not mass-weighted. From the public manual of the
authors' program (read only): `delta` 0.03 (the paper gives values for the Muller-Brown
surface only), steps of 0.1 Angstrom, the second structure turned onto the first at the
start, and, in its default procedure, five relaxations of points 0.5 Angstrom apart after
the ends met. The manual names an exact Hessian every 10 steps for its IRC; for DS-AFIR it
does not say how the Hessian of the LQA steps is kept, nor whether its mixed internal and
Cartesian coordinates apply to the walk.

What mopt-rs does (`path::dsafir`): each step of an end follows the steepest-descent path of
the quadratic model of the AFIR function for 0.1 Angstrom (`integrator=lqa`), with the
Hessian of the energy of that end exact at its first point and at every tenth after it
(`hessian_every=10`; 0: at the first point only) and updated in between by Bofill's formula
from the gradients of the energy. `-ieip_opt dsafir_variant=paper` (registry:
`variant=paper`) is that without anything else (the paper's `DS = Single`; the relaxation
of the path of the default procedure is the step after the method, "Relaxing the path of a
two-point method"). The distance is the plain Cartesian distance of the points;
rigid-body motions are taken out of the gradient and of `u`.

Additions of mopt-rs, each a key of its own: `align` (the other end turned onto the moving
one before every step; on by default), the seeded displacement of the second structure by
`symmetry_noise` (0.05 Bohr, `ieipmain`), `attraction_only` (`Y = 1`: the second term of the
function left out), `afir_curvature` (the second derivatives of the two distance terms in
the Hessian of the step), `alternate` (the ends move in turn), and, from before the LQA
steps, `integrator=euler` (steps of the fixed length straight down the function),
`quasi_newton`, and `relax=N`. The relaxation of the paper's default procedure is not part
of DS-AFIR here: it is the LUP relaxation after the method (`method_relax=lup`, or
`lup=N`, which names it).

Measured with GFN0-xTB on HCN -> HNC, H2CO -> trans-HCOH, CH3NC -> CH3CN, and the [1,3]
shift of bromine in allyl bromide (both ends optimized): whether the ends met within 2000
steps, the gradients and exact Hessians of the search, the highest energy either end
reached above the first structure (kcal/mol), and the barrier of the saddle that `optmain
-order 1` without exact Hessians reaches from the estimate. The saddles the other methods
of this page find: 68.285, 87.293, 66.983, and 20.006 kcal/mol. Unless said otherwise:
`delta` 0.03, steps of 0.1 Angstrom, `align`, `symmetry_noise` 0.05 with seed 1, no
relaxation of the path.

| setting | HCN | H2CO | CH3NC | allyl bromide |
|---|---|---|---|---|
| LQA (the default) | met, 27 (4); 68.3; **68.285** | met, 46 (6); 89.6; **87.293** | met, 212 (22); 99.8; **66.982** | met, 57 (7); 40.7; 20.150 |
| LQA, followed by the LUP relaxation (`lup=5`, the chain of `nebmain -lup`) | met, 72 (4); 68.3; **68.285** | met, 96 (6); 89.6; 20.359 | see below | see below |
| LQA, `attraction_only` | met, 30 (4); 68.3; **68.285** | met, 18 (2); 88.8; **87.293** | met, 41 (5); 75.5; **66.983** | met, 42 (5); 36.4; 20.282 |
| LQA, `hessian_every=0` | met, 27 (2); 68.2; **68.285** | met, 55 (2); 90.0; **87.293** | met, 374 (2); 78.0; no saddle | met, 57 (2); 42.3; 20.167 |
| LQA, `afir_curvature` | met, 27 (4); 68.2; **68.285** | met, 24 (4); 90.4; **87.293** | met, 115 (13); 82.3; **66.982** | met, 57 (6); 42.6; 20.278 |
| LQA, `symmetry_noise=0` | met, 46 (6); 69.5; **68.285** | met, 85 (10); 91.8; 89.551 | met, 43 (5); 72.4; **66.983** | met, 55 (6); 39.8; 20.068 |
| LQA, `force=0.1` | met, 26 (4); 69.2; **68.285** | met, 20 (3); 91.3; **87.293** | met, 39 (4); 96.8; 68.907 | met, 41 (5); 44.4; 20.215 |
| `variant=paper` (no `align`), `symmetry_noise=0` | no, 2002 (201); 212.8; no saddle | no, 2002 (201); 199.8; a minimum | no, 2002 (201); 167.1; 151.1 | no, 2002 (201); 91.5; 60.8 |
| `variant=paper` (no `align`) | met, 387 (40); 87.8; **68.285** | met, 24 (4); 89.6; **87.293** | no, 2002 (201); 160.6; 70.130 | no, 2002 (202); 66.4; 20.066 |
| `integrator=euler` | met, 483; 97.0; **68.285** | met, 197; 94.8; **87.293** | no, 2002; 124.6; no saddle | no, 2002; 30.8; 1.658 |
| `integrator=euler`, `force=0.1`, `symmetry_noise=0`, followed by the LUP relaxation (`lup=5`; the default before the LQA steps; measured with DS-AFIR's own relaxation of earlier versions) | no, 2002; 1545; **68.285** | met, 36; 89.1; **87.293** | met, 1250; 191.6; **66.983** | met, 79; 46.3; 20.279 |

(Each cell: met or not, gradients (exact Hessians), the highest energy of an end, the barrier
of the saddle reached. "No saddle": the search of order 1 did not converge within 1000
gradients; 89.551 is the
planar second-order saddle of formaldehyde, 68.907 another saddle between CH3NC and CH3CN,
1.658 the rotation of the CH2Br group. The saddle of allyl bromide is flat, with the bromine
3.8 Angstrom from the carbons: without exact Hessians the search of order 1 meets its
thresholds on the ridge 0.03 to 0.28 kcal/mol above it, with an imaginary frequency of 1 to
23 cm^-1; with `-fc 1` it reaches 20.006.)

With four more seeds of the displacement (five in all), the default and `attraction_only`
met on every one of the 20 runs each:

| setting | HCN | H2CO | CH3NC | allyl bromide |
|---|---|---|---|---|
| LQA (the default): gradients; right saddle | 26 to 27; 5 of 5 | 23 to 48; 5 of 5 | 48 to 212; 3 of 5 (68.906 twice) | 55 to 57; the flat ridge |
| LQA, `attraction_only`: gradients; right saddle | 29 to 31; 5 of 5 | 18; 3 of 5 (the planar 89.550 twice) | 41 to 43; 5 of 5 | 42; the flat ridge |

What the numbers say:

- **The integrator decides.** With steps of a fixed length straight down the function the
  moving end overshoots the stiff modes across its drive, zigzags, and is pushed far up (to
  1545 kcal/mol on HCN with the former default, whose estimate reached the right saddle
  only because the path it had walked passed near it). The LQA step relaxes those modes
  within the step: the ends meet in 18 to 212 gradients and 2 to 22 exact Hessians, and on
  HCN and formaldehyde the ends stay within 2 kcal/mol of the barrier.
- **The superposition of the start alone is not enough.** Turned onto the first structure
  once, HNC lies reversed on HCN and the straight line between the ends runs through atoms;
  the paper's variant met on two of the four reactions and on none without the
  displacement. Laying the other end onto the moving one before every step (`align`) is what
  the default adds. The authors' program may get the same from its coordinates; the
  sources do not say.
- **Symmetric ends need the displacement.** Between two planar structures no gradient
  leads out of the plane: without `symmetry_noise` formaldehyde gives the planar
  second-order saddle.
- **The second term of the function costs gradients where the molecule has soft modes.** It
  pushes an end away from its reference along the softest mode (the methyl rotation of
  CH3NC: 48 to 212 gradients, ends up to 104 kcal/mol, another saddle in two of five runs).
  `attraction_only` took 41 to 43 gradients there with the right saddle every time, but
  gave the planar saddle of formaldehyde in two of five runs. The two settings found the
  right saddle equally often (13 of 15 on the three reactions with a clear saddle); the
  default stays with the paper's function, and `attraction_only` is the cheaper setting to
  try next.
- **`delta` 0.03.** With 0.1 the ends meet a little sooner but cut the corner on CH3NC
  (another saddle).
- **The relaxation of the path does not pay here.** Re-measured with the relaxation as the
  step after the method (five iterations of `nebmain -lup` on images 0.5 Angstrom apart, the
  ends fixed; the default row reproduces 27 (4) and 46 (6) gradients on HCN and
  formaldehyde): it more than doubled the gradients (72 and 96), kept the saddle of HCN, and
  on formaldehyde moved the estimate to where the search of order 1 reached another saddle
  (20.359 kcal/mol) instead of 87.293. The ends of CH3NC and allyl bromide of the table were
  not at hand; on other optimized ends of the same reactions (DS-AFIR alone: 138 and 1192
  gradients, saddles 72.448 and 55.936) the relaxation took 283 and 2432 gradients and gave
  an estimate from which the search of order 1 did not converge (CH3NC) and one that reached
  the rotation of the CH2Br group (1.658). (With DS-AFIR's own relaxation of earlier
  versions: 64, 88, 439, and 114 gradients, the saddles 68.285 and 87.293, no saddle on
  CH3NC, 20.256 on allyl bromide.) `method_relax=lup` with `method_relax_iterations=5` is
  the same step (on these runs the same gradients; `lup_max_step` keeps 0.1 Angstrom, the
  generic step the chain's own longest step unless `method_relax_max_step` is given).
- **An updated Hessian alone is enough for three of the four**; CH3NC needed the exact ones
  in between (`hessian_every=0`: 374 gradients and no saddle).

Refine every estimate with `optmain -order 1` and check it with `freqmain` and `ircmain`:
five seeds and four reactions are a small sample, and two of the four have more than one
saddle between their ends.

### DS-AFIR between two states: an estimate of their crossing

When the two structures are in different states (another charge, multiplicity, or
electronic state: `-elec`, `-spin`, and `-es` take one value per structure, as in `ieipmain
pair -dsafir -es 0 1` for a structure in the ground state and one in the first excited
state), each end of DS-AFIR moves on the surface of its own state: its energy, its gradient,
and the Hessian of its LQA steps are those of that state. The lower end moves, as always.
Where the ends meet, the mean of the two ends (the second laid onto the first) is written
as `<stem>_crossing_guess.xyz`, the log gives the energies of the two states there, and it
names the search for the crossing from it: `optmain FILE -elec C1 -spin M1 -es S1 -mf
opt_meci C2 M2 S2` for two states of one charge and multiplicity (a conical intersection),
`-mf opt_mesx` otherwise. The relaxation of the path after the method (`method_relax`,
`dsafir_lup`) needs one surface and is refused. iEIP and 2PSHS use the states of both structures as MultiOptPy does; the other
methods run in the state of the first structure, and the log says so.

This use of DS-AFIR is an addition of mopt-rs: the text of the paper was not at hand for
what its authors do for crossings. A search on a model function of the two states (`-mf
seam` and the like) is another way; see "A two-point method on a function of two states"
below.

`-ieip_opt optimize=on` goes on by itself (off without it, for every method of `ieipmain`):

- an estimate of a transition state is optimized with `optmain -order 1` in the state of the
  first structure, from the Hessian of the calculator at its first point
  (`optimize_hessian=none`: as `optmain` starts without options), with a vibrational
  analysis of what it finds (`optimize_check=off`: none); a structure with no or several
  imaginary frequencies is reported;
- an estimate of a crossing is optimized with `optmain -mf opt_meci` or `opt_mesx` as above,
  with the analysis of the seam there. Then the crossing is followed down both surfaces
  (`optimize_irc=off`: not): a meta-IRC (`ircmain`) from it in the state of each structure,
  whose end is compared with that structure. An end within 0.1 Angstrom (RMSD after
  superposition; a judgment value) is called the structure, anything else another one.

`optimize_settings=FILE` names a settings file of `optmain` with further settings. The
optimizations run in the directory of the search, `optimized_estimates.csv` lists them
(estimate, kind, converged, iterations, energy, imaginary frequencies, file, and the two
RMSD values of a crossing), and the log has a line for each.

Measured on ozone with the simplified TDA on GFN0-xTB orbitals, the first and the second
excited singlet state (the first structure the minimum of the first state, the second from
a minimization in the second state, which ends next to the crossing): the ends met, the two
states were 5.8 kcal/mol apart at the estimate, `opt_meci` converged from it in 31
iterations to the conical intersection that a search from a bent structure finds (the
structures agree within 0.002 Angstrom, the gap is 5e-5 Hartree), and the two meta-IRCs
ended 0.0001 and 0.002 Angstrom from the two structures: 202 gradients and 6 exact Hessians
in all. One molecule and one pair of states; nothing is known yet about how often the mean
of the ends is close enough to the seam for larger molecules.

### A two-point method on a function of two states (`-mf` with `-dsafir`, `-gsm`, ...)

`-mf NAME` together with one of the methods of mopt-rs (`-dsafir`, `-gsm`, `-se_gsm`,
`-fsm`, `-lmm`) runs that method on the model function `NAME` of two states: every point of
the search has the energy and the gradient of the function (`seam`, `conical`, `avoiding`,
and the other functions of states that `optmain -mf` and `nebmain -mf` take). The first
state is that of the first structure (`-elec`, `-spin`, `-es`), the second that of the
second structure. Without one of these methods `-mf` is MultiOptPy's mode as before (it
comes first among MultiOptPy's methods and runs `optmain`). The functions that project
gradients for a crossing (`opt_mesx`, `opt_meci`) are refused: their gradient is not that of
an energy, so a path on them means nothing.

A function such as `seam` is low where the two states are close and low in energy, so the
points of interest are the low points of the path, not its top. The command therefore

- still writes the highest point as `<stem>_ts_guess.xyz` (a saddle of the function, for
  what it is worth) and gives the energies of the two states there;
- writes every point of the path that is lower than both its neighbours as
  `<stem>_crossing_guess.xyz` (`_2`, `_3`, ... for further ones) and gives the energies of
  the two states and their distance in kcal/mol at each;
- names the search for the crossing from a candidate (`optmain FILE ... -mf opt_meci` or
  `opt_mesx`, as above), and `-ieip_opt optimize=on` runs it for every candidate, with the
  analysis of the seam and, unless `optimize_irc=off`, the two paths down from the crossing.

When the path has no such point, the log says so and nothing is called a candidate.

Measured on the same ozone pair (first and second excited singlet state, `-dsafir -mf seam
-es 1 2`): the first structure is high on the function (the states are far apart there), so
only the second end moved, 7 steps towards the first; the function went down in its first
step and up from there. That point was the candidate, with the states 2.6 kcal/mol apart
(16.4 at the highest point); `opt_meci` converged from it in 20 iterations to the conical
intersection that the search with an end on each surface finds (structures within 0.001
Angstrom, energies within 4e-5 Hartree): 72 gradients and 6 exact Hessians in all, without
the paths down. In this pair the second structure is itself next to the crossing, so the
candidate is one step from an input structure and the search added little; a pair with the
crossing well between the two structures has not been measured. The other methods (`-gsm`,
`-fsm`, `-lmm`) run on the function through the same code but have not been measured on it.

### Experimental: the local minimax method and descent-ascent steps of the dimer

Both are new in mopt-rs, off unless asked for, and taken from other fields; no use of them
for molecular transition states was found.

- `-lmm` (registry `lmm`): the local minimax method of Li and Zhou (SIAM J. Sci. Comput.
  2001, 23, 840), with the normalized Armijo rule and the trial steps of Barzilai and Borwein
  as arXiv:2109.01865 restates them. From the minimum `A`, the peak of a direction `v` is
  the maximum of the energy along the ray `A + t v`; the directions descend on the unit
  sphere with the gradient at the peak (`v <- (v - a g) / |v - a g|`) until the peak meets
  the force thresholds. The first direction points to the second structure. mopt-rs adds
  the search of the peak (bracketing and the secant on `g . v`) and the projection of
  rigid-body motions. The method needs the pass to be reachable along straight rays from
  `A` in the coordinates of the points.
- `-ieip_opt dimer_translation=plain|optimistic|extragradient` (registry `dimer` key
  `translation`), with `dimer_rate_across` and `dimer_rate_along` (two step sizes across and
  along the axis): the translation of the dimer while its curvature is negative is a
  descent across the axis and an ascent along it; these replace the L-BFGS step by plain,
  optimistic (`-(2 G_k - G_k-1)`), or extragradient steps (Korpelevich 1976; the look-ahead
  costs one gradient per step), stabilizers of such descent-ascent problems from other
  fields (their analysis, arXiv:1901.08511, is for convex-concave problems; applying it to a
  saddle of order 1 is untested).

Measured (GFN0-xTB; gradients, then the barrier of the saddle `optmain -order 1` reaches):

| method | HCN -> HNC | CH3NC -> CH3CN |
|---|---|---|
| `-lmm` | 30; the first ray has no maximum within 30 evaluations | 30; the same |
| `-lmm -coord tric-cart` | 30; the same | 30; the same |
| `-use_dimer` (L-BFGS, the default) | 117; 68.285 | 988; a dissociation |
| `dimer_translation=plain` | 266; 68.285 | 3170 (iterations used up); 72.447 |
| `dimer_translation=optimistic` | 3912 (iterations used up); 68.285 | 8089 (used up); 72.447 |
| `dimer_translation=extragradient` | 348; 68.285 | 4017 (used up); 72.447 |

On the Muller-Brown surface (from the minimum B towards C, `crates/mopt-rs/tests/pair_registry.rs`)
the local minimax method reached the saddle with 6 gradients (the chain relaxation of
`nebmain` 210, the direct MaxFlux method 32, BITSS 111 and 14 energies), and the
descent-ascent steps of the dimer (rates 5e-4) did not converge in 3000 to 4000 gradients
where the L-BFGS translation needed 125. On the molecules the first straight ray of the
local minimax method from HCN and from CH3NC had no maximum within the 30 evaluations of a
ray (the energy kept rising along it; why was not analyzed further), so the method did not
start; the descent-ascent steps reached the saddle of HCN with two to thirty times
the gradients of the L-BFGS translation and the same wrong saddle on CH3NC. Neither is a
default; both are kept for further tests. The Muller-Brown results of every name, these
included, are printed by `crates/mopt-rs/tests/pair_registry.rs`.

### Measurements

GFN0-xTB built in, the defaults of every method, `-ieip_opt figures=off`; each estimate
refined with `optmain <stem>_ts_guess.xyz -order 1 -fc 1 -ns 100`. Gradients and exact
Hessians are the totals of the run (`Evaluations (in total)`). HCN -> HNC (`hcn_A.xyz`,
`hnc_B.xyz`; the saddle lies 68.285 kcal/mol above HCN):

| method | gradients | Hessians | end | refined: barrier (kcal/mol), iterations |
|---|---|---|---|---|
| iEIP | 26 | 0 | converged after 13 iterations | 68.285, 4 |
| `-use_dimer` | 117 | 0 | converged | 68.285, 1 |
| `-2pshs` | 130 | 1 | sphere shrunk onto the first structure | 68.285, 3 |
| `-gnt` | 116 | 0 | passed a transition state | 68.285, 21 |
| `-addf` (from HCN) | 102 | 1 | 1 estimate, every path ended | 68.285, 3 |
| `-use_spm` | 256 | 0 | converged | 68.285, 1 |

CH3NC -> CH3CN (`ch3nc_A.xyz`, `ch3cn_B.xyz`; the methyl migration, a saddle of `Cs`
symmetry with C-N 1.86 and C-C 1.61 Angstrom, lies 66.983 kcal/mol above CH3NC):

| method | gradients | Hessians | end | refined: barrier (kcal/mol), iterations |
|---|---|---|---|---|
| iEIP | 34 | 0 | converged after 17 iterations | not found (100 iterations; the estimate has two imaginary frequencies) |
| iEIP, CH3CN as `_A` | 80 | 0 | converged after 40 iterations | 66.983, 15 |
| `-use_dimer` | 988 | 0 | ends on the flat surface of CH3 + CN | not found |
| `-2pshs` | 435 | 1 | sphere shrunk onto the first structure | 66.983, 8 |
| `-2pshs -2pshs_step 0.02` | 656 | 1 | sphere shrunk onto the first structure | 66.983, 23 |
| `-gnt` (also with `-gnt_step 0.2`, `0.1`, `-gnt_vec 1,6`) | 16 to 101 | 0 | passed a transition state | not found |
| `-addf` (from CH3NC) | 314 | 1 | 2 estimates, every path ended | 66.983, 10; and 83.375, 7 |
| `-use_spm` | 168028 | 0 | cycles used up | not found |
| `-use_spm`, CH3CN as `_A` | 3192 | 0 | converged | 58.776, 9 (a hydrogen shift) |

What this shows: the searches between two minima that follow the energy (2PSHS, ADDF) found
the methyl migration from either side; the dimer, the spring pair, and the Newton trajectory
start from one minimum and climb to whichever saddle their direction leads to, which on
CH3NC is not the migration. iEIP keeps MultiOptPy's procedure, which is not symmetric in the
two structures: the term of the energy difference draws the images together when the first
is the lower one and pushes them apart when it is the higher one (both orders reproduce
MultiOptPy, `tests/golden/ieip_lj_ar7_swapped.json`). On these reactions the order with the
lower structure as `_A` found the saddle; with the higher one first the images met at a
point with two imaginary frequencies, from which no saddle of order 1 was reached.

The defaults that are not MultiOptPy's:

- Spring pair rates (`spm_drift`, `spm_climb`; the paper sets them per example). The drift
  is a gradient step, stable below `2 / lambda_max` (about 0.87 here: the stiffest Cartesian
  Hessian eigenvalue of these molecules is 2.3 Hartree/Bohr^2); the climb along the line of
  the pair grows the distance from the minimum by `1 + a3 lambda` per cycle:

  | drift | climb | HCN: gradients, end | from CH3CN: gradients, end |
  |---|---|---|---|
  | 1.0 | 1.0 | 24120, 60 cycles used up (drift does not settle) | - |
  | 0.3 | 0.3 | 240, 60 cycles used up | - |
  | 0.5 | 1 | 312, 60 cycles used up | - |
  | 0.5 | 3 | 260, converged | 3492, converged |
  | 0.5 | 5 | 256, converged | 3192, converged |
  | 0.5 | 10 | 260, converged | 3018, converged |
  | 0.3 | 3 | 402, converged | - |
  | 0.3 | 10 | 386, converged | 6104, converged |
  | 0.7 | 3 | 222, converged | 2598, converged |

  0.5 and 5 are the defaults: 0.7 is close to the limit of stability for stiffer bonds, and a
  climb of 5 stays stable at the saddle for curvatures down to -0.4 Hartree/Bohr^2.
- `symmetry_noise` 0.05 Bohr: on linear HCN without it, the dimer rotated into the
  dissociation of the hydrogen atom and the 2PSHS path stayed on the line (no saddle); with
  it every method found the saddle (first table).
- The dimer evaluates the end point anew after a rotation by more than three trial angles:
  on HCN three rotations of about 85 degrees each, extrapolated, turned a curvature of 0.10
  into -1.6, and the dimer climbed the wrong mode.
- Estimates that fall apart into more fragments than the first structure (bonds up to 1.5
  times the sum of the covalent radii; the covalent bonds of the saddles above stretch to
  about 1.3 times) are reported as dissociations and the run does not count as converged;
  ADDF ends such paths as dissociation channels instead of taking their flat ends for
  transition states.

## Automated transition-state search (`run_autots`)

`run_autots INPUT -cfg config.json [-ma ...] [--run_step4] [--skip_step1] [--skip_to_step4]`
runs the four steps of MultiOptPy's AutoTS, each with its block of the settings file (once
each, or as a `workflow` block says, see below):

1. **First path** (`step1_settings`). By default `optmain` from the input with AFIR (`-ma` or
   `manual_AFIR`) or another bias potential that drives the reaction; without one, step 1 is
   refused. With `"step1_method": "bitss"` (mopt-rs), the first and last structures of the input
   are the reactant and the product, and BITSS between them (`optmain -mf bitss`) makes the
   path: its optimization trajectory, used as it is, the way AFIR's is. With the name of a
   search between two structures (`"step1_method": "gsm"`, see below) that search makes it.
2. **Path** (`step2_settings`): `nebmain` from the first path, or `nebmain -dmf` with
   `"DMF": true`, or BITSS between the two ends of the first path with
   `"step2_method": "bitss"`. The highest `top_n_candidates` estimates of transition states go
   on.
3. **Transition states** (`step3_settings`): `optmain -order 1` from each candidate, with a
   vibrational analysis (mopt-rs; `"check_frequencies": false` turns it off). A structure with
   no or several imaginary frequencies is reported and does not go on; the same transition
   state found twice goes on once.
4. **IRC** (`--run_step4`, `step4_settings` with `intrinsic_reaction_coordinates`): the IRC of
   each transition state and the optimization of its two ends (`step4b_opt_method`), each
   verified to be a minimum by default (see "The ends of the IRC are checked" below).

The work directory (`work_dir`) keeps MultiOptPy's file names (`<stem>_step1_traj.xyz`,
`<stem>_step3_TS_Opt_Inputs/`, `<stem>_ts_final_<k>.xyz`, `<ts>_IRC_Endpoint_<j>.xyz`,
`<ts>_results/energy_profile.txt`) and adds:

- `autots_summary.csv`: every transition state, its energy and imaginary frequencies, the
  energies of its IRC ends, the barriers from each end, and the check of each end
  (`minimum`, `not_verified`, `not_checked`);
- `autots_report.html` (on by default; `"html_report": false` turns it off): the whole run on
  one page. It contains:
  - the energy during step 1: with and without the bias, or the two images and the BITSS
    function;
  - during molecular NEB and direct MaxFlux runs, Step 2 updates this same page after each
    saved band. Reload the page and choose a saved iteration to compare it with the initial
    path. Both curves use the energy of the first image of that run's initial band as zero
    (kcal/mol). The horizontal axis is the fractional image index, so bands with different
    numbers of images can be compared. The structure viewer shows the latest saved band.
    The history remains available after TS refinement and IRC. The original path profile,
    viewer, candidate list, and evaluation table retain their places. When the band driver
    saves figures, Step 2 also embeds the available energy, image-force, and convergence
    plots and offers their original SVG files for download;
  - the energy profile of the relaxed path with its candidates;
  - every optimization of step 3, with its lowest frequency and an animation of that mode.
    One that did not converge, or has no or several imaginary frequencies, is shown as a
    *candidate* with a warning, not as a transition state.
  - an energy diagram of all transition states on one scale, each drawn between the two
    minima its IRC ends at. Minima that are the same structure get the same label (M1, M2,
    ...), so reactions that share a minimum connect. A table of the minima comes with it.
  - the IRC profiles and paths;
  - the evaluations of every step and of the whole run: gradient calculations, exact Hessian
    calculations, energies alone, and the microiterations of an ONIOM environment (also the
    last lines of `autots.log`);
  - the settings;
  - the rights and credits: the license, the origin of the workflow, the sources of the
    viewer's colors and radii, and the references the steps logged for their methods and
    calculators.

  A viewer written into the page shows the structures and paths:
  - drag to rotate and the wheel to zoom;
  - ball and stick, sticks, space-filling or wireframe, for one molecule or all;
  - the labels of the atoms, for one molecule or all: their numbers from 1 (the default),
    none, their elements, or elements and numbers; the same choice is in the pages of
    `run_mapper`, `run_automecp` and `relaxedscan`, which use this viewer, and it is
    remembered by the browser;
  - play and a slider for paths and modes;
  - the cell of periodic (extended xyz) structures, with bonds to the next cells drawn as
    halves;
  - the ONIOM layers of the settings: the high layer in the chosen style, a middle layer as
    sticks, and the rest as faded lines.

  Every structure can be saved as an xyz file, one frame or all. The page follows the light or
  dark theme of the system, or the one chosen on it. It loads nothing from elsewhere and can be
  moved or sent on its own. Sections of steps that did not run say so.

Both files are written also when a step fails, with what was done until then; the error is
returned after them.

The same four steps can be written in Python from the building blocks (`Optimizer` with
`add_afir`, `relax_band`, `vibrational_analysis`, `irc_path`). Each decision the command
takes between the steps then becomes code of your own, such as which candidates go on and
what counts as a transition state or a minimum. A worked example, with a script that is
tested against this command, is in [python_workflows.md](python_workflows.md).

### Plain variables

With an objective of plain variables in `step1_settings` (`"othersoft": "mb2d"`, any objective
of `optmain -os`, any number of variables) the four steps run on the variables, in the process
of the command:

- the input holds frames of variables, the reactant first and the product last (the number of
  variables, a comment line, and `name value` lines: the format of the paths of `run_mapper`),
  or one variables file with `--skip_to_step4` (the transition state);
- step 1 is the two-point method `step1_method` of the registry of two-point methods between
  the first and the last point, `dsafir` without it (AFIR pushes need atoms; DS-AFIR is their
  two-ended form), with the keys of `step1_settings` that the method has as its settings;
  `--skip_step1` takes the input as the path;
- step 2 relaxes that path with the chain of `nebmain` (`neb`, Hessian-based steps;
  `step2_settings` are its settings) or BITSS (`"step2_method": "bitss"`); the highest
  `top_n_candidates` maxima of the relaxed chain go on;
- step 3 optimizes every candidate at saddle order 1 with the optimizer of plain variables and
  takes it as a transition state when it converged with one negative eigenvalue of the
  Hessian of the objective, counted by the one criterion of an imaginary mode for plain
  variables (`opt::mode_kill::imaginary_modes` with unit masses: an eigenvalue below
  -`IMAGINARY_CURVATURE`, 1e-8 in the units of the objective, the count of `optmain` and its
  mode kill on plain variables as well; before, a relative 1e-8 of the largest eigenvalue,
  looser than this where that eigenvalue is above 1 and tighter where it is below 1);
- step 4 runs the IRC with unit masses and optimizes its two ends.

The files are variables files and frames of variables (`<stem>_step1_path.txt`,
`<stem>_step2_path.txt`, `<stem>_ts_final_<k>.txt`, `<stem>_ts<k>_irc.txt`,
`<stem>_ts<k>_end_<j>.txt`); `autots_summary.csv` has the values of the objective as they are.
`autots_report.html` draws the path of step 1, the relaxed chain, the candidates, the
transition states and the IRCs with their ends on the variables (the surface itself on a grid
behind them for two variables; for more, on the two variables in which the points spread
most), the values along every path, the summary, and the evaluations of every step.
The Step 1–4 cards include frame playback, selectable coordinate axes and native text
downloads. The Step 2 NEB selector compares each evaluated band with its initial band
in objective units. Step 3 shows the optimization of each candidate, and Step 4 shows
the IRC and the coordinates of its endpoints. The
objective is not journaled: `--restart` runs the steps again. `tests/cli_autots_variables.rs`.

### Step 1 by a search between two structures

Step 1 need not push a reactant complex with an artificial force. When the reactant and
the product are known, any search of `ieipmain` between two structures can make the first
path:

```json
{
  "work_dir": "hcn_gsm",
  "step1_method": "gsm",
  "step1_settings": {"othersoft": "gfn0xtb"},
  "step2_settings": {"othersoft": "gfn0xtb", "NSTEP": 30},
  "step3_settings": {"othersoft": "gfn0xtb", "opt_method": ["rsirfo_block_bofill"],
                     "calc_exact_hess": 5, "frequency_analysis": true}
}
```

```
run_autots reactant_and_product.xyz -cfg config.json
```

- The input is one file with the reactant as its first structure and the product as its
  last (two frames of an xyz file; frames between them are not used).
- In a periodic cell the same works (the hop of a vacancy in copper, 31 atoms, the
  built-in EAM potential): each of the nine searches gave a first path whose points
  carry the cell, and with `gsm` the run went on through the chain and the
  transition-state optimization (0.5281 eV above the relaxed vacancy, the barrier that
  `nebmain` finds from the two ends) to step 4, where the IRC ended on both sides in the
  relaxed vacancy.
- Run as written with HCN and HNC as the two frames: the string of step 1, a chain of 30
  iterations, and three candidates that all end at saddles with one imaginary frequency,
  two of them the same one. With `gfn1xtb` in the place of `gfn0xtb` the same input ends
  in step 1: both molecules are linear, the straight line between them takes the hydrogen
  through the other two atoms, and the self-consistent charges of GFN1-xTB do not
  converge at such a structure of the string.
- `step1_method` names the search: `ieip` (the improved elastic image pair method),
  `2pshs`, `gnt` (growing Newton trajectory), `gsm`, `fsm`, `dsafir` (also written
  `ds-afir`), `lmm`, `dimer`, `spm`. `afir` (the default) and `bitss` are the two ways
  above.
- `step1_settings` holds settings of `ieipmain` then (its option names without the dashes,
  as in its settings file: `"othersoft"`, `"usextb"`, `"ieip_options": ["figures=off"]`,
  ...), not those of `optmain`. A key that `ieipmain` does not have is refused by name.
- The search runs in `<stem>_step1_<method>/`. Its points from the reactant to the product,
  `<stem>_path.xyz` there, are copied to `<stem>_step1_traj.xyz` and are the first path
  that step 2 relaxes. Every search of `ieipmain` between structures writes that file (see
  "Transition states from two structures or one"): the strings their path, iEIP the
  histories of its two images, 2PSHS its walk, the Newton trajectory its points, the dimer
  and the spring pair their trajectories, each ordered from the first structure to the
  second with the two structures as its ends.
- Steps 2 to 4 are as before: `nebmain` (or `nebmain -dmf`, ...) relaxes the path, and the
  highest images go to step 3. The estimates of the search itself stay in its run
  directory (`<stem>_ts_guess*.xyz`); they are not candidates of step 3.
- A search that did not end on its criteria still leaves its points; the log says which
  it was. One run only: `repeat` of a workflow block (below) is for step 1 with a bias
  potential.

### Several runs of a step (`workflow`, MultiOptPy's AutoTS v2)

A `workflow` block in the settings file says which steps run, how often, and with which
changes of their settings each time:

```json
"workflow": [
  {"step": "step1", "repeat": 2, "repeat_settings": [
      {"param_override": {"manual_AFIR": ["50", "1", "16"]}},
      {"param_override": {"manual_AFIR": ["-300", "1", "14"], "NSTEP": 20}}]},
  {"step": "step2", "repeat": 2, "repeat_settings": [
      {"param_override": {"NSTEP": 10, "DMF": true}},
      {"param_override": {"NSTEP": 15, "QSM": true}}]},
  {"step": "step3"},
  {"step": "step4"}
]
```

- An entry names a step (`step1` .. `step4`). `enabled: false` leaves it out; a step that
  is not listed does not run. `repeat` is the number of runs (1 unless given), and
  `settings_key` the block the settings come from (`<step>_settings` unless given).
- `repeat_settings[k].param_override` is laid over the block for run `k`. A list shorter
  than `repeat` uses its last entry for the runs after it.
- **Step 1**: every run starts from the last structure of the run before it
  (`<stem>_step1_run<k>_traj.xyz`, `_final.xyz`); the trajectories together are the first
  path (`<stem>_step1_combined_traj.xyz`). This is how two artificial forces are applied
  one after the other.
- **Step 2**: every run relaxes the path of the run before it
  (`<stem>_step2_run<k>_init_path.xyz` in, `_final_path.xyz` out), so a path can be relaxed
  first with one method and then with another, as in the example (the direct MaxFlux
  method, then the chain with `-qsm`). The candidates are those of the last run. With
  `"mode": "independent"` every run starts from the first path and the candidates of all
  runs go on (the highest `top_n_candidates` of them).
- **Step 3**: a later run takes up the candidates that have no transition state yet
  (converged, with one imaginary frequency where that is looked at), with its settings;
  what an earlier run found stays (`<stem>_s3_run<r>_ts_final_<k>.xyz`).
- **Step 4** runs once, with its first settings, and only with `--run_step4`.
- `--skip_step1`, `--run_step4`, and `--skip_to_step4` work as without the block.
- A run of step 1 or 2 after the first that fails is reported and ends that step with what
  the runs before it made.

Without a `workflow` block the four steps run once each. The differences from MultiOptPy's
implementation are in `migration.md` ("AutoTS with a workflow block").

Run on the aldol example of the quick start with the built-in GFN1-xTB (the settings of
`examples/quickstart/config_autots_builtin.json` and a block with one run of step 1 and
two of step 2, `{"NSTEP": 10, "DMF": true}` and then `{"NSTEP": 15, "QSM": true}`, both
with `"QSMv2": false`): the second run started from the path of the first, and of the
three candidates of the second run two ended at transition states with one imaginary
frequency; the third did not converge.

### Steps 3 and 4 on different surfaces

A transition state of step 3 is a stationary point of the surface of step 3. If step 4
runs on another one, its IRC starts at a point with a gradient, and its barriers compare a
transition state of one level with minima of another. With `--run_step4`, `run_autots`
therefore compares the two blocks before it starts and warns, in the log and in the
report, when they differ in the calculator, the method (`-func`), the basis set, the
per-element basis sets or core potentials, the unrestricted reference, the grid, the
solvation, the settings of the calculator (`-calc_opt`), the ONIOM setup, charge and
multiplicity, the electronic state (`-es`), the model function (`-mf`), or the bias
potentials, or the hard constraints. What only steers the optimization (the method, the
Hessians, the thresholds) is not compared. Steps 2 and 3 are compared the same way (the
calculator and its settings, the state, the model function, the biases and the
constraints). The run goes on in any case.

### Constraints and bias potentials of every step

The hard constraints (`-fix`, `-pc`, `-gfix`, `-uc`, `-vc_fix`) and the static bias
potentials (`-kp`, `-ubias`, `-wwp`, ... with the spellings of `optmain`) given on the
command line of `run_autots`, or in the block `shared_settings` of the settings file, are
merged into every step block and every run of a `workflow` block; a key that a block sets
itself wins over the shared value, and the log says so (the comparison of the surfaces then
names the difference). `-ma` stays the AFIR push of step 1 (not a bias of every step): with a
`workflow` block it replaces the `manual_AFIR` of every run of step 1, and the log says so;
`manual_AFIR` in `shared_settings` is refused. Every step holds the constraints at the values
of its input, which the step before held (the structure files have 12 decimals of
Angstrom), so the values of the first input hold to 1e-10 in every file of the run
(`tests/constraints_matrix_autots.rs`; the animations of the normal modes of the vibrational
analysis are displacements along the modes, not structures of the run); a variable cell with
constraints is refused by name (the chain of step 2 keeps the cell of its first image, and
`optmain -vc -irc` holds no constraints). The report and the log name the constraints and the
biases of the surface, and every transition state has its energy with and without the
biases (`tests/bias_matrix_autots.rs`).

### The ends of the IRC are checked

By default every end of the IRC (step 4) is verified to be a minimum. Its minimization checks
the converged structure by its Hessian with the one criterion of an imaginary mode
(`opt::mode_kill`: a mass-weighted eigenvalue below -`IMAGINARY_CURVATURE`, -1e-8, with the
zero modes projected out) and leaves a saddle point along its imaginary modes (`-modekill`
of `optmain`, continued with the optimizer of the ends, `-modekill_method classic`), so that an
IRC that ends on a symmetric stationary point (a symmetric transition state whose IRC keeps
the symmetry) does not give a saddle point as a minimum. The last structure is then judged by
the vibrational analysis of `optmain` (`-freq`, the same criterion). An end that did not
converge or keeps an imaginary mode is marked *end not verified*: a warning in `autots.log`,
the columns `end1_check` and `end2_check` of `autots_summary.csv` (`minimum`,
`not_verified`, `not_checked`), the line of the end in `energy_profile.txt` and the warnings
and the summary table of the report. `"check_frequencies": false` turns the check off for
step 3 and step 4 (the ends are then `not_checked`); `mode_kill`, `mode_kill_method` and
`frequency_analysis` in `step4_settings` override the defaults of the ends. On plain variables
the ends are checked by the eigenvalues of the Hessian of the objective (unit masses).
An IRC that fails numerically (it cannot follow its path within the constraints, or the
calculator fails along it) ends that transition state's step 4 with a warning in the log and
the report; the other transition states go on.

With `--skip_to_step4` the IRC runs from the input itself (`ircmain` with the settings of
step 4). Its start computes the exact Hessian of the input on the surface of the run (the
analytic Hessian where the engine has one; the constraints of the run and the static bias
potentials included), or takes the one that `initial_hessian` of `step4_settings` hands in
(the log says which), and decides in one place (`irc::start`) what is done from it: with
exactly one imaginary mode by the one criterion (`opt::mode_kill::IMAGINARY_CURVATURE`, the
rigid motions projected out) and a gradient within the convergence criteria of step 4, the IRC
of a transition state (both sides, the ends verified as minima); otherwise a meta-IRC, the
descent from the point itself. A meta-IRC comes with a notice, a warning in the console and
`autots.log`, among the warnings of the report and next to its path there: the input is not a
first-order saddle point, with its number of imaginary modes and, where it is not stationary,
its largest gradient component. Its path is labelled meta-IRC everywhere (the column `path` of
`autots_summary.csv` is `meta_irc`, the copied end is `<stem>_meta_IRC_Endpoint_1.xyz`,
`energy_profile.txt` has no barriers, the report shows no transition state, barrier or reaction
energy for it); its imaginary modes are recorded as found. Nothing of the criterion is
loosened: a second-order saddle point is never called a transition state. The evaluations of
the IRC count with step 4. On plain variables the IRC of the input decides in the same way, on
the Hessian of the objective (unit masses), with the same notice and labels
(`<stem>_ts1_meta_irc.txt`).

Before this, `--skip_to_step4` ran `optmain -order 1` from the input first (an optimization to
a saddle point with the Hessians of `step4_settings`, a model Hessian with updates unless
`calc_exact_hess` said otherwise) and the IRC from where it converged, whose start computed the
exact Hessian; with `check_frequencies` on (the default) the input, which had no vibrational
analysis, was skipped and no IRC ran at all. On plain variables a start that was not a
transition state was skipped with a warning.

Measured on the transition state of the reaction OH- + CH3F made symmetric under a plane
(`tests/cli_autots_irc_ends.rs`, GFN1-xTB, `--skip_to_step4`): the reactant side of the IRC
ended at the symmetric
stationary point of the complex, a saddle point with one imaginary mode of -64.5 cm^-1. The
mode kill with the HiSD continuation (a gradient descent at order 0, the default of `optmain`
for minima) lowered the energy by 1.7e-6 Hartree in 1000 iterations and did not converge;
with the optimizer of the ends (`classic`) it converged in 59 iterations at 8.4e-5 Hartree
(0.053 kcal/mol) below the saddle point, a minimum by its Hessian.

### One force field across the steps (GFN-FF)

The built-in GFN-FF sets its topology (bonds, hybridization, parameters) up from the first
structure it evaluates, so a step that starts from a transition state would see another
force field than the search that found it. Where a step uses `othersoft: gfnff` without a
topology of its own, `run_autots` writes the first frame of its input to
`gfnff_topology.xyz` in the run directory and gives every step `-calc_opt
topology=gfnff_topology.xyz`; a command that starts from a result of the run (`ircmain`,
`freqmain`) gets the same force field with that option. `run_mapper` writes the same file.

What else a step hands to the next, and how:

| state | handed on by `run_autots` | by `run_mapper` |
|---|---|---|
| the structure (with the environment of an ONIOM surface and the cell) | the files of the step before | in memory, every stage of a job |
| the GFN-FF topology | `-calc_opt topology=gfnff_topology.xyz` in every step | one calculator per run, its topology from the first input |
| hard constraints | made explicit at the input of the run in every step | resolved once at the first input |
| static bias potentials | the shared options in every step; differences named by `surface_differences` | the biased calculator of the run |
| the followed electronic state (`-es`) | the setting of each step block; a difference between steps 2, 3 and 4 is named | the settings of the run |
| SCC starting densities | not handed on (results do not depend on them) | not handed on |
| the microiteration tolerances of ONIOM (`-oniom_opt`) | the settings of each step block; a difference is named (`the ONIOM setup`) | the settings of the run |
