# Differences from MultiOptPy: calculators and surfaces

The calculators built into mopt-rs (pair potentials, embedded atoms, GFN0-xTB), combined
surfaces and bias potentials, crossings of two states, ONIOM, and periodic GFN-FF.

## Native calculators

The reference data in `tests/golden` was produced with `tools/golden/run_reference.py` (MultiOptPy)
and `tools/golden/ase_reference.py` (ASE). The checks below are reproducible with the scripts in
`tools/golden/`.

### Lennard-Jones Hessian

`LennardJonesCore.calculate_hessian` uses `-V'(r)/r` where the pair Hessian
`(V'' - V'/r) d d^T / r^2 + (V'/r) I` needs `+V'(r)/r`. On an argon cluster the analytic Hessian
differs from finite differences of MultiOptPy's own gradient by `8.37e-5` Hartree/Bohr^2 (largest
element `1.6e-3`); mopt-rs' Hessian agrees with finite differences
(`check_lj_hessian.py`). LJ energies and gradients agree with MultiOptPy to `1e-14`, and full
FIRELARS and RS-I-RFO (BFGS, FSB, Bofill) trajectories agree to `1e-8` Bohr at every iteration
(`tests/golden_lj.rs`).

### EMT double-counts pair terms

MultiOptPy's EMT is a port of ASE's EMT. ASE loops over a full neighbor list and adds each pair's
energy and force only to the central atom; MultiOptPy loops over the same full list but also adds
the energy and the reaction force to the neighbor, so every pair term is counted twice. For a
7-atom copper cluster MultiOptPy gives `-91.0` eV where ASE gives `+6.99` eV, and the forces differ
by up to `1.17` eV/Angstrom. Restricting MultiOptPy's loop to `a2 > a1` reproduces ASE's forces to
`6e-9` eV/Angstrom (`check_emt_double_count.py`). mopt-rs follows ASE (Jacobsen, Stoltze, Norskov,
Surf. Sci. 1996, 366, 394) and is tested against ASE.

### Tersoff energy and gradient

For a 5-atom silicon cluster MultiOptPy's Tersoff energy is `-0.0026` eV while ASE's Tersoff with
the same Si parameters (Tersoff 1988) gives `-4.35` eV; the forces differ by up to `12.4`
eV/Angstrom, and MultiOptPy's gradient disagrees with finite differences of its own energy by
`2.6e-4` Hartree/Bohr (`check_tersoff.py`). Its hetero-element tables (`Si-C`, `Si-Ge`, `C-Ge`)
have no cited source. mopt-rs implements Tersoff (Phys. Rev. B 1988, 38, 9902; 1989, 39, 5566,
with the 1989 mixing rules) and is tested against ASE.

### Embedded-atom method

New in mopt-rs (`pes::native::eam`, `-os eam`): the embedded-atom method with the analytic
functions and parameters of Zhou, Johnson, and Wadley (Phys. Rev. B 2004, 69, 144113) for
Cu, Ag, Au, Ni, Pd, Pt, Al, Pb, Fe, Mo, Ta, W, Mg, Co, Ti, Zr and their alloys, on the same
engine as EMT: exact gradient and Hessian, isolated systems and periodic cells.

The interactions are cut at `sqrt(5)` times the largest `r_e` of the elements present, the
cutoff of the tabulated files generated from these parameters, where the functions have
fallen to about `1e-6` of their values at `r_e`. Over the last tenth of the cutoff they are
switched off smoothly (see "Neighbor search and cutoffs" below; `-calc_opt switch=0` cuts
them as the tabulated files do). An atom without neighbors has the value of the low branch
of the embedding function at zero density (below `2e-6` eV in the published table) rather
than exactly zero, so there is no step when its first neighbor arrives.

Check of the parameters: for every element the crystal (fcc, bcc, or hcp with the ideal
axial ratio) at the nearest-neighbour distance `r_e` has the cohesive energy the potential
was fitted to within `2e-4` eV per atom (3.54, 2.85, 3.93, 4.45, 3.91, 5.77, 3.58, 2.02,
4.29, 6.81, 8.09, 8.76, 1.55, 4.41, 4.87, 6.36 eV in the order above), and its energy is
lowest within `1e-3` of that distance. Gradient and Hessian agree with central differences
for a cluster of four different metals and for a cell with an impurity
(`pes::native::eam::tests`).

### Neighbor search and cutoffs

New in mopt-rs.

**Linked cells** (`pes::native::neighbors`). The neighbor lists of the embedded-atom
potentials (EMT, EAM, MEAM), of Tersoff, and of Lennard-Jones with a cutoff come from a
linked-cell search: the atoms and those periodic images that can have a neighbor are sorted
into cells no smaller than the cutoff, and each atom looks into its cell and the 26 around
it. The lists are the same as those of a search over all pairs and lattice translations
(isolated systems, skewed cells with atoms outside them, cutoffs shorter and longer than
the cell, slabs and wires: `pes::native::neighbors::tests`). Time of the search at constant
density (65 neighbors per atom, debug build, one core):

| Atoms | Linked cells | All pairs |
|---|---|---|
| 1000 | 0.11 s | 1.8 s |
| 4000 | 0.22 s | 25.9 s |
| 16000 | 0.77 s | 412.7 s |

Memory is that of the lists (atoms times neighbors) and of one entry per atom, image, and
cell. The Hessian of a calculator is still a full matrix.

**Switching** (`pes::native::switching`). A potential that is cut at a distance has a step
of its energy there, which a search with a cutoff brings with it. Where mopt-rs cuts a
potential, it multiplies it by `S(r) = f(u) / (f(u) + f(1 - u))`, `f(x) = exp(-1 / x)`,
`u = (r_cut - r) / (r_cut - r_on)`, over the last part of the cutoff. `S` joins 1 and 0
with every derivative continuous (infinitely smooth), so the energy and all its
derivatives go to zero continuously; a polynomial switch would leave a jump in some
derivative.

| Potential | Default | Setting |
|---|---|---|
| `eam` | switched over the last tenth of the cutoff | `-calc_opt switch=ANGSTROM` (0: cut) |
| `lj` | all pairs, no cutoff, as in MultiOptPy (golden data) | `-calc_opt cutoff=ANGSTROM`, `switch=ANGSTROM` (a tenth of the cutoff) |
| `emt` | as in ASE (golden data): a step of relative size `1e-5` | `-calc_opt switch=ANGSTROM` |
| `meam` | the cutoff function of the model: energy and gradient continuous, the Hessian not | `-calc_opt smooth=true`: the infinitely smooth step in its place (distance and screening) |
| `tersoff` | the cutoff function of the model: energy and gradient continuous, the Hessian not | |

Measured: two copper atoms pulled through the cutoff of `eam` have a step of `2.6e-6` eV
without the switching and none (below `1e-14` eV) with it. The crystals of the sixteen
metals have a shell of neighbors at the cutoff `sqrt(5) r_e` itself, where rounding decides
which atoms count when the potential is cut; switched, their cohesive energies differ from
that by `1.5e-4` eV per atom at most (Co), the same for ranges of 0.02, 0.05, and 0.1 of
the cutoff, since no other shell lies in the range. The width is therefore not decided by
the crystals; a tenth of the cutoff is a convention.

Lennard-Jones in a cell: summed over nearest images (the default, as MultiOptPy), the
energy has a kink where a pair passes half the cell, and cells shorter than twice the range
miss images; with a cutoff all images within it are counted
(`pes::native::lennard_jones::tests`: a cutoff longer than the cell).

From Python the settings go to `NativeCalculator(name, settings=...)`,
`NativeEngine(name, settings=...)`, and `_core.evaluate_native(..., settings=[...])`.

### Modified embedded-atom method

New in mopt-rs (`pes::native::meam`, `-os meam`): the modified embedded-atom method of
Baskes (Phys. Rev. B 1992, 46, 2727) with the many-body screening of Baskes, Angelo, and
Bisson (1994), for Cu, Ag, Au, Ni, Pd, Pt, Al, Pb, Rh, Ir, Li, Na, K, V, Nb, Ta, Cr, Mo, W,
Fe, Mg, Si, C, Sn. The parameters are the first entry of each of these elements in the
library distributed with LAMMPS, and the conventions are those the library is used with
there (the weight `t1` raised by `3/5 t3`, the forms of `G`, `C_min = 2`, `C_max = 2.8`,
the screened nuclear repulsion below `alpha (r / r_e - 1) = -1`). Gradient and Hessian are
exact; isolated systems and periodic cells. From Rust, `Meam::new` takes parameters of
other elements and `Meam::with_pair` the reference of a pair of elements.

Differences from LAMMPS, and limits:

- The pair function is evaluated in closed form; LAMMPS interpolates it from a table of
  1000 points.
- The cutoff is 4 Angstrom, the default of LAMMPS, or 1.25 times the largest
  nearest-neighbour distance of the elements where that is longer (Pb 4.375, Na 4.645, K
  5.785 Angstrom): at 4 Angstrom potassium (4.628 Angstrom between neighbours) would not
  be bound at all. For the crystal of potassium the energy (-0.940000 eV) and the elastic
  constants (3.71, 3.15, 1.88 GPa) are the same to all digits printed with cutoffs of 1.1,
  1.17, 1.25, 1.5, and 2 distances, because farther neighbours are screened; 1.25 keeps a
  bond inside the cutoff until it is stretched by a fifth.
- Second-neighbour MEAM, reference structures other than fcc, bcc, hcp, diamond, B1, and
  B2, and parameter files are not included. A pair of different elements without a given
  reference gets, as in LAMMPS without a parameter file, an fcc reference with the means of
  the two cohesive energies, distances, and `alpha`; that is not fitted to any alloy, and
  the run log says so.
- The second derivative of the cutoff function of the model jumps where a neighbour or a
  screening atom comes in, so the Hessian is not continuous there (the gradient is).
  `-calc_opt smooth=true` (`Meam::with_smooth_cutoff`) puts an infinitely smooth step with
  the same limits in its place. The reference crystals keep their energies (their
  neighbours are either not screened or screened completely); structures with pairs
  screened in part get other energies than with the function the parameters were made
  with, which is why it is not the default.

Checks (`pes::native::meam::tests`): every reference crystal has the energy of the
universal equation of state at 0.9, 0.99, 1, 1.01, and 1.05 nearest-neighbour distances to
`1e-10` eV; gradient and Hessian agree with five-point differences for a cluster of five
elements with pairs screened in part and completely, for cells of fcc with an impurity,
bcc, diamond, hcp, for bcc pressed along an axis (second neighbours screened in part), and
for two atoms in the range of the nuclear repulsion.

Elastic constants of the cubic metals at their equilibrium, from second differences of the
energy (strain `2e-3`, GPa). They test the angular terms and the transcription of the
parameters, which were fitted to the shear constants; the bulk modulus `(C11 + 2 C12) / 3`
equals `alpha^2 E_c / (9 Omega)` of the equation of state within 0.5 %. The measured
values (room temperature) were written down from memory of standard tables and were not
looked up again; the test asks for agreement within 5 %.

| Metal | C11 | C12 | C44 | (C11 - C12) / 2 | measured (C11 - C12) / 2 | measured C44 |
|---|---|---|---|---|---|---|
| Cu | 170.2 | 122.7 | 75.9 | 23.8 | 23.5 | 75.4 |
| Ag | 124.2 | 93.5 | 46.2 | 15.3 | 15.3 | 46.1 |
| Au | 186.3 | 157.2 | 42.1 | 14.5 | 14.5 | 42.0 |
| Ni | 247.3 | 147.7 | 125.1 | 49.8 | 49.6 | 124.7 |
| Pd | 234.4 | 176.3 | 71.3 | 29.0 | | |
| Pt | 347.2 | 251.1 | 76.6 | 48.1 | 48.0 | 76.5 |
| Al | 112.3 | 66.2 | 28.0 | 23.1 | 23.0 | 28.5 |
| Pb | 48.9 | 41.5 | 14.8 | 3.7 | 3.6 | 14.9 |
| Rh | 422.6 | 192.3 | 194.3 | 115.2 | | |
| Ir | 599.8 | 256.4 | 269.4 | 171.7 | | |
| Li | 13.4 | 11.3 | 9.6 | 1.05 | | |
| Na | 7.6 | 6.3 | 4.3 | 0.63 | | |
| K | 3.71 | 3.15 | 1.88 | 0.28 | 0.28 | 1.88 |
| V | 230.3 | 120.2 | 43.2 | 55.1 | 54.7 | 42.6 |
| Nb | 245.3 | 132.2 | 28.4 | 56.6 | 56.0 | 28.7 |
| Ta | 262.3 | 156.2 | 82.7 | 53.1 | 52.9 | 82.5 |
| Cr | 346.3 | 66.1 | 100.1 | 140.1 | 141.0 | 100.8 |
| Mo | 459.6 | 168.2 | 111.2 | 145.7 | 142.0 | 110.0 |
| W | 517.6 | 203.2 | 157.2 | 157.2 | 159.0 | 160.6 |
| Fe | 230.4 | 135.3 | 117.2 | 47.6 | 47.5 | 117.0 |

Not done: a comparison of energies and forces with LAMMPS itself (not installed here), and
any check of the alloys.

### Excited states on GFN0-xTB orbitals

New in mopt-rs (`pes::native::gfn0::excited`; `-os gfn0xtb -es N`): excited singlet states
by a simplified Tamm-Dancoff approximation on the orbitals of the built-in GFN0-xTB, as a
built-in electronic structure with several states of one spin (see running.md). It is not
part of GFN0-xTB and not a parametrized method for spectra.

- Without exact exchange in the Hamiltonian, the response has none either: `A = (e_a -
  e_i) + 2 (ia|jb)'`. With the fraction 0.5 of sTDA-xTB, which belongs to another
  Hamiltonian, the lowest excitation energy of formaldehyde was -2.1 eV.
- The hardness of an atom in the Coulomb integrals is the diagonal element of the charge
  model of GFN0-xTB (`gam` alone is negative for hydrogen).
- Gradients of excitation energies are analytic (a Z-vector for the response of the
  orbitals, `pes::native::tda_gradient`; also for the TDA and sTDA of the built-in GFN1-xTB):
  within 1e-8 to 1e-7 Hartree/Bohr of the central differences below, 15 to 60 times faster.
  With `-calc_opt excited_gradient=numerical` they are central differences that follow the
  states by the overlaps of their amplitudes carried into the orbitals of the reference
  structure. Largest difference of a gradient component of formaldehyde from that with a
  step of `1.25e-4` Bohr (Hartree/Bohr):

  | Step (Bohr) | S1 | S2 | S3 |
  |---|---|---|---|
  | 4e-3 | 1.9e-6 | 1.2e-6 | 1.3e-6 |
  | 2e-3 | 4.6e-7 | 2.9e-7 | 3.2e-7 |
  | 1e-3 | 1.1e-7 | 7.1e-8 | 7.9e-8 |
  | 5e-4 | 2.7e-8 | 1.7e-8 | 1.9e-8 |

  The error falls with the square of the step; the default is `1e-3` Bohr (`1e-7`
  Hartree/Bohr, three orders below the convergence criteria).

**Staying with a state (`-es_follow`, new in mopt-rs; `pes::states`, `pes::followed`).**
MultiOptPy counts excited states by their energies at every structure (the root number goes
to the program as it is). mopt-rs does the same by default and can follow a state instead:
the state that was number `N` at the first structure is found again at every later
structure by the overlaps of its amplitudes with the states there (running.md, "Staying
with one excited state"). The calculators that hand out their states are the built-in
GFN0-xTB and PySCF with a restricted reference. The journal records which state every
evaluation was for (one more number per record; journals written before are read as they
are), so that a restart stays with the state. `-stability` and `-stability_external`, which
`ieipmain` and `mdmain` did not take, are options of these two commands as well now.

`-es` was parsed but not used before; it now selects the state, and `-mf` without a charge
and multiplicity of the second state takes the next state above `-es` (it used to take the
same state twice). Bias potentials are now added on top of the surface of `-mf` (they were
dropped when `-mf` was given).

Searches for conical intersections of excited states (S1/S2, S2/S3, S3/S4 of water,
hydrogen sulfide, hydrogen cyanide, ozone, HOF, ammonia, acetylene, formaldehyde,
ethylene, from structures without symmetry) served as a test of the gradient projection
and of the seam analysis together. Where both agree (ozone S1/S2, hydrogen sulfide S1/S2,
ethylene S1/S2, acetylene S1/S2 and S3/S4: gap below `5e-4` Hartree, the eigenvalue of the
difference of the Hessians along the second direction of the branching plane 10 to 60 times
the next one, overlap of that direction with the one of the branching plane updating 0.96
to 0.99) the point is a conical intersection and a minimum of the seam. In other cases the
search reported convergence while the seam analysis found a mean gradient within the seam
of `1e-3` Hartree/Bohr or more, with a low overlap of the two directions (searches that
kept their symmetry; this led to the check of the branching plane above), or two states
that run side by side with nearly the same gradient. `tests/cli_freqmain.rs` has the ozone
case and water in C2v.

## Combined surfaces and bias potentials

### One mechanism for bias potentials, model functions, and EDEEL

MultiOptPy has separate handlers for bias potentials, model functions (`-mf`), and EDEEL, each
calling the calculator itself. mopt-rs describes every surface as an expression tree
(`pes::combine::PesExpr`) of calculator evaluations, in-process terms (bias potentials and
user-defined terms), linear combinations, and nonlinear combiners. The optimizer requests the
evaluations through ask/tell, so combined surfaces work with Rust and Python calculators alike,
and independent evaluations run in parallel. A bias can be placed on one state of a combined
surface, which MultiOptPy cannot express. For "surface + bias" the split into the raw surface
(quasi-Newton-updated) and the exact bias Hessian is the same as in MultiOptPy. Tests:
`tests/combined_surfaces.rs`.

### `avoiding` model function

`-mf avoiding` is the Avoiding model function of Maeda, Ohno, and Morokuma (Adv. Phys. Chem.
2012, 268124): the upper adiabatic state of the two states coupled by
`U = (alpha/2) exp(-((E1 - E2)/alpha)^2)`, `(E1 + E2)/2 + sqrt(dE^2 + 4U^2)/2`, with
`alpha = 30 kJ/mol`; `-mf avoiding_lower` is the lower state. MultiOptPy's `avoiding` has the
coupling `U = (alpha/2) exp(-dE^2/alpha)` with `alpha = 0.01`, whose exponent is an energy and
not a number, so that its width depends on the unit of energy; it is kept as
`-mf legacy_avoiding`. In it MultiOptPy's energy is `(E1 + E2)/2 + sqrt(dE^2 + 4U)/2` while its
gradient is that of `(E1 + E2)/2 + sqrt(dE^2 + 4U^2)/2`, so the energy and the gradient describe
different surfaces; mopt-rs uses the two-state form `sqrt(dE^2 + 4U^2)` for the energy,
gradient, and Hessian. Test: `pes::combine::functions::tests::model_functions_are_consistent`.

### `conical` model function uses the absolute gap

MultiOptPy evaluates `sigma dE^2 / (dE + alpha)` with the signed `dE = E1 - E2`, which is singular
at `E1 - E2 = -alpha` when the states are given in the other order. mopt-rs uses `|E1 - E2|` as in
Levine, Coe, and Martinez (J. Phys. Chem. B 2008, 112, 405), so the function does not depend on
the state order. Test: `pes::combine::functions::tests::model_functions_are_consistent`.

### Gap curvature of the gradient projection (`opt_mesx`, `opt_mesx_2`, `opt_meci`)

The pseudo-gradient contains `2 (E1 - E2) x` with `x = (g1 - g2)/|g1 - g2|`; its derivative along
`x` is `2 |g1 - g2|`. MultiOptPy adds `2 x x^T` to the projected Hessian, which is consistent only
for `|g1 - g2| = 1` Hartree/Bohr and otherwise mis-scales Newton steps toward the seam (by a factor
of about 20 for a typical `|g1 - g2|` of 0.05 Hartree/Bohr). mopt-rs adds `2 |g1 - g2| x x^T`.

Within the seam, MultiOptPy projects the mean Hessian `(H1 + H2) / 2`. The energy within the
seam to second order has the Hessian of the Lagrangian, `W = (1/2 - l) H1 + (1/2 + l) H2` with
the multiplier `l = (mean gradient . x) / |g1 - g2|` (the seam bends), which is the mean only for
equal and opposite slopes; mopt-rs projects `W`. The difference matters where one state is far
from its minimum: at the equilibrium of state 1 (`l = -1/2`, `W = H1`) the mean takes the
downhill curvature of state 2 for the seam's, and from a structure already on the minimum of the
seam the exact Hessian gave a step of 0.19 Bohr (`etmain`, the crossing of the solvent shift of
`alpha_SA = 1` of the radical anion of 1,4-pentadiene with GFN1-xTB), after which 400
iterations did not return. A model Hessian (`-modelhess`) is given to both states and goes
through the projection like an exact one (before, it replaced the projected Hessian as it was).

The Hessian is updated with the gradient of the surface without its in-process terms (the bias
potentials, whose Hessians are exact). Below a combiner that part was taken at the energies of
the states without their terms, so a constant term in one state (the solvent shift of a state in
`etmain`) moved the point of the combination: the gap in `2 (E1 - E2) x` was that of the
unshifted states, off by the difference of the shifts (0.063 Hartree at `alpha_SA = 1` of the
radical anion of 1,4-pentadiene with `-et_lambda_s 0.6 0.6`), and the updates learned the turning
of `x` times that gap as curvature. The crossing at `alpha_SA = 0.5` of that run did not converge
in 400 iterations from Lindh's model; it now takes 22 (`electron_transfer.md`). The derivatives of the part without terms are
now taken at the energies with them (`pes::combine`, test
`the_gradient_without_terms_is_taken_where_the_shifted_states_are`); only its energy is that of
the unshifted states. The same holds for the weight of a penalty function and for any other
function of the states.

`opt_mesx` and `opt_mesx_2` are the same method in MultiOptPy and are aliases in mopt-rs. Tests:
`pes::combine::projection::tests::mesx_pseudo_gradient`,
`mesx_reaches_the_analytic_minimum_on_the_seam`,
`a_model_hessian_goes_through_the_projection_of_a_crossing_search`.

### Branching plane updating

The BPU recursion of `opt_meci` is unchanged. Where MultiOptPy draws a random vector (the mean
gradient is parallel to the difference gradient at the first iteration), mopt-rs uses the
Cartesian axis with the smallest overlap with `x`, orthogonalized, so runs are reproducible. The
branching plane advances once per accepted iteration, not once per gradient evaluation. Test:
`meci_reaches_the_conical_intersection`.

### Check of the branching plane of a search for a conical intersection

MultiOptPy's `opt_meci` (gradient projection with branching plane updating) starts the
second direction of the branching plane from the mean gradient and updates it from the
steps. mopt-rs keeps this and adds a check when the search has converged
(`opt::crossing_check`, on by default, `-no_crossing_check`): the second direction from the
difference of the gradients of the two states at displaced structures, and with it the
mean gradient within the seam; if that is not converged, the search goes on with the
direction found.

Why: from a start with symmetry the coupling direction is not in any gradient of the
search, and the search stops where the gap closes. Searches with the excited states on
GFN0-xTB orbitals, RMS of the mean gradient within the seam at the end (Hartree/Bohr):

| Search | Without the check | With the check |
|---|---|---|
| water (C2v) S2/S3 | 1.3e-2, 6 iterations | 9.1e-5, 22 iterations |
| formaldehyde (C2v) S2/S3 | 3.1e-2, 11 iterations | 1.2e-4, 27 iterations |
| ozone S1/S2 (no symmetry) | 1.4e-5, 34 iterations | 1.4e-5, 36 iterations |
| hydrogen sulfide S1/S2 (no symmetry) | 8.1e-6, 58 iterations | 8.1e-6, 60 iterations |
| acetylene S1/S2 (no symmetry) | 2.0e-5, 242 iterations | 2.0e-5, 244 iterations |

Where the search had the plane right (overlap of the two directions 0.999 or more), the
check costs two iterations and changes nothing.

The displacement: overlap of the direction found with that from the Hessians, after one
product, for displacements of 0.002, 0.005, 0.01, 0.02, and 0.05 Bohr: ozone 0.9923 for
all five, hydrogen sulfide 0.9606, formaldehyde (C2v) 0.9882 to 0.9883, water (C2v) 0.9965.
The direction does not depend on the displacement in this range; the default is 0.01 Bohr,
and a second product along the direction found changes the overlap in the fourth digit.

The same direction is now used by the seam analysis. The eigenvector of the difference of
the Hessians, used before, is off where the Hessians are finite differences across the
cusp: for acetylene S1/S2 it gave a mean gradient within the seam of 1.8e-3 where the
displaced gradients give 2.0e-5, the value the search had converged to.

### EDEEL follows the published method

Reference: Mutsuji, Saita, Maeda, RSC Adv. 2023, 13, 32097 (doi:10.1039/D3RA05784D). MultiOptPy's
EDEEL handler is marked as under construction and differs from the paper:

- The diabatic energies are eqn (7) and (8), `V11 = E^C_{n+m} - E^D_n + E^D_{n+1}` and
  `V22 = E^C_{n+m} - E^A_m + E^A_{m+1}`, where `E^C_{n+m}` is the complex *without* the moving
  electron. MultiOptPy takes five free charge/multiplicity pairs; mopt-rs (`EdeelStates`) checks
  that charges and multiplicities follow the electron counts `n, n+1, m, m+1, n+m` (e.g. the
  complex of a M(III)/M(II) self-exchange is evaluated as M(III)-M(III)).
- The crossing is a minimum-energy seam of crossing optimized by gradient projection (the paper
  cites Bearpark et al. 1994 and Maeda et al. 2010). MultiOptPy's default `sx` target minimizes
  the penalty function `(V11 + V22)/2 + sigma (V11 - V22)^2` (`sigma = 2.0`), which is not part
  of the method and converges to a point with a finite gap; it remains available as
  `EdeelTarget::Penalty`.
- The coupling `V12 = V11 - E^C_{n+m+1}` (eqn (9)), the reorganization energy (eqn (10)), the
  Landau-Zener transmission coefficient and rate constant (eqn (1)-(5)), and the activation
  strain analysis (eqn (11)-(13)) are provided in `analysis::electron_transfer`; MultiOptPy has
  none of them. The published rate of [Fe(H2O)6]3+/2+ is reproduced from its tabulated barrier.
- The complex is evaluated once per iteration and shared by both states; fragment gradients and
  Hessians are scattered onto the complex. The paper keeps the donor-acceptor distance fixed
  during the seam optimization; this is a distance constraint of the run.

Tests: `edeel_shares_the_complex_and_matches_finite_differences`,
`edeel_crossing_optimization_reaches_the_seam`, `analysis::electron_transfer::tests`.

### AFIR under periodic boundary conditions

MultiOptPy's AFIR term uses plain Cartesian distances. mopt-rs uses minimum-image distances in
periodic cells, as in the periodic AFIR applications of Takagi et al. (Phys. Rev. B 2017, 95,
184110) and Sugiyama et al. (Phys. Chem. Chem. Phys. 2019, 21, 14366); for isolated systems the
term is MultiOptPy's. Its gradient and Hessian are analytic (MultiOptPy uses automatic
differentiation). The `-ma gamma1,gamma2` ramp over 300 iterations and `gamma = 0` disabling a
term follow MultiOptPy. Tests: `pes::bias::afir::tests`, `manual_afir_pushes_fragments_together`.

### Every bias potential works in isolated, 1D, 2D, and 3D cells

All bias options (`-ma`, `-kp` ... `-nrp`) measure distances, angles, and centroids through the
minimum image along the periodic axes of the cell, with exact gradients and Hessians from
automatic differentiation over the atoms each term involves. MultiOptPy evaluates them on the
raw Cartesian coordinates. Absolute positions (`-fp`, `-vpp`, `-wwp`, `-vpwp`, `-nrp`) are
measured from the minimum image of the reference point.

Tests: `tests/bias_restraints.rs`, `tests/bias_fields.rs` (every potential passes
`pes::term_check` in isolated, 1D, 2D, and 3D skewed cells: finite-difference gradient and
Hessian, invariance under lattice translations of single atoms, no wrapping along non-periodic
axes).

### Bias potentials with microiterations share the ONIOM machinery

`-aerp`, `-aerpv2` (ellipsoid rotation angles) and `-smp` (solvent particles) minimize auxiliary
variables at every evaluation. MultiOptPy runs a separate FIRE loop per potential (up to 15000
steps per ellipsoid) and adds only the coupling part `-H_xy H_yy^-1 H_yx` as an effective
Hessian. mopt-rs relaxes them with `pes::microiter`, the inner loop that ONIOM microiterations
use (damped Newton with mode kill, so the auxiliary variables end at a minimum), and gives the
full Hessian of the relaxed surface `H_xx - H_xy H_yy^+ H_yx`. Starting values: a 72-point grid
per rotation angle (MultiOptPy: random angles); particles spread over a sphere around the
targets (MultiOptPy: normal random positions with a 100 Bohr spread).

Tests: `pes::microiter::solver::tests` (mode kill leaves a saddle point, Schur complement),
`tests/bias_fields.rs::ellipsoidal_potentials_minimize_the_rotation_and_pass_the_term_checks`,
`tests/bias_fields.rs::spacer_particles_relax_and_pass_the_term_checks`.

### Corrected bias potentials

| Option | MultiOptPy | mopt-rs |
|---|---|---|
| `-lmefp` | sums the Cartesian components of the unit vectors of both atom pairs | `-(F/2)(r_1 + r_2)`: constant pulling force along each pair (EFEI) |
| `-lmefp`, `-lmefpv2` | `1 pN = 1.213e-5` a.u. | `1 pN = 1.2138e-5` a.u. from the CODATA constants used throughout |
| `-kdac` | dihedral with the sign opposite to `-kda` / `-kdav2` | IUPAC sign for all three, so `phi0` means the same angle in each |
| `-kav2` | quadratic continuation beyond 0.001 rad of 0 or pi omits `d2theta/du2` (Hessian jumps at the cut) | continuation is twice continuously differentiable |
| `-metad` dihedral | angle modulo 2 pi subtracted directly (bias jumps at 0) | Gaussians in the difference wrapped into (-pi, pi] |
| `-metad` out-of-plane | angle modulo 2 pi (negative angles moved next to 2 pi) | angle in [-pi/2, pi/2] used as is |
| `-aerp`, `-aerpv2` | direction-dependent well depth normalized by the ellipsoidal distance of all atoms together | per atom, along its unit vector |
| `-aerp`, `-aerpv2` | ellipsoid-ellipsoid term rotated about the helper's default axis | about the bond (`z`) axis, like the atom term |
| `-smp` | cavity uses the distances of particle and atoms from the coordinate origin | distance between the particle and the nearest target atom |
| `-nrp` | inner wall pushes atoms out of the inner sphere; outer wall uses the contraction constant in both phases | Wang et al., Nat. Chem. 2014: `f(t) U(r, r1, k1) + (1 - f(t)) U(r, r2, k2)`, walls confine from outside; the recommended `8 14 1.5 0.5 1.0 0.5` gives the paper's 14 A / 1.0 for 1.5 ps and 8 A / 0.5 for 0.5 ps |

Tests: `tests/bias_restraints.rs::known_values`, `angle_restraints_are_smooth_at_linear_geometries`;
`tests/bias_fields.rs::metadynamics_dihedral_gaussians_are_periodic`,
`nanoreactor_confines_from_outside_like_the_paper`.

## ONIOM

MultiOptPy reads `-oniom HIGH_ATOMS LINK_ATOMS LOW` into `force_data["oniom_flag"]`, but its
`optimization.py:ONIOMHandler.compute` begins with `raise "Not Implemented!"` (which itself
fails: a string is not an exception), so no ONIOM calculation runs. Its helpers
(`Utils/oniom.py`) pair every given link atom with the nearest atom of the high layer and put
the hydrogen at the sum of the covalent radii of the high-layer atom and hydrogen along the
bond. mopt-rs implements ONIOM as a calculator (`pes::oniom`), so that every command has it:

- subtractive ONIOM with N levels on nested regions (`-oniom`, `-oniom_middle`), mechanical
  or electronic embedding; every term with a calculator made for its own system;
- the link atoms at a share `g` of the bond (the default, as Gaussian and Dapprich et al.
  1999: a linear map, so that gradients and Hessians go back exactly) or at MultiOptPy's
  fixed distance (`-oniom_opt link=distance`, with the second-order term of the map in the
  Hessian); the pairing of given link atoms as MultiOptPy, or `auto` for every bond across
  the border;
- the charge and the multiplicity of each model system (`-oniom_opt high_charge=`,
  `high_multiplicity=`, `middle_*`), by default those of the whole structure with the
  multiplicity changed by one where the link atoms change the parity of the electrons.

`pes::oniom::tests` (one level on every region gives the energy and the gradient of the
whole structure to 1e-10, for two and three layers; gradients and Hessians against
differences for both kinds of link atoms), `tests/cli_oniom.rs` (ethanol, GFN0-xTB on the
hydroxymethyl group and GFN-FF on the whole: an optimization to a stationary point of the
ONIOM surface, its vibrations without imaginary modes).

Added since:

- **Microiterations in every run but the molecular dynamics** (`pes::oniom::micro`,
  `cli::oniom_micro`; running.md, "Microiterations"). The optimizer, the IRC, the chain, the
  pair of a two-point method, and the searches move the core; the environment is relaxed to
  a minimum (with mode kill) before every evaluation. A run is never continued without them
  because an option does not fit: atoms of the environment that a constraint or a bias
  potential names are moved with the core, user functions and bias potentials are evaluated
  at the core in the whole system, `-es`, `-mf`, and `-ih` are taken to the relaxed surface,
  ReSPer coordinates are made of the atoms of the core; what cannot be taken (a bias on
  every atom, `-vc`, `-mf bitss`) is refused. `-oniom_opt micro=off` alone moves every atom.
  The ONIOM surface of whole structures is a type of its own (`pes::oniom::WholeOniom`) that
  the registry hands out only through `create_oniom`; `CalculatorRegistry::create` refuses
  an ONIOM spec, so that a command cannot run on the unrelaxed surface by oversight.
  `tests/cli_oniom.rs`, `tests/cli_relaxedscan.rs::an_oniom_scan_keeps_its_microiterations`,
  `tests/cli_confsearch.rs::oniom_with_gfn0_xtb_in_gfn_ff`.
- **Electronic embedding** (`-oniom_opt embedding=electronic`, `pes::oniom::embedding`,
  `pes::point_charges`): the higher level of every region in the charges of the atoms around
  it, the classical Coulomb energy of the model with them taken off with the lower level;
  the charge of each `Q2` redistributed (Lin and Truhlar 2005), left out, or kept. The
  built-in GFN1-xTB takes point charges as xtb's embedding of GFN1-xTB does; GFN-FF gives
  its topology charges. `pes::oniom::tests` (gradients against differences for the three
  boundary rules), `pes::native::gfn1::tests` (the gradients of the atoms and of the
  charges), `tests/cli_oniom.rs::electronic_embedding_goes_with_the_microiterations`.
- **States of `-mf` with ONIOM**: the model systems follow the charge and the multiplicity
  of the state of the whole structure (before, every state was computed with the model
  system of the first).

Errors of the first version of the microiterations that are not repeated:

- runs with bias potentials (AFIR), `-mf`, `-es`, `-ih`, a constraint on an atom of the
  environment, or ReSPer coordinates went on without microiterations with a note in the log;
  `confsearch` and `ieipmain` made the ONIOM calculator directly and moved every atom, and
  the chains of `nebmain` were chains of all atoms (a path in water was cut into 186 images
  by the motion of the solvent and did not converge);
- the fragments of `-gfix` and the reactive bonds of CCQN were applied to the structure of
  the core with the atom numbers of the whole system;
- the point group of the final structure was that of the core alone;
- the files of a run were rewritten with the whole system from the environments of the last
  4096 evaluated structures, so that an older structure of a long run got the environment of
  another one; they are kept within a budget of memory now, and a structure that is not
  kept has its environment relaxed again when the file is written.

## Periodic GFN-FF

MultiOptPy has no periodic GFN-FF; `gfnff` of mopt-rs follows xtb 6.7.1. Its approximations
are kept so that the numbers are xtb's (see "Periodic GFN-FF" in `calculators.md`). Where mopt-rs
differs from xtb:
- **Cells.** A cell is kept as an integer vector instead of xtb's index into a translation
  table. xtb stores some indices of the setup's table and reads them with the energy's
  table. When a translation of the second shell between 42.43 and 44.72 Bohr comes earlier
  in the order, that picks another image (cubic cells of 12.25 to 12.9 Bohr). mopt-rs uses
  the cell of the setup. In xtb's own reference set the case does not arise.
- **Unreachable atoms.** A bonded three-body term whose third atom lies in a cell that
  xtb's sum of translations does not reach is left out; xtb reads outside its table there.
- **One-sided bonds.** xtb removes these (metals) before counting bond paths. They do not
  occur for H to I, so that step is not ported.
