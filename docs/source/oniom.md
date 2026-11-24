# ONIOM (`-oniom`)

```text
optmain ethanol.xyz -os gfn0xtb -oniom 2-4,8,9 auto gfnff
optmain enzyme.xyz -os pyscf -func b3lyp -oniom 1-20 auto gfnff -oniom_middle 1-80 auto gfnff
```

`-oniom HIGH_ATOMS LINK LOW` makes the calculator of `-os` the high level on the atoms
`HIGH_ATOMS` and `LOW` (any calculator name) the low level on the whole structure, combined
by subtraction (Svensson et al. 1996; Dapprich et al. 1999): `E = E_low(real) +
E_high(model) - E_low(model)`. Every command takes it (optimizations, chains, IRC,
vibrations), as it is a calculator like the others. The model system is the region with a
hydrogen link atom on every bond to the outside: `LINK` lists the outside atoms (each with
its nearest atom of the region, as MultiOptPy pairs them), `auto` takes every bond across
the border (1.2 times the sum of the covalent radii), `none` caps nothing. `-oniom_opt
link=scaled` (the default) puts the link atom at the share `g = (R(Q1) + R(H)) / (R(Q1) +
R(Q2))` of the bond, as Gaussian; `link=distance` at `R(Q1) + R(H)` from `Q1`, as MultiOptPy
(its Hessian has the second-order term of that map). The model systems have the charge of
the whole structure and its multiplicity, changed by one where the link atoms change the
parity of the electrons (a singlet becomes a doublet and a doublet a singlet; a high-spin
state keeps its unpaired electrons and takes one more, so that a triplet becomes a quartet,
not a doublet), unless `-oniom_opt high_charge=Q high_multiplicity=M` (`middle_charge=`,
`middle_multiplicity=` for the middle layer) says otherwise. `-oniom_middle ATOMS LINK CALC` adds a middle layer (three-layer ONIOM: the
atoms of the middle region hold those of the high one). Every term has a calculator of its
own, made for the system it computes. An excited state (`-es`) is that of the high level;
a periodic cell is that of the whole structure (the model systems are clusters, made with
the minimum image); a stress (`-vc`) is not available. One evaluation of the ONIOM surface
counts as one in the totals of the run. By default this is mechanical embedding (the model
systems are computed by themselves).

The lower levels are not tied to the built-in force field: `LOW` (and `CALC` of
`-oniom_middle`) is any calculator name of `-os`. Run through the Python package, that
includes every ASE calculator (`ase:module:Class`), the profiles of machine-learned
potentials, and the Python engines; the environment is relaxed with that calculator in the
microiterations as it would be with GFN-FF. Each level has its own settings: `-func`, `-bs`,
and `-calc_opt` belong to the highest level and are not handed down, and `-oniom_opt
low.KEY=VALUE` (`middle.KEY=VALUE`) gives those of a lower level, `method` and `basis` for
its method and basis set and any other key as `-calc_opt` takes it for that calculator (the
keyword arguments of an ASE calculator, `low.hb_switch=off` for GFN-FF). The threads, the
memory, and `software_path.conf` are those of the run.

```text
optmain complex.xyz -os pyscf -func b3lyp -bs def2-svp -oniom 1-20 auto tblite:GFN2-xTB
optmain cluster.xyz -os gfn1xtb -oniom 1-12 auto ase:MODULE:CLASS -oniom_opt low.KEYWORD=VALUE
optmain solute_in_water.xyz -os gfn1xtb -oniom 1-14 none openmm -oniom_opt low.forcefield=amber14-all.xml,amber14/tip3p.xml low.topology=whole.pdb
```

(`tests/test_oniom_engines.py`: an ASE calculator below a built-in one, with its
microiterations and its own settings; `tests/test_openmm_engine.py`: OpenMM below GFN-FF.)

## OpenMM (`-os openmm`)

OpenMM is not a dependency of mopt-rs; where it is installed (`conda install -c conda-forge
openmm`), `openmm` computes energies and forces with its force fields (Python package; the
Hessian comes from differences of the forces). An OpenMM system belongs to one set of atoms,
so the calculator takes one for every set it is asked to compute and picks the one that
fits a structure:

- `-calc_opt system=FILE.xml[,FILE2.xml...]`: systems written by OpenMM's `XmlSerializer`
  (from any tool that makes OpenMM systems); a structure takes the one with its number of
  particles and, where masses are set, its elements;
- `-calc_opt forcefield=amber14-all.xml[,...] topology=FILE.pdb[,FILE2.pdb...]`: a force
  field of `openmm.app.ForceField` on the topologies of PDB files, whose atoms are in the
  order of the structures (the elements are compared). The structure itself comes from the
  input of the command, the PDB file gives residues, atom names, and bonds.

Bonds are flexible (no constraints, no rigid water), since the optimizer moves every atom.
`nonbonded=NoCutoff|CutoffNonPeriodic|CutoffPeriodic|Ewald|PME|LJPME` (without it:
`NoCutoff` for a molecule, `PME` in a cell), `cutoff=NM`, `create.NAME=VALUE` for further
keywords of `ForceField.createSystem`, `platform=CUDA|OpenCL|Reference|CPU`, and
`properties.NAME=VALUE` for the platform. Without `platform` the calculator takes one that
computes in double precision: CUDA or OpenCL with `Precision=double` where there is one, the
Reference platform otherwise. The CPU platform works in single precision (energies to about
seven digits), which is too coarse for the energy differences of a converging optimization
of a large system and for differences of its forces: on chorismate mutase (2CHT, 5696
atoms) a Hessian-vector product by forward differences of the forces along a step of 1e-4
Bohr was off by 1.6 percent on the CPU platform and by 0.15 percent (the error of the
differences themselves) on Reference, which took 54 ms per evaluation against 5.6 ms. It is
used when asked for, and the run log says what it costs. In a periodic cell the first lattice vector
has to lie along x and the second in the xy plane (OpenMM's form); the stress is not
available.

As a lower level of ONIOM, `openmm` is asked for the whole structure and for every model
system (the region with its link hydrogens). The system or the topology of the whole
structure is enough: the command tells the calculator which atoms a model system is made
of and which bonds it cuts, and the system of the model is cut out of that of the whole
structure (`mopt_rs.calculators.openmm_model`):

- the atoms of the region keep their masses, charges, and Lennard-Jones parameters;
- a term of the force field (a bond, an angle, a torsion, an exception of the non-bonded
  forces) is kept when all its atoms are in the region, or when its atom outside is the
  outer atom of a cut bond and stands for the link atom of that bond;
- a link atom has no charge and the Lennard-Jones parameters of a hydrogen on carbon of
  the Amber force fields; the bond to it has the force constant of the cut bond and its
  length scaled as the link atom is placed, and its 1-4 interactions keep the scale of
  those of the outer atom;
- the non-bonded forces of the model are those of a molecule (no cutoff, no cell).

This is a choice for the terms at the border and not a parametrization of the link atom:
what the two levels have in common cancels in `E_low(whole) + E_high(model) -
E_low(model)` however the link atom is described. Systems with other forces than
`HarmonicBondForce`, `HarmonicAngleForce`, `PeriodicTorsionForce`, `RBTorsionForce`,
`CMAPTorsionForce`, and `NonbondedForce` are refused by the name of the force; a system
or a topology given for the model system (`low.system=whole.xml,model.xml`,
`low.topology=whole.pdb,model.pdb`) is used in the place of the cut-out one. Electronic
embedding with the charges of an OpenMM system is not connected yet (`charges=FILE` gives
them).

Checked with OpenMM 8.2 (`tests/test_openmm_engine.py`): a serialized system against its
energy by hand and against differences of the energy, TIP3P water molecules from a force
field and PDB files, and an optimization of a water dimer with GFN-FF on one molecule and
OpenMM below it, whose first energy is `E_low(whole) + E_high(model) - E_low(model)` of the
two calculators by themselves to 1e-8 Hartree. The cut-out systems
(`tests/test_openmm_model.py`): a chain with bonds, angles, a torsion, and scaled 1-4
pairs, cut after two and after three atoms and at both sides of one atom, against the
energies of the terms by hand; and a run with OpenMM on both levels and a link atom, whose
energy is that of the whole chain to 1e-9 Hartree (the two model terms cancel).

## Electronic embedding (`-oniom_opt embedding=electronic`)

```text
optmain claisen_in_water.xyz -os gfn1xtb -oniom 1-14 none gfnff -oniom_opt embedding=electronic
```

With electronic embedding (Vreven et al., J. Chem. Theory Comput. 2006, 2, 815) the higher
level of every region is computed in the field of the charges of the atoms around it, and the
classical Coulomb energy of the model atoms with the same charges is taken off with the lower
level:

```text
E = E_low(real) + E_high(model; q_env) - E_low(model) - sum_(i in model, K) q_i q_K / r_iK
```

so that the interaction of the model with its environment is that of the higher level. The
charges are fixed: by default those of the lowest level (`charges=lowest`; for GFN-FF its
topology charges, which are fixed with its topology, so that the gradient has no derivatives
of the charges), or one per atom of the whole structure from a file (`charges=FILE`, numbers
separated by white space, in e). The link atoms carry no charge. The charge of an atom `Q2`
(outside, replaced by a link atom) would sit next to the link atom; `boundary=rc` (the default)
moves it to the midpoints of the bonds of `Q2` to its other neighbours in the environment, in
equal shares (the redistributed charge scheme of Lin and Truhlar, J. Phys. Chem. A 2005, 109,
3991; the total charge is kept), `boundary=zero` leaves it out, `boundary=keep` leaves it
where it is.

- **Which levels take charges.** The higher level of a region must take point charges:
  the built-in GFN1-xTB does, as xtb embeds GFN1-xTB (`src/embedding.f90`): every shell sees
  `V_a = sum_K q_K / sqrt(r^2 + 1/g^2)` with `g` the harmonic mean of the hardness of the
  shell and that of the element of the charge (nitrogen for a charge of no element, such as
  a redistributed one), the potential enters the Fock matrix and the energy as `sum_a q_a
  V_a`. GFN0-xTB and the other built-in calculators do not, and a run with them is refused
  with the reason.
- **The lower level.** The classical term taken off is the Coulomb energy of the model atoms
  with the charges of the lowest level. It cancels the electrostatics of the lowest level
  between the model and the environment exactly for a force field with fixed point charges;
  GFN-FF's electrostatics (EEQ charges at every structure, damped at short range) differ, so
  with GFN-FF as the lower level the cancellation is approximate.
- **Hessians** of the embedded surface are central differences of its gradient (6N
  gradients), since the model systems depend on the positions of every charge.
- **Microiterations** relax the environment on the whole embedded surface (the higher level
  sees the charges move, so every inner step computes it too) instead of on the lowest level
  alone.
- A periodic cell is refused (the charges around a cluster in a cell are not set up).

## Microiterations (on by default; `-oniom_opt micro=off`)

```text
optmain claisen_in_water.xyz -os gfn1xtb -oniom 1-14 none gfnff -order 1 -fc 5
```

With `-oniom`, `optmain`, `ircmain`, and `nebmain` move only the core, and the environment is
relaxed to a minimum before every evaluation (Vreven et al., J. Comput. Chem. 2003, 24, 760):
- **Core:** the outermost model region and the outer atoms of its links.
- **Environment:** every other atom. With mechanical embedding the model systems do not see
  it, so its relaxation needs only the lowest level on the whole structure (with electronic
  embedding the whole surface).
- **The environment moves with the core.** Where rigid motions of the whole system are
  symmetries (no fixed atoms, no cell, both levels invariant), the environment of the nearest
  core structure evaluated so far is moved by the rigid motion that fits that core best onto
  the new one before it is relaxed, and a core that is a rigid motion of a relaxed one takes
  the moved environment without a relaxation. The optimizers recentre and rotate the core at
  every step (MultiOptPy's Kabsch step) and the IRC puts its centre of mass at the origin;
  before this, those moves pushed the core against an environment that stayed where it was,
  and the relaxation that followed could end in another arrangement (the IRC from a
  transition state of vinyl alcohol with two waters, converged to 4e-4 Hartree/Bohr, saw a
  gradient of 1.1e-3 at its start and refused it; the input minimum took another arrangement
  of the water, 1 kcal/mol lower by chance).
- **Every relaxation runs to a minimum.** It starts from the environment of the nearest core
  structure evaluated so far (the last one in an optimization) and runs until its gradient
  meets the thresholds of the forces of the run itself times `micro_tolerance_factor` (1 by
  default): for `optmain` and the commands built on it the largest force and the RMS force of
  the criteria of the run (3e-4 and 2e-4 Hartree/Bohr, with `-tcc` 1.5e-5 and 1e-5), for
  `ircmain` the thresholds at which a side of the path ends, for `nebmain` the tolerance of the
  forces of the chain. A factor below 1 does not take the tolerance below 1e-5 (RMS 6.7e-6)
  unless the thresholds of the run are below that themselves: OpenMM in double precision
  relaxes no further than 3e-6 to 5e-6 (below), and an IRC of chorismate mutase asked for
  1.5e-6 took 5,000 inner iterations per relaxation.
  The core is converged against a gradient that the relaxation of its environment makes
  noisy: a relaxation stops anywhere below its tolerance, and what is left in the
  environment moves the core gradient by `H_xy H_yy^-1 g_y`, along the soft modes of the
  environment far more than `g_y` itself. Measured on chorismate mutase (2CHT, the guess of
  the transition state of AutoTS, the free environment of 662 atoms perturbed by 0.05
  Angstrom at random five times and relaxed again at the same core; the core gradient is
  OpenMM's at the five relaxed structures, which differ only there):

  | `micro_tolerance_factor` | largest gradient of the environment | inner iterations | spread of E* (Hartree) | core gradient, largest difference to the first start (Hartree/Bohr) |
  |---|---|---|---|---|
  | 1 (the default) | 2.4e-4 to 3.0e-4 | 107 to 149 | 5.1e-6 | 5.8e-5 |
  | 0.1 | 2.8e-5 to 3.0e-5 | 158 to 243 | 4.0e-7 | 5.1e-6 |
  | 0.03 | 8.7e-6 to 9.0e-6 | 200 to 290 | 4.0e-7 | 2.9e-6 |
  | 0.01 | stopped at 3e-6 to 5e-6 | 347 to 680 | 1.3e-6 | 6.0e-6 (pairwise) |

  One start of the five (the third) relaxed into another minimum of the environment, 1.4e-4
  Hartree higher with a core gradient 2e-4 Hartree/Bohr apart, at every tolerance; the
  spreads above are those of the four others (with all five they are 1.4e-4 and 2e-4
  whatever the tolerance). At 1 the noise of the core gradient is a fifth of the threshold
  of the forces, and the quasi-Newton updates of the core read it as curvature; at 0.1 it is
  a sixtieth, at the cost of about half as many inner iterations again; tighter than that
  OpenMM's gradient in double precision is at its floor (the relaxations to 3e-6 stopped
  where the energy no longer fell, which the log says). `micro_tolerance=VALUE` gives a
  threshold of its own for the largest gradient component. With the inertia of the saddle
  searches kept (below) the transition state of chorismate mutase converged at 1 as at 0.1,
  with fewer gradients, and at 0.1 the mapper found another transition state in its test of
  vinyl alcohol with two waters; 1 stays the default, and `micro_tolerance_factor=0.1` is
  the setting for a search whose core gradient is too noisy to converge (the proxy below). A relaxation that ends above its
  tolerance (stopped where its energy no longer falls, a lowest level at its precision) says
  so in the log with the gradient it reached. The low layer is not
  mode-killed by default: the core is, by the mode kill of its optimizer, and the
  environment is relaxed to its thresholds without a check of its curvature (user,
  2026-11-20: on chorismate mutase with OpenMM's Reference platform the check took 40
  Hessian-vector products, 80 gradients of 5696 atoms, after every relaxation).
  `micro_mode_kill=N` turns the check on for the first relaxation and every Nth: the lowest
  curvature of the environment is checked, and a negative-curvature mode is followed
  downhill and the environment relaxed again (mode kill, `crate::pes::microiter`), so that it
  does not stop at a saddle point of the cheap level. The check does not build the Hessian of
  the environment: the lowest curvature comes from Davidson's method on Hessian-vector
  products from central differences of the gradient (at most 40 products, two gradients
  each), instead of two gradients per environment coordinate; `micro_mode_kill=off` says the
  default. The relaxation runs for as many iterations as it takes. It is a
  limited-memory BFGS whose pairs are kept from one relaxation to the next (the environment
  changes little between outer steps, so the next relaxation starts with the curvature the
  last one learned).
- **Where a line search fails, the relaxation goes on with the gradient alone, then with
  FIRE.** A line search fails where the surface does not follow its gradient: at a
  discontinuity of the lowest level (a neighbour list of a force field that changes, a kink)
  the gradient says downhill and no step along it lowers the energy, or only one at the
  round-off; near convergence a lowest level in single precision has energies whose noise
  is larger than what a step lowers. A quasi-Newton method with a line search then stops, or
  lowers the energy by amounts at the round-off without end (an AFIR path in water stood at
  one relaxation for that reason: its gradient stayed at 3e-5 Hartree/Bohr while the energy
  fell by 1e-9 per hundred steps). The relaxation goes on in two stages from the point
  reached, to the same thresholds:
  1. the limited-memory BFGS with a line search on the gradient alone (the steps of its
     model, the energies not compared; Cartesian environment and the limited-memory BFGS
     only);
  2. where that line search fails as well, the FIRE of the optimizer registry (`-opt fire`;
     it moves by the forces alone), without the pairs of the model that failed. The command
     hands the optimizers to the relaxation with its settings (the surfaces are a layer
     below them; `opt/micro.rs`).

  A relaxation that FIRE does not converge either (its largest gradient component not
  falling to 0.9 of the lowest so far for 5000 evaluations; its best point is kept) is
  counted and reported at the end with the gradient it reached. A relaxation of the
  limited-memory BFGS whose energy (beyond the round-off) and largest gradient component (to
  0.9 of the lowest so far) do not fall for 100 accepted steps stops short of the threshold
  in the same way.
- **What is written.** The stream and `mopt.log` show the core as the optimizer moves it. Of
  every relaxation they show the energy and the gradient (largest component and RMS) every
  50 inner iterations, the end of every descent (converged or not, after how many inner
  iterations, with which gradient), a switch to FIRE, and with `micro_mode_kill=N` the test
  for a saddle point at its end: the lowest curvature of the environment and whether it is a
  minimum, or a saddle point whose mode is then followed downhill. The coordinates of the environment are not
  written step by step (a run has as many relaxations as evaluations); the files of the run
  get the whole structures at its end.
- **The surface of the optimizer:** `E*(core) = min over the environment of E_ONIOM`. Its
  gradient is the ONIOM gradient of the core at the relaxed environment.
- **Hessians** (`-fc`, `-freq`, the IRC): those of the relaxed surface,
  `H_xx - H_xy H_yy^-1 H_yx` (the effective Hessian, `micro_hessian=effective`, the
  default), by one of three routes, chosen when the surface is set up and told in the log:
  - Every level has an analytic Hessian and its matrices fit in the memory a Hessian may
    take: the analytic Hessian of the whole system at the relaxed environment, reduced to
    the core with the Cholesky factor of `H_yy` (positive definite at a relaxed
    environment; the pseudo-inverse where it is not). The matrices take about
    `8 (3 N^2 + 2 m^2)` bytes for `N` coordinates of the whole system and `m` of the
    environment: the Hessian of the whole system while the terms are summed, then the block
    of the environment and its factor (36 GB for 10,000 atoms nearly all in the
    environment, 0.36 GB for 1,000).
  - Mechanical embedding with analytic Hessians of the model systems (GFN1-xTB on the core
    with OpenMM or GFN-FF around it): the model systems depend on the core alone (the
    outermost region and the outer atoms of its links), so their analytic Hessians give
    their part of `H_xx`, and central differences of the gradient of the lowest level along
    the core coordinates give the rest of `H_xx` and the coupling `H_yx`. The higher levels
    are computed once per Hessian instead of at `2n` displaced cores.
  - Otherwise central differences of the ONIOM gradient along the core coordinates give
    `H_xx` and `H_yx` together (2 x 3 gradients per core atom instead of per atom).

  On the last two routes the response of the environment comes from the dense block `H_SS`
  of its responding coordinates (`micro_hessian_solve=auto`, the default) wherever its
  gradients cost no more than the products the conjugate gradients below may take (`2 m`
  gradients against `n` solves of at most 100 products of two gradients each) and it fits in
  the memory: the analytic Hessian of the lowest level where it has one (GFN-FF) and the
  whole system fits, central differences of its gradient along every responding coordinate
  (step 1e-3 Bohr) otherwise. The block is decomposed, inverted over its positive
  curvatures (the curvatures that are not positive are left out and said in the log, where
  they are a saddle point of the environment rather than noise; a check of stability is not
  judged then), and `H_xS H_SS^-1 H_Sx` follows exactly. On chorismate mutase (606
  responding coordinates, 72 of the core) that is 1,212 gradients of OpenMM in 26 to 30 s,
  where the conjugate gradients took 7,300 products (6.5 minutes) and ended at relative
  residuals of 0.2 to 0.95; the block of the guess of AutoTS relaxed to 3e-4 had a
  curvature of -1.2e-3 Hartree/Bohr^2 (the environment stood on a saddle point that its
  relaxations did not leave, its gradient being below their tolerance), which conjugate
  gradients never met (their lowest curvature was +2.3e-3); relaxed to 3e-5 the environment
  left it, and the block ran from +2.9e-4 to 1.62 Hartree/Bohr^2.
  `micro_hessian_solve=dense` always takes the block, `cg` the conjugate gradients.

  The dense response of the newest Hessian of the core also corrects every relaxation after
  it for the gradient it left in the environment (`micro_correct=on`, the default): to first
  order the minimum of the environment lies `-H_SS^-1 g_S` away, and there the energy is
  lower by `g_S^T H_SS^-1 g_S / 2` and the core gradient differs by `-H_xS H_SS^-1 g_S`. The
  correction is taken where that move stays within 0.05 Bohr in every coordinate (else the
  log says that the residual is beyond the reach of the quadratic model, and the gradient is
  taken as it is). Every evaluation writes a line with `E*`, the core gradient, the largest
  and RMS gradient left in the environment, and the correction. On chorismate mutase with
  the environment relaxed to 3e-4 the corrections reached 3.4e-4 Hartree/Bohr in the core
  gradient, as much as the gradient itself (the response there is the one of a saddle point
  of the environment); relaxed to 3e-5 they were 4e-6 to 5e-6, the size of the noise of the
  table above.

  Between exact Hessians the core Hessian is updated (block Bofill by default for a saddle
  point) from the change of the core gradient over a step, whose noise is that of two
  relaxations: on chorismate mutase the pairwise RMS difference of the core gradient over
  the four relaxations of the table above that reached the same minimum was 1.27e-5
  Hartree/Bohr per coordinate at a tolerance factor of 1 and 8.6e-7 at 0.1, so the noise of
  the change over a step is about 1.1e-4 and 7e-6 for the 72 coordinates of the core, while
  the mismatch `|y - H s|` that the updates act on was 1.3e-4 to 3.9e-4 (factor 1) and
  1.5e-4 to 3.8e-4 (factor 0.1) over steps of 0.1 Bohr (the debug level of the log gives it
  at every update). At 1 a third to all of the update was noise, and the soft eigenvalues of
  the core (3e-4 Hartree/Bohr^2) changed sign between exact Hessians; at 0.1 it is a
  twentieth or less over those steps. Near convergence the steps are short (0.01 to 0.03
  Bohr) and the noise of the change of the gradient is a curvature of 2e-4 to 7e-4 along
  them, as large as the soft eigenvalues themselves: at 0.1 too, single updates turned a soft
  mode negative (-1.7e-4 to -1.4e-3) after the gradient had fallen to 1e-5, the step ran along
  it to the trust radius, and the force rose again to 3e-4 (three times in the 43 iterations
  of the search below). A saddle search therefore skips an update that gives its Hessian
  more negative curvatures than the saddle order and its last exact Hessian had (with an
  exact Hessian that has two, updates that keep two are taken; the debug level of the log
  says which were skipped). This needs no estimate of the noise and holds for every saddle
  search, also without ONIOM; on chorismate mutase it skipped 6 to 12 updates per search and
  took the searches to convergence in 21 to 40 iterations (the table below). Mode following by
  overlap (`-opt mf_rsirfo_*`) is not made the default either: the followed mode of
  chorismate mutase (curvature -0.11 to -0.12) was the lowest at every iteration, the next
  one at -1.1e-3 or above, so that following by overlap picks the same mode.

  Where the response is not dense, `H_yy^-1` is applied by conjugate gradients on
  Hessian-vector products of the lowest level (central differences of its gradient, step
  1e-3 Bohr, since forward differences along 1e-4 Bohr were off by 0.15 percent),
  preconditioned by the limited-memory BFGS of the
  relaxations on top of the factorized model of the environment that the relaxations use
  (Lindh with D3 by default): no matrix of the environment is built, and the memory is that
  of a few `m x n` blocks. A solve stops where its residual runs away (products dominated by
  the noise of the differences) and returns its solution of the lowest residual; a Galerkin
  start from earlier solutions is taken only where it lowers the residual, and only solutions
  that converged serve as starts. The response that the effective Hessian takes off is the
  Galerkin one in the span of the solutions: with `V` an orthonormal basis of them,
  `T = V^T H_yy V` (one more product per column of `V`) and `B = V^T H_yx`, it is
  `B^T T^+ B`, positive semidefinite and below the exact `H_xy H_yy^-1 H_yx` whatever the
  solves reached, and equal to it where they converged. A solve that stopped early thus
  leaves part of the response out (the effective Hessian errs towards the core block) but
  puts no curvature into the core that it does not have; the coupling assembled column by
  column from unfinished solves had neither property, and on chorismate mutase the mode kill
  of the core was displaced again and again along curvatures its energy did not have.
  That Galerkin Hessian is the one for the steps of the optimizers. A Hessian for a check of
  stability (the mode kill of an optimizer, the minima and saddle points of the mapper, the
  start of an IRC: `EvalRequest::for_stability`) does not take it: its solves run to
  convergence (a relative residual of 1e-3, with as many products as the environment has
  responding coordinates at most), the response is assembled from them as they are, and
  when an iterative solve fails and the responding environment has at most as many
  coordinates as the core, the mapper can check the response by central differences
  and direct Cholesky solves at the same residual tolerance. If the response cannot be
  verified, no Hessian is returned and the log says that the stability cannot be judged, with
  how the solves ended (at a direction without positive curvature, at the limit of products,
  or where the residual or the preconditioner ran away) and the lowest curvature they met:
  the optimizer then accepts the structure unchecked, the mapper warns for an unchecked input and leaves an unverified IRC endpoint unknown.
  An IRC takes the direction at its start from the
  Hessian of the steps, with a warning that the order of the transition state is not
  confirmed. An environment held only loosely (a water at the edge of the core, curvatures of
  1e-6 Hartree/Bohr^2 and below along its motions) is where the solves end so. Every line of a Hessian of the core in the log gives the
  lowest wavenumbers (with the masses of the core) of the core block, of the effective
  Hessian with the response column by column, and of the one taken. A response that no positive definite `H_yy` gives
  (`H_xy H_yy^-1 H_yx` not positive semidefinite, or beyond the largest curvature of the core
  block by more than 1 Hartree/Bohr^2) is left out: the Hessian of the core is then the core
  block alone, and the log says why. Every Hessian of the core writes a line to the log with
  its solves, their products and how they ended, the largest residual and response, and the
  extreme curvatures of the core block and of the effective Hessian. On chorismate mutase
  (24 core atoms, 662 free atoms around them, 5010 fixed) with OpenMM's CPU platform and
  the pairs alone as the preconditioner the solves diverged (residuals of 1e24, curvatures
  of the effective Hessian of 1e21) and the minimization climbed after its mode kill. On
  the Reference platform with the model under the pairs the 72 solves ended at relative
  residuals of 0.92 at most after 100 products each (7,200 gradients of OpenMM, 6.5
  minutes), and the effective Hessian had curvatures from 1.4e-4 to 1.55 Hartree/Bohr^2
  against 1.3e-3 to 1.55 for the core block: an unfinished solve leaves part of the
  response out (`b . z` grows towards `b . H_yy^-1 b` as conjugate gradients go on), so the
  effective Hessian errs towards the core block, not beyond the exact one.
  With the check, and the steps of the trust-region problem in RS-I-RFO
  ([numerical safeguards](migration_safeguards.md#restricted-step-with-negative-curvature-and-almost-no-gradient-along-it))
  it does not. `micro_hessian_memory=GiB` sets the memory a Hessian may take (by default
  half of the free memory of the machine, `MemAvailable`; 2 GiB where that is not known);
  a Hessian that does not fit on any route is refused with the memory it would take. The
  size of the whole system alone decides nothing else: the factorization costs `m^3 / 3`
  operations (10^12 at 5,000 atoms), and no inverse is formed, so the error of the solve is
  of the order of the round-off times the condition number of `H_yy`.
  `micro_hessian_shell=R` (Angstrom) lets only the atoms of the environment within `R` of an
  atom of the core respond, `H_xx - H_xS H_SS^-1 H_Sx`, and holds the rest: on every route
  the blocks, the rows of the coupling, and the conjugate gradients are those of the shell.
  The coupling of an atom of the environment to the core falls off with its distance, and
  so does what it adds to the curvature of the core. For the choice of `R`, the description
  of the surface (the `Calculator:` line of every command) gives how many free atoms of the
  environment lie within 4, 5, 6, 8, and 10 Angstrom of an atom of the core, with
  minimum-image distances in a periodic cell, and how many respond
  (`RelaxedEnvironment::shell_counts` from Rust). On chorismate mutase (24 core atoms, 662
  free atoms around them) they are 84, 149, 211, 406, and 600: `micro_hessian_shell=6` takes
  633 coordinates of the 1986, and the solves of a Hessian for a check of stability, which
  run to convergence, are that much shorter.
  `micro_hessian=held` gives the core block `H_xx` with the environment held at its relaxed
  position, and leaves the coupling to the quasi-Newton updates. Its calculator description
  reports zero responding atoms alongside the shell counts. Measured on a transition
  state of the Claisen rearrangement in water (GFN1-xTB:GFN-FF, 14 core atoms, 60 environment
  atoms, `-order 1 -fc 5 -tcc -tr 0.2`, from the guess of AutoTS): the effective Hessian
  converged in 15 iterations and 39,558 gradients, the held one in 22 iterations and 67,382
  gradients, both at a saddle point with one imaginary frequency (-417 and -412 cm^-1).
- **A buffer of the environment in the core** (`micro_core_shell=R`, Angstrom; off by
  default): the free atoms of the environment within `R` of an atom of the model region are
  moved by the optimizer with the core (the log says how many), and the rest is relaxed
  around them. The stationary points are those of the whole surface whatever the core; the
  optimizer then has the stiff coupling of the nearest residues in its own Hessian instead of
  in the response of the environment.
- **An environment that goes to another minimum.** A relaxation that moves an atom of the
  environment by more than 0.5 Angstrom and ten times farther than the largest step of an
  atom of the core took the environment to another local minimum: the surface of the core is
  not continuous there, and the Hessian of the core and its updates from before belong to the
  other minimum. The log says so in one line, the dense response of the last Hessian is not
  used for the corrections after it, and the end of the run counts such relaxations. This is
  detection only: the optimizer is not told (see the measurements below for how often it
  happened).
- **Rigid motions of the core.** With nothing holding the environment (no `-fix` there),
  `E*` does not change when the core moves rigidly, because the environment follows it as
  a whole. These are zero modes of the effective Hessian, and they are projected out of the
  steps as for an isolated molecule. Without that projection a transition-state search
  stepped along them (eigenvalues of 1e-5 next to -0.049 for the Claisen rearrangement in
  water), and the next relaxation had to rotate the whole environment by an almost flat
  collective motion. With fixed atoms in the environment `E*` is not invariant, and nothing
  is projected.
- **Options of the relaxations** (`-oniom_opt KEY=VALUE`):
  - `micro_optimizer`: the minimizer. `lbfgs` (the default) is the limited-memory BFGS
    above. `tn` is a truncated Newton method: each step solves `H p = -g` by conjugate
    gradients on Hessian-vector products (one gradient each), preconditioned by the same
    pairs, to a residual that shrinks with the gradient. Any method of `-opt` (such as
    `rsirfo_bfgs`) runs with a dense Hessian of the environment, for environments of up to
    500 atoms.
  - `micro_coord=fragments`: the limited-memory BFGS takes its steps in the translation,
    rotation, and internal coordinates of each fragment of the environment (a water, a
    ligand; fragments of more than 30 atoms are cut into pieces), in the manner of the
    hybrid delocalized internal coordinates of Billeter, Turner, and Thiel (Phys. Chem.
    Chem. Phys. 2000, 2, 2177). The Wilson matrix is block diagonal, so the transformations
    cost the sum over the fragments. `cart` (the default) keeps the Cartesian coordinates.
    Not in a periodic cell.
  - `micro_predict=on`: each relaxation starts from the predicted response of the
    environment to the step of the core, `y + R dx`, where it is lower in energy than the
    last relaxed environment. `R = -H_yy^-1 H_yx` comes from the effective Hessian when one
    was computed and is corrected by Broyden's update after every relaxation.
  - `micro_adaptive=ETA`: the tolerance of a relaxation is ETA times the largest core
    gradient of the last evaluation, between `micro_tolerance` and 1e-3 Hartree/Bohr, so
    that the relaxations far from convergence stop earlier. At the end of `optmain` the
    environment of the final structure is relaxed to `micro_tolerance`, and the change of the
    energy and the core gradient is reported. Off (0) by default.
  - `micro_precondition=NAME` (a model Hessian of `-modelhess`; `lindh2007d3`, the default,
    the best of the measurements below; `none`: without): the block of the
    environment of that model, built at the free atoms (core and environment) and the fixed
    atoms within 4 Angstrom of the environment (their bonds, angles, and torsions hold the
    environment as they do on the surface; without them an atom bonded to the fixed frame
    looks free), shifted by `micro_precondition_shift` (default 1e-3 Hartree/Bohr^2) so that
    it is positive definite, is factorized and kept while no free atom has moved 0.3 Bohr from
    where it was built. Up to 16 factors are kept at once (within 512 MB; the least recently
    used is dropped), so that the images of a chain each keep theirs instead of building one
    again whenever the next image is relaxed. On chorismate mutase (686 free atoms, 641 fixed
    neighbours, 1986 variables of the environment) the model takes 12 to 22 s and the factor
    0.4 s in a run; the model of the free atoms alone takes 3.6 s by itself (`lindh2007d3` of
    `-modelhess`: 3.0 s of Lindh terms, 0.7 s of the D3 Hessian, `bench_microiterations
    models`). The 89 to 280 s that the logs of AutoTS on it gave for every build of that model (231
    builds in the chain of 14 images with one slot, 8.8 hours in all, and 151 to 233 s in its
    search for the transition state) are not the cost of the model: the same build measured
    by itself took 3.6 s; those logs do not say what the time went to. An
    environment of more than 6000 variables has none (the factor is a dense matrix), and a
    model that cannot be built is said once in the log and the relaxations go on without it.
    A relaxation that fails with it, stops short of the threshold, or changes the bonds of the
    environment (a bond of the start, closer than 1.25 times the sum of the covalent radii,
    ends farther than 1.5 times; or a pair ends closer than 1.1 times) is made again from the
    same start without it; after three of these, if they are a quarter of the relaxations, the run
    goes on without the preconditioner (both are in the log). The steps that the
    preconditioner gives can lead elsewhere than those without it: with electronic embedding
    of the methyl group of ethanol they reached a structure where a point charge sat on an
    atom of the higher level (the energy of electronic embedding has no lower bound there),
    and with mechanical embedding a relaxation stopped short at a saddle point.
    Its inverse is the initial matrix of the limited-memory BFGS of the relaxations
    (Nocedal and Wright, section 7.2) and the correction of the search for the lowest
    curvature (a preconditioned inverse iteration; Knyazev, SIAM J. Sci. Comput. 2001, 23,
    517), as model Hessians precondition the optimization of materials (Packwood et al.,
    J. Chem. Phys. 2016, 144, 164109).
  - `micro_curvature=guided|plain`: the search for the lowest curvature starts from the mode
    of the last check and uses the pairs of the relaxation and the preconditioner (`guided`,
    the default), or starts from a fixed vector without either (`plain`, as before).

  What the relaxations cost, measured on the TS optimization of the Claisen rearrangement in
  water (GFN1-xTB for the 14 atoms of the substrate, GFN-FF for 20 waters; the environment of
  every frame relaxed in turn at the cores of the trajectory, `bench_microiterations`):

  | Frames, threshold | Setting | Gradients | Inner iterations | Checks at their limit |
  |---|---|---|---|---|
  | 11, 1.5e-5 | none | 1181 | 268 | 11 of 11 |
  | 11, 1.5e-5 | `micro_precondition=lindh` | 500 | 60 | 3 of 11 |
  | 11, 1.5e-5 | `micro_precondition=lindh2007d3` | 304 | 40 | 1 of 11 |
  | 11, 1.5e-5 | `lindh`, shift 1e-2 / 1e-4 | 563 / 470 | 66 / 53 | 5 / 1 of 11 |
  | 71, 3e-4 | none | 5838 | 14 | 71 of 71 |
  | 71, 3e-4 | `micro_precondition=lindh2007d3` | 2371 | 14 | 5 of 71 |

  Without a preconditioner the check of the curvature is most of the cost (880 of the 1181
  gradients, 5680 of 5838) and never converges: after its 40 Hessian-vector products the
  residual is 1e-4 to 4e-4 against curvatures of 1e-5 to 4e-5, because the soft
  intermolecular modes of the waters lie among their stiff bonds and angles. The energies of
  the relaxed surface agree with those without a preconditioner to 3e-6 Hartree at the
  threshold 1.5e-5 (5e-5 Hartree at 3e-4), within the tolerance of the relaxations. The end of
  a run reports the gradients of the relaxations, the share of the checks, and the checks that
  ended at their limit, and every check its products and residual in the log. No relaxation
  of these runs fell back to FIRE.
  - Every relaxation (with or without the preconditioner) refuses a trial structure with an
    atom of the environment within 0.5 Angstrom of another atom: it has no energy, and the
    line search shortens the step. No two atoms of a molecule are that close (H2 is 0.74
    Angstrom), and a surface with a hole cannot pull an atom into it. A start that already
    has atoms that close is relaxed as it is. The end of a run gives the relaxations made
    again and the structures refused.
- **Transition states on the relaxed surface, measured.** Chorismate mutase (2CHT,
  GFN1-xTB for the 24 atoms of the substrate, OpenMM's Reference platform for the 5672 of
  the protein, 662 of them free, `micro_hessian_shell=6`, `-order 1 -fc 5`, the default
  criteria), from the guess of AutoTS (the highest image of its chain); and a proxy that
  shows the same behaviour in minutes (the SN2 transition state Cl- + CH3Cl, GFN1-xTB, in
  114 waters of GFN-FF, the 68 outside 7 Angstrom fixed, started 0.12 Angstrom off along
  the reaction coordinate; the whole route there, so the response is always dense):

  | System, variant | Iterations | Cost | Wall time | Converged | Lowest wavenumber (cm^-1) | Koopman modes (last 10 windows: of 100, |lambda| >= 1 / negative real part) |
  |---|---|---|---|---|---|---|
  | 2CHT, before (the log of AutoTS: conjugate gradients, relaxations to 3e-4) | 75, stopped | 15 Hessians of 7,300 products each, besides the relaxations | not in the log | no (the steps always at 0.1 Bohr) | -510 | not run |
  | 2CHT, tolerance factor 1, dense response, correction | 25, stopped (memory of the machine) | | | not by then (11 changes of the number of negative curvatures in 25 iterations) | -512.8 | not run |
  | 2CHT, factor 0.1, dense, correction | 43 | 9 Hessians (10,998 gradients of OpenMM in their blocks), 942 inner iterations | 19 min with `-km` | yes | -510.8 | 36 / 40 |
  | 2CHT, the same with `-tr 0.3` | 32 | 7 Hessians (8,610), 1,027 inner iterations | 17 min with `-km` | yes | -510.6 | 20 / 45 |
  | 2CHT, factor 0.1, the inertia of updates kept | 40 | 8 Hessians (9,762), 945; 19,283 gradients in all | 8.6 min | yes | -510.4 | not run |
  | 2CHT, the same with `-tr 0.3` | 25 | 5 Hessians (6,126), 718; 11,881 in all | 5.9 min | yes | -510.8 | not run |
  | 2CHT, factor 1, inertia kept, `-tr 0.3` | 21 | 5 Hessians (6,114), 89; 6,979 in all | 4.5 min | yes | -510.9 | not run |
  | 2CHT, factor 1, inertia kept (the defaults) | 41 | 9 Hessians (10,992), 112; 12,735 in all | 8.2 min | yes | -510.8 | not run |
  | proxy, factor 1 (before the inertia was kept) | 99 | 1,122 | 245 s | yes | | |
  | proxy, factor 0.1 (before the inertia was kept) | 83 | 1,703 | 357 s | yes | | |
  | proxy, factor 0.1, `-opt mf_rsirfo_block_bofill` | 83 | 1,703 | 307 s | yes (the same steps) | | |
  | proxy, factor 0.1, no correction (before the inertia was kept) | 200 | | | no | | |
  | proxy, factor 0.1, no correction, inertia kept | 104 | 2,043 | 284 s | yes | | |
  | proxy, factor 0.1, inertia kept | 73 | 1,594 | 260 s | yes | | |
  | proxy, factor 1, inertia kept | 200 | 1,345 | 212 s | no | | |

  Every search of chorismate mutase ended at the same saddle point (E* -49.603674 to -49.603676
  Hartree, one imaginary wavenumber of -510 cm^-1). Before, the searches did not converge: the
  noise of the core gradient (relaxations to the thresholds of the run, 5.8e-5 Hartree/Bohr)
  and updates over short steps turned the soft modes negative, and the steps ran along them.
  Relaxations to a tenth of the thresholds converged in 43 and 32 iterations; keeping the
  inertia of the updates converged in 40 and 25, and then also with relaxations to the
  thresholds themselves (41 and 21 iterations, with fewer gradients: 112 against 945 inner
  iterations). The tighter tolerance gives besides a core gradient ten times less noisy at the
  saddle point, corrections of the residual of 5e-6 instead of 3e-4, and an environment that
  leaves the saddle point of the environment that the guess of AutoTS stood on (at the
  thresholds of the run its block had a curvature of -1.2e-3 Hartree/Bohr^2 to the end), at
  1.5 times the gradients; it is not the default because of the mapper (above). The searches spend their first 20 to 30 iterations
  moving the substrate in its pocket at the trust radius with a ratio of actual to predicted
  change near 1 (the energy falls by 2.6e-3 Hartree from the guess), so that a larger trust
  radius (`-tr 0.3`) saves a third of the iterations. On the proxy the correction of the
  residual mattered: without it (`micro_correct=off`, factor 0.1) the search had not converged
  after 200 iterations where it converged in 83 with it, and with the inertia kept it took 104
  iterations against 73. `micro_core_shell=3` (7 atoms of water moved with
  the core) took 135 iterations and `micro_core_shell=4` (36 atoms) did not converge in 200,
  against 73 without a buffer (all three with the inertia kept), so the buffer stays off. With
  the inertia kept the proxy converged at a tolerance factor of 0.1 (73 iterations) and not at
  1 (200 iterations): there the tighter tolerance decided.
- **Verified saddle points: one surface for the search and for what follows it.** The IRC
  started from the transition state of chorismate mutase above (the search of factor 0.1 with
  `-tr 0.3`) found two negative curvatures in its own Hessian (`-9.877e-3` and `-2.735e-5`
  Hartree/(Bohr^2 amu), mass-weighted) where the search had one, and ran a one-sided meta-IRC.
  The two Hessians were built by the same code; what differed:

  | | the search (its last exact Hessian) | the start of the IRC |
  |---|---|---|
  | geometry | iteration 21, four iterations before convergence (the last four were updates the inertia guard kept at one negative curvature) | the converged structure |
  | environment | relaxed to 3e-5 (factor 0.1) from that of the iteration before | relaxed to 1e-5 (`micro_tolerance=1e-5`) from that of the file of the search |
  | request | a Hessian for the steps | a Hessian for a check of stability (identical matrix on the dense route; refused where the block has a non-positive curvature) |
  | core block | analytic Hessian of the model system (GFN1-xTB) and central differences of OpenMM along the core (5e-3 Bohr) | the same |
  | response | the block of 618 to 621 responding coordinates (6 Angstrom) by central differences (1e-3 Bohr), inverted over its positive curvatures | the same |
  | correction of the residual | on the gradient only; the Hessian is taken at the environment as relaxed | the same (no dense response yet at the first point) |
  | mass weighting, projection | Cartesian count (curvatures below -1e-6), no projection (fixed frame) | mass-weighted count (below -1e-8), no projection (fixed frame); the inertia is the same in both (Sylvester) |
  | symmetrization | `(H + H^T) / 2` | the same |

  The Hessian of the core at equal settings (a single point relaxed to the tolerance and taken
  with the dense response; mass-weighted eigenvalues, Hartree/(Bohr^2 amu)), at the converged
  structures of the search above (E4) and of the search with the defaults (E6):

  | structure, environment | E* (Hartree) | lowest curvature of the block | 1st | 2nd | 3rd |
  |---|---|---|---|---|---|
  | E6, as written (3e-4) | -49.6036713 | +2.37e-4 | -9.873e-3 | +4.07e-5 | +7.15e-5 |
  | E6, relaxed to 3e-5 | -49.6036756 | +2.36e-4 | -9.871e-3 | +4.70e-5 | +6.42e-5 |
  | E6, relaxed to 1e-5 | -49.6036763 | +2.54e-4 | -9.871e-3 | +4.20e-5 | +6.47e-5 |
  | E4, as written (3e-5) | -49.6036748 | **-1.66e-5** | -9.877e-3 | +3.24e-5 | +7.86e-5 |
  | E4, relaxed to 1e-5 (= the IRC) | -49.6036749 | +2.15e-4 | -9.877e-3 | **-2.73e-5** | +7.29e-5 |

  At five environments of each structure perturbed by 0.05 Angstrom and relaxed again to 1e-5:
  E4 three times in one minimum of the environment (E* -49.6036754, 5e-7 below that of the
  IRC; second curvature +4.45e-5, +4.38e-5, +4.38e-5) and twice on saddle points of the
  environment that the relaxations stopped at (lowest curvature of the block -4.6e-4 and
  -3.8e-4; second curvature -9.0e-6 and +3.5e-5); E6 three times in its minimum (+4.155e-5,
  +4.155e-5, +4.186e-5) and twice in another one with a soft mode of the environment (block
  +1.03e-5, second curvature -1.10e-3). The second mode is a motion of the substrate as a
  whole in the pocket: 74 to 96 percent of it lies in the six rigid motions of the core
  (mass-weighted), its overlap with the reaction mode is below 2e-3, and the gradient along it
  at the structures is 1e-7 to 1e-6 (mass-weighted; the structures are stationary along it).
  Within one minimum of the environment the curvature repeats to 7e-7, a fiftieth of itself;
  its sign is not noise. It is the minimum of the environment that decides it: the
  environment of the search stood on a saddle point of its own (-1.66e-5), the dense response
  left that curvature out silently, and the tighter relaxation of the IRC slid from it into a
  neighbouring minimum in which the structure is a saddle point of second order, 5e-7 Hartree
  above the one in which it is of first order. On the proxy (SN2 in water, the structure of
  the search re-evaluated on a GFN-FF topology of its own, so not stationary there) the second
  curvature (-3.3e-5, a rigid motion of the solute) has the same sign in every minimum of the
  environment (eight relaxations; -2.1e-5 to -3.3e-5), and no saddle point of the
  environment was met.

  What follows from this:
  - **One tolerance for every check.** Every Hessian for a check of stability (the
    verification of a converged saddle search, the start of an IRC, the stationary points of
    the mapper, the relaxation of `freqmain`) is taken with the environment relaxed to
    `micro_check_tolerance` (1e-5 by default, the floor above, never looser than the run's
    own); the relaxed environment is kept for the structure and written with it, so that the
    command that follows starts from the environment that was judged. The log of the search
    and of the IRC give the tolerance.
  - **No Hessian on a saddle point of the environment.** Where the block of the responding
    environment (dense or analytic route) has a curvature below minus the threshold of the
    mode kill of the environment (1e-6 Hartree/Bohr^2), the environment is moved off along
    that mode by 0.1 Bohr to the lower side, relaxed again, and the Hessian taken again (up to
    three times; the log says so, and a check is not judged on an environment that stays on a
    saddle point). The analytic route inverts over the positive curvatures as the dense one
    does (it took the pseudo-inverse over all of them before).
  - **The error of the curvatures of a check.** The environment of a check is moved to the
    minimum of the quadratic model of its block, `-H_SS^-1 g_S`, and the Hessian taken again;
    the change of each mass-weighted eigenvalue is its error bar, and where the count of
    imaginary modes is not the same at both ends of the bars this is repeated (up to three
    times). On chorismate mutase the step was 4.1e-4 Bohr and the error 9e-10 against a second
    curvature of +3.38e-5.
  - **The search is verified.** A saddle search with microiterations asks at convergence for
    such a Hessian (also where the converging iteration had one for its steps), counts its
    imaginary modes by the one criterion (`opt::mode_kill`), and leaves a surplus mode by the
    mode kill of the core and goes on, at most ten times; a structure that keeps one ends
    without convergence. The check is accepted on the criteria of the forces: the step its
    exact Hessian proposes along the soft modes (4.7e-3 Bohr at a force of 1.2e-5) is no
    measure of the distance to the stationary point there, and judged again it sent the search
    on for 20 iterations. Every check of the mode kill is accepted so (`-modekill` and the
    minima of the mapper as well): with the tighter environment of the checks a minimization
    of the mapper test of vinyl alcohol in two waters went back and forth between convergence
    and its check for its 400 iterations. No threshold of wavenumbers is involved anywhere.
  - **Measured on chorismate mutase** (the defaults, from the guess of AutoTS): 38 iterations,
    15,645 gradients, 596 s (41 iterations and 8.2 minutes before); the first Hessian found
    the environment on its saddle point (-1.156e-3) and left it; the check at convergence had
    the curvatures -9.866e-3, +4.19e-5 +- 5.3e-6, +8.72e-5, +1.22e-4, +1.37e-4, +1.58e-4
    (one imaginary mode, -510.6 cm^-1). The IRC from it judged its start alike (-9.866e-3,
    +3.66e-5 +- 5.9e-6, +8.84e-5, +1.17e-4, +1.36e-4, +1.51e-4) and ran both sides; the ends
    optimized to -49.6376661 Hartree (the reactant; EQ0 of the mapper -49.6376802) and
    -49.6632046 (the product, 75 iterations) against E* -49.6036742 at the transition state: a
    barrier of 21.3 kcal/mol and a reaction energy of -16.0 kcal/mol. From the transition
    state of the search with factor 0.1 the verified search took 5 iterations (225 s): its
    environment on a saddle point (-1.66e-5) was left, and the point was a first-order saddle
    in the minimum reached (+3.38e-5 +- 1e-9), E* -49.6036741, 1.2e-8 Hartree above.
  - **Rigid motions of an invariant system.** Where nothing holds the environment, the core
    gradient loses its part along the rigid motions of the core (the net force and torque the
    residual of the relaxation leaves; `E*` does not change along them). A minimization of a
    water in a shell of water (`constraints_matrix_orientsearch`, the cell with a bond
    constraint) stood for 1000 iterations with a step of zero and 7e-4 Hartree/Bohr along a
    rotation of the core.
- **Frequencies and IRC:** a transition-state search, its frequencies (those of the core),
  and the IRC run on this surface. The IRC is then the microiterative IRC of Polyak et al.
  (Phys. Chem. Chem. Phys. 2013, 15, 14188).
- **Output:** every evaluated core structure is kept with its relaxed environment, and at the
  end the structure files of the run (trajectory, optimized structure, IRC paths and ends,
  animations of the modes) are written with the whole system. The log ends with what the
  relaxations did.
- The Cartesian geometry blocks in a single-structure ONIOM `optmain` log show
  the real high-layer atoms at the start, at each iteration, and at the end.
- For a smaller optimization trajectory, set `-oniom_opt traj_high_only=on`
  (default `off`). Each `optmain` XYZ frame contains the real atoms of the
  highest layer in input order. Link atoms and atoms promoted to the optimization
  core are omitted. The same option in the mapper's `step1_settings.oniom_options`
  selects the atoms of its input and job trajectories. In `mdmain` it selects
  matching positions and velocities. Atom indices in analyses of these reduced
  MD files refer to their frame order. Optimized structures
  and IRC endpoints keep their full coordinates. When AutoTS needs the full path
  from step 1 as input to step 2, it retains that path separately.

## Replicas of the environment (`-oniom_opt micro_replicas=K`; new, off by default)

```text
optmain ts_guess.xyz -os gfn1xtb -order 1 -oniom 5673-5696 none openmm -oniom_opt micro_replicas=4
ircmain ts_optimized.xyz ... -oniom_opt micro_replicas=4 micro_replica_start=files micro_replica_files=ts_optimized_replicas.xyz
```

An environment of hundreds of atoms has many minima near each other, and which one a
relaxation finds can decide what a stationary point of `E*` is (above: a first-order saddle
point of chorismate mutase in one minimum of its environment, a second-order one in the next,
5e-7 Hartree apart). `micro_replicas=K` keeps `K` replicas of the environment and gives the
optimizer the soft minimum of their surfaces. This is a formulation of mopt-rs of its own,
labelled new: the multistructural microiteration of K. Suzuki, K. Morokuma, and S. Maeda (J.
Comput. Chem. 2017, 38, 2213; K. Suzuki, S. Maeda, Phys. Chem. Chem. Phys. 2022, 24, 16762) has
the same aim, but it is not implemented here and this is not that method.

- **The surface.** Each replica `k` is relaxed at every core `x` from its own environment of
  the core before, `E_k(x) = min over y near y_k of E(x, y)`, and
  `E_soft(x) = E_min - kT ln sum_k exp(-(E_k - E_min) / kT)` (log-sum-exp shifted by the
  lowest, `crate::math::stable`), with the weights `w_k = exp(-(E_k - E_min) / kT) / sum`.
  Every replica is at a minimum in its own environment, so `dE_k/dx` is its core gradient
  `g_k`, and `g_soft = sum_k w_k g_k` with no further term; the Hessian is
  `H_soft = sum_k w_k H_k - (1/kT) sum_k w_k (g_k - g_soft)(g_k - g_soft)^T`, with `H_k` the
  effective Hessian of replica `k` (the routes above). The second term is the weighted
  covariance of the replica gradients (equal to `sum w g g^T - g g^T`, written so that it cannot
  lose its sign to cancellation). One replica is the surface of one environment (the same code,
  bit for bit); `T -> 0` gives the lowest replica; `E_min - kT ln K <= E_soft <= E_min`. Where
  two replicas cross in energy, the minimum of the two has a kink (its gradient jumps by the
  difference of theirs) and the soft minimum is smooth: the weights pass from one to the other
  over `~ kT / |g_1 - g_2|`, and the covariance term is the negative curvature of that passage
  (`-(g_1 - g_2)^2 / 4kT` at the crossing of two).
- **What it is not.** `K` minima are not a thermal ensemble of the environment: `E_soft` is not
  a free energy (no entropy of the minima, no vibrations of the environment), and its energies
  and barriers compare only between points evaluated with the same replicas at the same
  temperature. A replica set changes discretely when a replica relaxes into another minimum, so
  `E_soft` is smooth only between such events, as `E*` is between the jumps of one environment.
- **Options.** `micro_replica_temperature=T` (K; 298.15 by default: at room temperature the
  minima of chorismate mutase that changed the character of its saddle point, 5e-7 and 1.4e-4
  Hartree apart, weigh alike, which is the case the replicas are for, while kT = 9.4e-4 Hartree
  keeps minima of other basins, kcal/mol above, out). `micro_replica_start=perturb` (the
  default): the first replica relaxes from the environment of the run, the others from that
  relaxed environment with every free atom moved by `micro_replica_amplitude` (0.05 Angstrom,
  the displacement of the measurements above, where 2 of 5 such starts relaxed into other
  minima) in a random direction of the seed `micro_replica_seed` (1); `md`: after
  `micro_replica_md_steps` (200, 100 fs) of molecular dynamics of the environment at the
  temperature with the core held (Bussi's thermostat of `mdmain`'s dynamics, the same seed);
  `files` with `micro_replica_files=FILE`: the environments of the last `K` whole structures of
  the file. Every start is a fixed function of the seed, the same for any number of threads.
- **Identity over a run.** Every replica relaxes from its own environment of the nearest kept
  core (the store of the kept cores holds one environment per replica; the replicas share its
  memory). Structures that the run remembers (the input, the frames a chain is made from) are
  the start of the first replica; the others follow their own environments from image to
  image.
- **Two replicas in one minimum** (no free atom apart by more than 0.1 Angstrom, energies
  within 1e-5 Hartree): at the start of a run a replica that relaxed into the minimum of
  another is started again from twice the displacement, up to three times (`perturb`, `md`; the
  structures of `files` are taken as they are). During the run, replicas that fall into one
  minimum are said in the log and keep their weights: taking one out, or starting it anew,
  would change the surface between two steps of the optimizer by up to kT ln 2, on top of the
  jump of the replica itself.
- **A replica on a saddle point of the environment** is moved off it as one environment is, in
  its own Hessians. **A replica far above the others** keeps relaxing (so that it keeps its
  identity), but its Hessian is not taken where its weight is below 1e-12 (27.6 kT above the
  lowest); the log and the end of the run count them.
- **Cost.** With mechanical embedding the model systems do not see the environment: their
  energy, gradient, and Hessian are computed once per core and only the lowest level per
  replica (a Hessian of the core takes the response of every replica, on the routes above).
  With electronic embedding every replica has its own higher level: `K` times the cost of one
  environment. The replicas relax in parallel where their evaluations share no calculator with
  electrons (mechanical embedding, a lowest level without electrons, and a Hessian route whose
  higher levels are computed once), whose SCF guesses would otherwise follow the order of the
  threads; one after the other in their order otherwise. The results are combined in a fixed
  order: bit for bit the same for any number of threads (`replica_tests.rs`). A calculator of
  the Python package (OpenMM) serializes its calls, so its replicas take `K` times as long.
  The preconditioners of the relaxations share one budget (16 factors within 512 MB for all
  replicas).
- **One surface across steps and commands.** The checks of stability (the verification of a
  saddle search, the start of an IRC, the stationary points of the mapper, the relaxation of
  `freqmain`) take every replica relaxed to `micro_check_tolerance` and the soft minimum of the
  same replicas at the same temperature; the log of a check gives the lowest curvatures of
  `H_soft` with their error (the change over the steps of every replica's environment to the
  minimum of its quadratic model, as for one environment). The structure files of the run hold
  the replica of the lowest energy, their comment line starting with
  `micro_replica=R/K weight=W replica_temperature=T`, and `NAME_replicas.xyz` beside each holds
  every replica of every frame (with its energy and weight). A command that is to go on on the
  same surface takes them with `micro_replicas=K micro_replica_start=files
  micro_replica_files=NAME_replicas.xyz` (the last `K` frames: those of the last structure). A
  command given such a structure without them says in its log that it is not on that surface
  (one environment, or replicas started anew) and does not claim to be. The journal of a run
  records the environments of every replica, and a restart goes on from them. `freqmain`
  analyses one whole structure: with replicas, the lowest one (its log says so).

**Every ONIOM run but the molecular dynamics has them.** `optmain`, `ircmain`, `nebmain`
(the chain and `-dmf`), `ieipmain`, `relaxedscan` (relaxed and rigid points), `confsearch`,
`orientsearch`, and the steps of AutoTS (the AFIR path, the chain, the transition states, their
IRCs, the optimization of the ends) move the core on the relaxed surface; `freqmain` relaxes
the environment of the structure it is given before it analyses it. A run is never continued
without microiterations because one of its options does not fit: what can be taken to the
relaxed surface is, and what cannot is refused with the reason. `-oniom_opt micro=off` is the
only way to move every atom in the optimizer (the log then says that the environment is not
relaxed), and the molecular dynamics (`mdmain`) move every atom by their equations of motion.

- **Constraints and bias potentials name atoms of the whole system.** Fixed atoms (`-fix`,
  such as the backbone of a protein model) of the environment stay where they are in the
  relaxations; those of the core are fixed in the optimizer. An atom of the environment that
  another constraint (`-pc`, the coordinates of a scan, `-uc`) or a bias potential (`-ma`
  AFIR, the restraints) acts on is promoted to the core: the optimizer moves it, holds the
  constraint, and feels the bias, and the rest of the environment is relaxed around it (the
  log lists the promoted atoms). Any choice of the core gives the same stationary points of
  the whole surface; only what the optimizer moves and what is relaxed differs. The
  constraints are renumbered to the core (`cli::oniom_micro::constraints_on_core`), and bias
  potentials and user functions are evaluated at the core in the whole system
  (`pes::oniom::micro::CoreTerm`: such a term does not depend on the relaxed atoms, so
  `min_y [E(x, y) + B(x)]` is `E*(x) + B(x)`). An AFIR path between atoms of the active site
  is then a path of the core with the solvent or the protein relaxed at every step, rather
  than a path on which the environment is dragged along by the optimizer. The fragments of
  `-gfix` and the reactive bonds and the product of CCQN are taken to the core the same way.
- **A constraint on a Hessian eigenvector** (`-pc eigvec`) is one on an eigenvector of the
  Hessian of the relaxed surface, which is the Hessian the optimizer has.
- **Electronic states.** An excited state (`-es`) is that of the highest level, on the relaxed
  surface. A function of several states (`-mf`) is evaluated state by state at one relaxed
  environment: with mechanical embedding and a lowest level without electrons (a force
  field) the environment does not depend on the state of the model system, so all states
  share it and a crossing is one of whole structures. The model systems follow the charge
  and multiplicity of the state (unless `-oniom_opt high_charge=` gave them, which is then
  refused for states of another charge or multiplicity). Where the environment would depend
  on the state (a lowest level with electrons, electronic embedding), a function of states
  that differ in charge, multiplicity, or root is refused.
- **A first Hessian from a file** (`-ih`, the Hessians a scan hands from point to point with
  `hessian=continue`) is taken as the Hessian of the core when it has the size of the core,
  and reduced to the core with the response of the environment taken off, `H_xx - H_xy
  H_yy^+ H_yx`, when it is one of the whole system.
- **ReSPer coordinates** of `relaxedscan` are made of the atoms of the core unless
  `resper_atoms=` chooses others (which are promoted to the core): the environment is
  relaxed at every point and does not define the coordinate.
- **Refused:** a bias potential or a user function that acts on every atom or does not say
  which atoms it uses (`-nrp`, a user-registered one: nothing would be left to relax), a
  variable cell (`-vc`; ONIOM has no stress), and `-mf bitss` (its two images are not set up
  on the relaxed surface; `nebmain` and `nebmain -dmf` search between two structures with
  microiterations).

When atoms of the environment are held, NEB keeps the core in the coordinate
frame of the environment throughout the relaxation. Python QM engines receive private
coordinate arrays. A backend may translate its model to its own origin; its Cartesian
derivatives must use the axes of the input and embedded charges must share the model
translation. The Psi4 interface requests `no_com` and `no_reorient`; Gaussian uses
`nosymm` by default. Translating or rotating the
core alone would change its interactions with the environment.

`nebmain` and `ieipmain` relax chains and pairs of the core on the relaxed surface. Every image
follows its own environment: a relaxation starts from the environment of the nearest core
structure evaluated so far, which in a chain is the image's own one of the iteration before (in
an optimization, the last point), and the first one of an image starts from the structure it
was made from (the frames of the input, such as an AFIR path, are kept for that). A point that
is asked again (another state of `-mf`, a Hessian after the gradient) is not relaxed again. The
distances along the chain (`-nd`) are those of the core, so a path in which solvent molecules
rearrange is not cut into images by those motions. The optimizations after the chain
(`-neb_opt optimize`) start from the whole structures with the calculator's Hessian, since the
chain kept Hessians of the core.

`confsearch` searches the conformers of the core (for a solute in a solvent shell or a ligand
in a pocket, the solute or the ligand): every optimization, every step of its dynamics, and
every Hessian has the environment relaxed, and the conformers are written with the whole
system.

The files of a run are written with the whole system at its end. The environments of the
evaluated core structures are kept for that within a budget of memory (256 MiB); a structure
that is no longer kept and a structure that a restart replayed from the journal have their
environment relaxed again when the file is written, so that every whole structure of the run
in a file has a relaxed environment of its own. The frames of an analysis (the animations of
`vibration_animation/` and `koopman_analysis/`: a core displaced along a mode) take the
environment of the nearest kept core as it is; relaxing one for every frame took 50
relaxations, half the time, of an optimization of chorismate mutase with `-km`. A restart replays the
evaluations of the core together with the environments that were relaxed at them (the
journal records both), so the run goes on from the environments it had.

In the code, the ONIOM surface of whole structures is a type of its own
(`pes::oniom::WholeOniom`), not a calculator: the registry hands it out only through
`CalculatorRegistry::create_oniom`, and a run turns it into a calculator by saying what it is
for (under the microiterations, for the molecular dynamics, at given structures, or with every
atom moving because `micro=off` says so). A command cannot move structures on it by oversight.
