# Calculators

`-os NAME` (or `-xtb METHOD`, `-dxtb METHOD`) chooses the program; the default is PySCF.

| Name | Program |
|---|---|
| `pyscf` (default) | PySCF HF/DFT, TDA excited states (`-func`, `-bs`, `-sub_bs`, `-ecp`, `-u`, `-grid`); restricted open-shell and generalized wavefunctions with `-calc_opt reference=ro` / `reference=g` |
| `psi4` | Psi4 |
| `tblite[:METHOD]`, `-xtb METHOD` | tblite GFN1/GFN2-xTB (`-alpb`, `-cpcm`) |
| `dxtb[:METHOD]`, `-dxtb METHOD` | dxtb with autograd Hessians |
| `lj`, `emt`, `tersoff`, `muller_brown` | potentials implemented in Rust. `lj` sums over all pairs as MultiOptPy does (nearest images in a cell); with `-calc_opt cutoff=ANGSTROM` it takes the pairs and images within the cutoff from a linked-cell search and switches the potential off smoothly over `switch=ANGSTROM` below it (a tenth of the cutoff by default). `emt` follows ASE; `-calc_opt switch=ANGSTROM` removes the small step of the energy at the cutoff of its neighbor list |
| `eam` | embedded-atom method of Zhou, Johnson, and Wadley (2004) for Cu, Ag, Au, Ni, Pd, Pt, Al, Pb, Fe, Mo, Ta, W, Mg, Co, Ti, Zr and their alloys, with exact gradient and Hessian; isolated systems and periodic cells. Switched off smoothly over the last tenth of its cutoff (`-calc_opt switch=ANGSTROM` sets the width, 0 turns it off) |
| `meam` | modified embedded-atom method (Baskes 1992; angular densities and many-body screening) for Cu, Ag, Au, Ni, Pd, Pt, Al, Pb, Rh, Ir, Li, Na, K, V, Nb, Ta, Cr, Mo, W, Fe, Mg, Si, C, Sn, with exact gradient and Hessian; isolated systems and periodic cells. Settings: `-calc_opt cutoff=ANGSTROM cmin=C cmax=C zbl=false smooth=true` (`smooth`: an infinitely smooth step in place of the cutoff function of the model, for a continuous Hessian) and, for a pair of different elements, `pair=A-B,LATTICE,EC,RE,ALPHA` (reference structure `fcc`, `bcc`, `hcp`, `dia`, `b1`, `b2`; cohesive energy in eV per atom, nearest-neighbour distance in Angstrom, `alpha` of the equation of state). Without `pair`, different elements interact through the means of their own references, which is not fitted to any alloy (the run log says so) |
| `gfn0xtb` | GFN0-xTB built into mopt-rs (H to Rn, every element of GFN0-xTB; isolated systems; the same numbers as xtb 6.7.1 for the 82 test structures with K to Rn: energies within 1e-9 Hartree, gradients within 2e-9 Hartree/Bohr; see "Elements of the built-in GFN0-xTB and GFN1-xTB" below), also `-modelhess gfn0xtb`. The Hessian of the ground state is analytic (`-calc_opt hessian=numerical`: central differences of the analytic gradient, `hessian_step=BOHR`); those of excited states are differences of their analytic gradients |
| `gfn1xtb` | GFN1-xTB built into mopt-rs (H to Rn, every element of GFN1-xTB; isolated systems; see "Elements of the built-in GFN0-xTB and GFN1-xTB" below): self-consistent charges, D3(BJ), the halogen-bond correction (acceptors N, O, P, S); the same numbers as tblite (energies within 2e-9 Hartree, gradients within 1e-9 Hartree/Bohr on 119 test systems, among them aqua, ammine and carbonyl complexes of the 3d, 4d and 5d metals, all fifteen lanthanides, ferrocene, and two pairs of metals; where tblite's search of the Fermi level leaves the electron count off by 1e-8, as for high-spin Fe(II), the energy within that much more). The SCC starts from the charges of the last structure with the same atoms, and from the superposition of atomic densities otherwise (`-calc_opt scc_guess=sad`: always from the superposition, so that a result does not depend on what was computed before), and extrapolates the Fock matrices by ADIIS, switching to CDIIS when ADIIS stalls (`diis_patience`) or falls too slowly (`diis_rate`, `diis_window`: by default a commutator that has not fallen to 0.3^3 of its value over 3 iterations; `diis_rate=off` keeps ADIIS until it stalls), with `diis_depth` matrices and the density damped by `diis_damping` while ADIIS runs; a run that runs away to an atomic charge beyond `charge_limit` (4 e), which the unbounded third-order term allows, is repeated with stronger damping. With a d-block metal, a lanthanide, or Ca, Sr, Ba (or after a failed run) the shell charges are also mixed by Broyden's method (mixing 0.4, as tblite) and the converged solution of lower energy is kept, since the extrapolated Fock matrices can settle on another self-consistent solution there (a Ti-Ru hydride: 0.245 Hartree above tblite's solution; SrCl2 with Sr at -2.10 e: 0.551 Hartree above; HfCl4: 0.604 Hartree above; measured on the 119 test systems, Broyden mixing for all of them agrees with tblite everywhere at 7439 SCC iterations in all against 5457 without, so it is used for these metals only); `-calc_opt scc_broyden=auto|on|off`. The Hessian is analytic (`-calc_opt hessian=numerical`: central differences of the analytic gradient), also as the model Hessian of a run on any surface, `-modelhess gfn1xtb` (with the charge and multiplicity of the run; see "Model Hessians" in [migration_optimizers.md](migration_optimizers.md)). Excited singlet states (`-es`) for tests: the TDA of its own SCC response, or sTDA with `-calc_opt excited=stda` (see "Excited states" below) |
| `gfnff` | GFN-FF built into mopt-rs, a port of xtb 6.7.1 (H to Rn, the transition metals and lanthanides with xtb's own rules for their bonds, angles, and torsions; see "Elements of GFN-FF" below; isolated systems and three-dimensional periodic cells, with the stress for `-vc`; see "Periodic GFN-FF" below), also `-modelhess gfnff`; the topology comes from the starting structure and a warning names bonds that form or break (`-calc_opt refresh_topology=true` sets it up at every structure, `hb_switch=on|off|SHARE` says how the hydrogen and halogen bond terms end at the thresholds of their lists: switched off smoothly over the last 10 % by default, so that the energy has no steps). The Hessian is analytic, for molecules and cells (`-calc_opt hessian=numerical`: central differences of the analytic gradient, `hessian_step=BOHR`; see "Analytic Hessian of GFN-FF" below) |
| `gaussian`, `orca`, `gamessus`, `nwchem`, `mopac` | programs that run through input and output files, by their ASE calculators (see "Programs run through files"); Gaussian and ORCA give their own Hessians |
| `gpaw`, `espresso` (`pwscf`, `qe`) | GPAW and Quantum ESPRESSO through ASE |
| `openmm` | force fields of OpenMM (optional; see "OpenMM" below): serialized systems or a force field on PDB topologies, also as a lower level of ONIOM |
| `pm6_rs`, `gfn2_rs`, `mace_mp`, ... | named ASE profiles (`mopt_rs.calculators.PROFILES`) |
| `ase:module:Class` | any ASE calculator |
| `multioptpy:NAME` | a program through an installed MultiOptPy |

MultiOptPy is not a dependency of mopt-rs: nothing of it is installed with mopt-rs or used
by it, and `multioptpy:NAME` works only where it was installed separately.

Programs other than the Rust potentials run in Python: use the pip-installed commands
(`optmain`, ...) or `mopt-py <command>`. `-calc_opt key=value ...` passes program-specific
settings (Python literals), for example `-calc_opt density_fit=True conv_tol=1e-10`; for
`ase:module:Class` they are constructor keywords.

The charge (`-elec`) and the multiplicity `2S + 1` (`-spin`) mean the same for every program;
each adapter converts to its program's convention (PySCF `mol.spin` and xTB `uhf` are `2S`).
The spin state can also be given as the number of unpaired electrons `2S` (`-unpaired N`;
`unpaired_electrons=` in Python for `Optimizer`, `Structure.build`, `evaluate_native`), as
PySCF, xTB, and several Rust engines count it; with both, they must agree. It only sets the
state: an unrestricted reference for a singlet is `-u`.

## Elements of the built-in GFN0-xTB and GFN1-xTB

`gfn0xtb` and `gfn1xtb` have parameters for every element of the two methods, H to Rn
(Z = 1 to 86); other elements are refused with that range named. The tables are generated from
the upstream files (`tools/params/gen_gfn0_xtb.py` from xtb's `param_gfn0-xtb.txt`,
`approxrab.f90`, `ncoord.f90` and multicharge's EEQ-2019 parameters;
`tools/params/gen_gfn1_xtb.py` from xtb 6.7.1's `gfn1.f90` and `src/param/`, checked value by
value against the parameter file of tblite, `tblite param --method gfn1`, including tblite's
869 element-pair parameters). What the heavy elements need:

- Basis: the lanthanides La to Lu carry 5d 6s 6p shells with the 4f shell in the core (three
  valence electrons), so no element of either method has f functions; the 6s and 6p shells
  are STO-6G expansions in both methods (xtb's `setGFN0NumberOfPrimitives` and
  `setGFN1NumberOfPrimitives`), and the 5s/5p and 6s/6p expansions are those of xtb's
  `slater.f90` (GFN1-xTB agrees with tblite to 1e-10 Hartree on the fifth- and
  sixth-period records with them, so tblite uses the same sets there; only its 4s/4p sets
  differ from xtb's, see `third_party/DISPERSION_AND_PARAMETERS.md`).
- Pair parameters: GFN0-xTB scales the Hamiltonian between two d-block metals (Sc to Cu, Y to
  Ag, La to Au) by 1.10, and between two of Cu, Ag, Au by 0.9 (`setGFN0PairParam`); GFN1-xTB
  takes the mean of 1.1, 1.2, 1.2 for the 3d, 4d and 5d (with the lanthanides) rows, and
  0.80 for Re-H and Pt-H beside the light pairs (`initHamiltonian` of `gfn1.f90`).
- GFN1-xTB's halogen bond counts Cl, Br, I and At (`xbond` of xtb's `scf_module.F90`).
- GFN0-xTB's short-range basis correction puts every element beyond Ar in the fourth row of
  its polynomial, as xtb's `itr` does.
- The D3 (GFN1) and D4 (GFN0) reference data and the EEQ-2019 charges cover these elements.

Validation (`crates/mopt-rs/tests/gfn0_xtb.rs`, `gfn1_xtb.rs`): 56 structures with every
element from Rb to Rn other than Ru and I (already tested), made by
`tools/golden/gfn1_xtb.py`: hydrides, halides, oxides, carbonyls (Mo, W, HRe(CO)5), ammine
complexes, cisplatin and PtH2(NH3)2, Hg(CH3)2, the trichlorides of all fifteen lanthanides,
XeF2, Rn with water, an At...N halogen bond; open shells (NbCl4, SnH3, GdH2, IrCl6 2-), cations
and anions. For GFN1-xTB the references are tblite 0.4.0 (`tests/golden/gfn1_xtb.json`, 119
records in all), compared within the tolerances of the earlier records: energies 2e-9
Hartree, gradients 1e-9 Hartree/Bohr, Mulliken charges 1e-8 e. For GFN0-xTB they are xtb 6.7.1
(`xtb --gfn 0 --grad`, `tools/golden/gfn0_xtb_binary.py`; `tests/golden/gfn0_xtb.json`, 82
records from xtb), compared within 1e-9 Hartree and 2e-9 Hartree/Bohr. The worst deviations
on the new records: GFN1-xTB 2.4e-10 Hartree in the energy (RbF),
2.4e-10 Hartree/Bohr in the gradient (HRe(CO)5) and 5.7e-9 e in the charges (HoCl3); GFN0-xTB
4.5e-10 Hartree (Mo(CO)6) and 1.2e-9 Hartree/Bohr (W(CO)6). The SCC of GFN1-xTB converges for
every new record from the superposition of atomic densities, in 3295 iterations together (Broyden
mixing included where it runs), 314 at most (CeCl3; LaCl3 245, HfCl4 296, the other lanthanide
trichlorides 53 to 83, IrCl6 2- 123, TcO4- 135, the molecules without a d-block metal, a
lanthanide or Ca, Sr, Ba 13 to 22). The analytic gradients and Hessians are checked against
differences on seven heavy-element molecules per method (CsF, GdH2, PtHCl, XeF2, HAt with
ammonia, SnH3, AgAuH), and the analytic gradients of the excited states (GFN0 sTDA, GFN1 TDA
and sTDA) on bent HgCl2.

## Kinds of wavefunction with PySCF (`-calc_opt reference=`)

| `reference=` | closed shell | open shell | gradient | Hessian (`-fc`) |
|---|---|---|---|---|
| `auto` (default) | RHF / RKS | UHF / UKS (`-u`: also for closed shells) | analytic | analytic |
| `ro` (`rohf`, `roks`, `r`) | RHF / RKS | ROHF / ROKS | analytic | differences of gradients for open shells |
| `u` | UHF / UKS | UHF / UKS | analytic | analytic |
| `g` (`ghf`, `gks`, `noncollinear`) | GHF / GKS | GHF / GKS | differences of energies | differences of those gradients |

- Restricted open-shell: one set of orbitals for both spins, so the wavefunction has the
  spin asked for exactly (no spin contamination; `<S**2>` is not printed for it).
- Generalized: every orbital is a mixture of alpha and beta, which allows spins that do
  not point along one axis (non-collinear). PySCF has no analytic gradients for these
  methods; mopt-rs takes them from central differences of the energy, `6N` calculations per
  gradient with a displacement of 0.005 Bohr (`-calc_opt gradient_step_bohr=`), so this is
  for small systems. `<S**2>` is printed at every iteration. For GKS, `-calc_opt
  collinear=col|ncol|mcol` is PySCF's way of evaluating the functional: `mcol`
  (multi-collinear) where the `mcfun` package is installed, otherwise `ncol` (locally
  collinear), unless given; `col` evaluates it as for collinear spins. The SCF starts from
  PySCF's guess, which usually leads to the collinear solution where one exists; another
  start can be set with a hook on the SCF object (`hook=module:function`).
- Excited states (`-es`) need `auto` or `u`. Orbital analyses (`freqmain -nro`, `-roet`)
  work with `ro` (the two spins share their orbitals and differ in occupations) but not
  with `g`.

## Elements of GFN-FF

`gfnff` has the parameters of H to Rn (Z = 1 to 86), every element of the xtb 6.7.1 release.
Other elements are refused with "GFN-FF has parameters for H to Rn (Z = 1 to 86) only, not
Fr (Z = 87)".
- **Why not Fr to Lr.** The xtb source used for the tables
  (`tools/params/gen_gfnff.py`) is a later state than the release, with tables to Lr. Its D4
  reference systems, from which the dispersion takes its C6 coefficients, stop at Rn (none for
  Z = 87 to 111). The xtb 6.7.1 executable gives no usable result there either: FrF and RaF
  stop in the topology charges, AcF to LrF find no bonds, and UF6 has a NaN dispersion
  energy. So there is nothing to take the parameters of these elements from or to compare
  them with.
- **The lanthanides are those of the release.** The later source refits `chi`, `gam`, `cnf`,
  `alp` (EEQ) and `r0` (bond radius) of La to Lu. It also scales the bond lengths of
  lanthanide and actinide halides (`scaleF`, 0.94 to 1.06) and takes H atoms with a topology
  charge above -0.0281 out of the neighbours of a lanthanide or actinide (`adjust_NB_LnH_AnH`).
  The release has none of this. With the later parameters the 15 LnCl3 of the references
  missed xtb 6.7.1 by up to 0.18 Hartree in the electrostatic energy. So the generator takes
  these 75 numbers from the xtb 6.7.1 library, read as data. Every other entry of H to Rn is
  the same in both, and the two steps of the later source are not ported.
- **References.** `tests/golden/gfnff.json` (`tools/golden/gfnff.py`, xtb 6.7.1 `--gfnff
  --grad`) has 173 molecules; 103 of them are new and cover every element from Rb to Rn:
  - the GFN1-xTB reference geometries of Rb to Rn (`elements_*`), with Ln (La to Lu)
    trichlorides and GdH2;
  - xtb's cerium complex `Ce_0a7745`, neutral and +1, and the charged forms of the molstock
    molecules `manganese` (-2), `bug332` (+1), and `fe_cnx6` (-4); `co_cnx6` with -3 has a
    NaN energy in xtb and is left out;
  - noble-gas atoms and dimers (Xe, Rn, Xe2, KrXe, Xe-water) and XeF4;
  - main-group compounds of the fifth and sixth periods (SnMe4, PbMe4, SnCl2, TeCl4, IF5,
    SbCl5, InMe3, BiMe3, I3-, SbF6-, Tl+-water);
  - 4d and 5d complexes: ruthenocene, osmocene, Cp2ZrCl2, (C6H6)Mo(CO)3, CpRe(CO)3,
    Pd(PH3)4, WCl6, NbF5, [Ag(NH3)2]+, ClAuPH3, CdMe2, [IrH6]3-, Zeise's anion;
  - lanthanide complexes: [La(H2O)8]3+, [Gd(NH3)6]3+, LaH3, LuMe3, CeCp3, [Eu(OH)(H2O)]2+;
  - hydrogen and halogen bonds with heavy atoms: water-SbH3, HI-water, I- -water,
    Me2Te-NH3, ICl-NH3.
  The 70 earlier records are kept as they were. Run again, xtb reproduces them to 1e-12 in
  the terms and 1e-15 in the gradient.
- **Agreement.** The test tolerances are 1e-10 Hartree per term and 1e-9 Hartree/Bohr for the
  gradient (`crates/mopt-rs/tests/gfnff.rs`); no new element needed a looser one. Over all 173
  molecules the largest deviations are 2.1e-12 Hartree in the bond energy and 2.3e-12 in the
  total (dimethyl sulfoxide, an old record), 5e-13 or less in every other term (bond, angle,
  torsion, repulsion, electrostatics, dispersion, HB, XB, bonded ATM), and 2.4e-11
  Hartree/Bohr in the gradient (feco5).
- **Derivatives.** For ten of the new structures the analytic gradient agrees with central
  differences of the energy (step 1e-4 Bohr) to 9.2e-9 Hartree/Bohr or better. The analytic
  Hessian of all 173 molecules is compared with differences of the gradient
  (`crates/mopt-rs/tests/gfnff_hessian.rs`).
- **Periodic case.** The XeF2 crystal (`gfnff_pbc_reference.json`, two molecules in the
  conventional cell) agrees with xtb to 4e-13 Hartree in every term, 5e-16 Hartree/Bohr in the
  gradient, and 6e-15 Hartree in the strain derivative; the counts of its lists agree. Its
  strain derivative also agrees with differences of the energy.
- **Geometries of the piano-stool complexes.** M-C-O is bent by 3 degrees there. With the
  first, exactly linear M-C-O both programs returned gradients of up to 1e6 Hartree/Bohr on
  the metal and the carbonyls, which differences of the energy do not show (those are below
  0.3). The energies still agreed except the hydrogen-bond term, by 2e-9 of 2e-7 Hartree.
  The cause was not looked into further.

## Periodic GFN-FF (`-os gfnff` in a three-dimensional cell)

`gfnff` takes a structure with a three-dimensional cell (an extended XYZ file with
`Lattice=`) and computes it as xtb 6.7.1 does:
- **Topology.** The positions are wrapped into the cell, bonds are looked for to the images
  in the 27 cells around the home cell, and each bond keeps the cell of its image. The bond
  paths (1-2, 1-3, 1-4) count only inside those 27 cells. Rings, topology charges, and the
  Hueckel pi system are those of the folded graph, where bonds through any cell join atom
  indices.
- **Lattice sums.** The repulsion, the coordination numbers, and the dispersion are summed
  over xtb's translations. The electrostatics are an Ewald sum with xtb's fixed cells and
  its splitting parameter. The bonded and hydrogen- and halogen-bond terms see the images of
  their atoms.
- **Stress.** Every term gives its strain derivative, so `gfnff` returns the stress
  analytically and `optmain -vc` works with it.

Each structure is taken at the images of its atoms closest to their positions at the setup,
so the topology stays valid however a caller wraps the atoms. A changed cell, as in `-vc`,
keeps the topology, as xtb does. One- and two-dimensional periodicity is refused, as xtb
refuses it; a slab or a wire goes in a three-dimensional cell with vacuum.

xtb's approximations are reproduced, because they decide its numbers:
- a pair of an atom with its own image counted twice in the repulsion, the dispersion, and
  the coordination numbers;
- rings that wrap around the cell;
- no hydrogen bond between an atom and its own image;
- the H-bond softening of an A-H bond lost when the bond crosses the boundary and is stored
  from A;
- the term `lambda Q` of a charged cell.

The specification with the sizes of these effects is in the comments of
`pes::native::gfnff` and its `lattice`, `periodic_cn`, `periodic_eg`, and `ewald` modules.

Against xtb 6.7.1 (`tests/golden/gfnff_pbc_reference.json`, 19 cells):
- **Cases.** Cubic ice (also shifted), water and ICN chains with one and two molecules per
  cell, the benzene crystal and its 2x1x1 supercell, diamond in three cells, graphene 1x1,
  2x2, and 3x3, polyethylene with one and two units, caffeine and methylammonium(+1)
  in a box, and the XeF2 crystal.
- **Agreement.** Every term, the total, the gradient, the strain derivative, and the EEQ
  charges agree to 1e-12 or better, and the numbers of bonds, bends, and torsions agree
  with xtb's lists.
- **Graphene 3x3.** Its cell contains the Dirac point, so the Hueckel levels at the Fermi
  level are degenerate up to xtb's 1e-9 perturbation. The bond energy differs by 8e-9
  Hartree, the gradient by 6e-8 Hartree/Bohr, and the strain derivative by 7e-7 Hartree,
  which is how differently two eigensolvers resolve that subspace.
- **Stress.** The strain derivative agrees with central differences of the energy (step
  1e-6) for ice, benzene, diamond, and polyethylene.

## Analytic Hessian of GFN-FF

`gfnff` computes its Hessian with respect to the positions analytically, for molecules and
for three-dimensional cells (`Calculator::analytic_hessian` is true). `-calc_opt
hessian=numerical` makes it of central differences of the analytic gradient instead, as
before (`hessian_step=BOHR`, 5e-3 by default).

- **Terms.** Every term of the energy is differentiated twice; none is left to differences:
  - the bond stretches, with the CN dependence of their reference lengths and the HB
    coordination of the softened A-H bonds;
  - the bends, the torsions and out-of-plane terms, and the bonded ATM term;
  - the bonded and non-bonded repulsion;
  - the D4-type dispersion, with the second derivatives of the reference weights and of the
    coordination numbers;
  - the EEQ electrostatics, with the response of the charges to the positions (from the
    bordered EEQ system: `dq/dx = -P r`, `P` the atom block of its inverse);
  - the hydrogen and halogen bonds;
  - in a cell, the lattice sums and the Ewald matrix.
- **How.** The few-atom terms (bends, torsions, ATM, hydrogen and halogen bonds) are the
  energies of the gradient, differentiated with hyper-dual numbers. A linear bend is written
  with the power series of `acos(-cos theta)^2` in `1 + cos theta`, which stays smooth at the
  linear geometry. The pair terms, the coordination numbers, and the EEQ response are written
  out. The chain rule through the coordination numbers is `sum_i w_i d2CN_i + J^T M J +
  J^T V + V^T J`, with the sparse CN Jacobian `J`.
- **The derivative of the gradient.** The Ewald splitting parameter and the Wigner-Seitz
  images are held fixed, as in the gradient. A hydrogen-bond term whose out-of-line exponent
  passes 15 is zero in xtb; it is now skipped before it is evaluated, in the gradient as well
  (same numbers, less work).
- **Agreement.** `crates/mopt-rs/tests/gfnff_hessian.rs` compares the Hessian with five-point
  differences of the gradient (step 1e-3 Bohr). It covers the 173 molecules of
  `tests/golden/gfnff.json`, with both endings of the hydrogen-bond terms, and the 19 cells of
  `tests/golden/gfnff_pbc_reference.json`.
  - **Largest deviation.** 3.3e-7 Hartree/Bohr^2 for the molecules, from the hexacyanide of Fe
    (whose largest element is 1e2), and 2.2e-8 or less for the others; 4.6e-10 for the cells.
  - **Jumps in the gradient.** A difference stencil that crosses one of the drops of a
    hydrogen-bond term (about 1e-10 Hartree/Bohr in the gradient) is off by up to 1e-7. Such
    columns are taken again with smaller steps.
  - **Fe and Co hexacyanides.** xtb's topology bonds N to the metal and puts a bend at N
    between C and the metal at theta = 0.01 to 0.1 degrees (theta_0 = 180 degrees). That is
    the cusp of the bend energy at theta = 0. Its curvature changes over 2.5e-4 (Fe) to 2e-3
    Bohr (Co), so the differences reach the analytic value only with steps of 1e-6 and 1e-5
    Bohr.
- **Cost.** One thread, AMD Ryzen 7 260, `cargo run --release --example
  bench_gfnff_hessian`. Differences are `6N` gradients, measured up to 400 atoms and
  estimated as `6N` times one gradient above that. Max difference is between the two
  Hessians, which is the error of the central differences.

| structure | atoms | one gradient (ms) | analytic Hessian (s) | differences (s) | ratio | max difference (Hartree/Bohr^2) |
|---|---|---|---|---|---|---|
| caffeine | 24 | 0.24 | 0.006 | 0.033 | 5 | 2.3e-5 |
| rivaroxaban | 47 | 1.4 | 0.025 | 0.37 | 15 | 2.8e-5 |
| Grubbs catalyst | 75 | 4.7 | 0.053 | 2.0 | 39 | 1.8e-5 |
| taxol | 113 | 14 | 0.18 | 10.9 | 60 | 2.9e-5 |
| 64 waters (grid) | 192 | 70 | 0.71 | 96 | 135 | 1.3e-5 |
| 216 waters (grid) | 648 | 341 | 3.6 | 1324 (estimated) | 368 | - |
| 512 waters (grid) | 1536 | 1302 | 13.8 | 11995 (estimated) | 871 | - |
| cubic ice (cell) | 24 | 31 | 0.22 | 3.3 | 15 | 4.9e-6 |
| benzene crystal | 48 | 101 | 0.87 | 26 | 30 | 1.7e-5 |
| diamond 2x2x2 | 64 | 54 | 0.14 | 20 | 143 | 6.7e-7 |
| benzene crystal 2x1x1 | 96 | 210 | 1.8 | 119 | 68 | 1.7e-5 |

The analytic Hessian costs about 10 to 25 gradients for the small molecules and cells. It
grows as `N^3` (the EEQ response), while the differences grow as `N` gradients, so the gain
grows with the size.

### The lattice vectors

`Gfnff::cell_hessian` (Rust; `GfnffTopology::cell_hessian_with`) gives the analytic Hessian
of a cell in the positions and the three lattice vectors together, the second derivatives of
the strain included.
- **Variables.** `(3N + 9) x (3N + 9)` Hartree/Bohr^2. The Cartesian positions and the
  lattice vectors are independent variables: the images of atom `j` are at
  `x_j + sum_k n_k a_k`, and variable `3N + 3k + c` is component `c` of lattice vector `k`.
  It is the derivative of the gradient and of the strain derivative (the stress) that the
  calculator returns. The second derivatives in a homogeneous strain, or at fixed
  fractional coordinates, follow from it by the chain rule.
- **How.** Every term sees the lattice vectors through the cells of its images, so its
  blocks with them are those of the positions times the coefficients of the cells. The
  reciprocal sum of the Ewald matrix depends on the lattice through the reciprocal vectors
  and the volume. Its derivatives come from the structure factors
  `S(G) = sum_j q_j exp(i G.x_j)` (the reciprocal energy is `sum_G W(G) |S(G)|^2 / 2`), with
  `G` and `W` differentiated with hyper-dual numbers in the nine components of the lattice.
  The EEQ charges respond to the lattice as to the positions. The Ewald splitting parameter,
  the images of the pairs, and the sets of translations are held, as in the stress.
- **Agreement.** `crates/mopt-rs/tests/gfnff_hessian.rs` checks the 17 cells of up to 64
  atoms of `gfnff_pbc_reference.json`. It compares the nine lattice columns and three
  position columns with five-point differences of the gradient and of
  `dE/da = A^-T (sigma - X^T G)` from the stress: they agree to 1.1e-9 Hartree/Bohr^2 (the
  2x1x1 benzene supercell, measured once, to 2.1e-9). The block of the positions equals the
  Hessian of the positions to 1e-16. Most lattice columns need the smaller steps: moving a
  lattice vector by 1e-3 Bohr moves far images across the sharp cutoffs of the lattice sums
  (60 Bohr).
- **Use.** With the stress asked for, the calculator returns this Hessian (in the positions
  and the lattice vectors) and its form along the strain (`EvalOutput::strain_hessian`), so
  the variable-cell surfaces (`-vc`) and the IRC in a moving cell take their lattice columns
  from it instead of from differences of the gradient, as they do with the other built-in
  periodic engines (LJ, EMT, EAM, MEAM, Tersoff).
- **Cost.** Against the analytic Hessian of the positions with the nine lattice columns from
  central differences (18 gradients with the stress):

| cell | atoms | analytic, positions and lattice (s) | positions analytic + 18 gradients (s) | ratio |
|---|---|---|---|---|
| cubic ice | 24 | 0.23 | 0.69 | 3.0 |
| benzene crystal | 48 | 0.87 | 2.5 | 2.9 |
| diamond 2x2x2 | 64 | 0.17 | 1.5 | 9.0 |
| benzene crystal 2x1x1 | 96 | 1.9 | 6.2 | 3.2 |

## Stability of SCF solutions (`-stability N`)

An SCF can converge to a solution that is not the lowest of its method: the restricted
solution of a stretched bond, a symmetric solution where the symmetry breaks.
`-stability N` (every command; 0, the default, never) has the calculator test its solution
every `N` evaluations, and where it is unstable, go on from the orbitals rotated along the
unstable direction until it is stable; the evaluation is then repeated there, so that the
optimization, the chain, or the path gets the energy and the gradient of the stable
solution, and the journal of the run records those. `-stability_external` says what is done
about an instability towards another kind of wavefunction (restricted to unrestricted):
`report` (the default: said, the kind is kept), `switch` (the calculator goes on with
UHF/UKS for the rest of the run), or `off` (not tested). What was found is in the log with
the evaluation it belongs to (`Warning: ... SCF stability: ...`); a stable solution is not
mentioned. The test is that of the program: PySCF's `stability()` (internal for RHF,
ROHF, UHF, GHF and their Kohn-Sham forms, external from restricted to unrestricted; an
unrestricted solution unstable towards a generalized one is said only); Psi4 repeats the
evaluation with `stability_analysis follow` (UHF/UKS) or `check` (RHF/RKS: an
instability is printed in its output, not followed); ORCA repeats it with `%scf
STABPerform true end` (with `switch`, `STABRestartUHFifUnstable true`). The runs of Psi4
and ORCA are not part of these tests (the option and the input are checked). Gaussian
would need a job of its own (`stable=opt`) before the job of the forces and has no test
here. A calculator without an SCF that it can test (the built-in engines, force fields,
Gaussian and the other programs) says so once and is not changed. Stretched H2 (2.5 Angstrom, HF/STO-3G): the symmetric
UHF solution is left for the broken-symmetry one, and the RHF solution is reported as
unstable towards UHF or, with `switch`, replaced by it (`tests/test_stability.py`).

## Any program setting, without a new option

Solvent models, external fields, or any other feature of a program need no mopt-rs option of
their own. `-calc_opt` (or `calc_option` in a settings file, `extra` of `CalculatorOptions` in
Python) reaches the program in three ways:

| Form | Goes to | Example |
|---|---|---|
| `key=value` | the engine's keywords: ASE constructor keywords, `psi4.set_options`, tblite `Calculator.set`, PySCF engine keywords | `perturb_h=True perturb_with=dipole perturb_dipole=[0,0,0.01]` (Psi4 field) |
| `prefix.path=value` | the program's own objects | PySCF `scf.conv_tol=1e-10`, `scf.with_solvent.eps=78.4`, `mol.cart=True`; tblite `add.electric-field=[0,0,0.001]`, `add.gbsa-solvation=water`, `set.accuracy=0.1` |
| `hook=module:function`, `hook.NAME=value` | user code called as `function(obj, **kwargs)` on the program object when the engine creates it; a returned object replaces it | anything the program's API allows |

PySCF also takes `apply=METHOD[,METHOD...]`, SCF methods called in order before the
attributes are set (`apply=ddCOSMO scf.with_solvent.eps=78.4` is ddCOSMO water). The hook object
is the PySCF SCF object (its molecule is `obj.mol`), the tblite `Calculator`, the dxtb
calculator, the ASE calculator, or the Psi4 molecule. Values are Python literals when they
parse (`78.4`, `[0,0,0.01]`, `True`), otherwise strings. For example, a field on PySCF through
a hook in `my_hooks.py`:

```bash
mopt-py optmain mol.xyz -func b3lyp -bs def2-svp -calc_opt hook=my_hooks:field hook.f=[0,0,0.001]
```

where `field(mf, f)` modifies `mf` as PySCF's documentation describes (the hook must keep the
energy and its gradient consistent, as PySCF's gradient code reads the modified object).

## Built-in engines by default; what was loaded

By default every program is the one built into mopt-rs (the Rust potentials) or installed with
pip in the running environment; nothing has to be configured. `software_path.conf` (`-osp`,
default `./software_path.conf`, `name::path` per line as in MultiOptPy) is optional: its
entries only override paths (a model or parameter file of a profile, the program paths of
`multioptpy:NAME`). Every run states in `mopt.log` and on screen what it used:

```text
Use pyscf:b3lyp/def2-svp potential.
Calculator: pyscf:b3lyp/def2-svp: pyscf 2.8.0 (/.../site-packages/pyscf/__init__.py)
software_path.conf: not found; using the engines built into mopt-rs and the installed Python packages
```

The calculator line names the distribution, its version, and the file Python imported
(`lj: built into mopt-rs 2.1.1` for the Rust potentials; for ASE calculators the class, e.g.
`pm6_rs.ase.PM6 from pm6-rs 0.3.1 (...)`), followed by any path taken from
`software_path.conf`. The second line says which file was read and which entries it has. From
Python, `calc.describe()` returns the same text, and `Calculator::description()` in Rust.

## Programs run through files

`-os gaussian`, `orca`, `gamessus`, `nwchem`, `mopac`, `gpaw`, and `espresso` set up the
ASE calculator of the program from the options every program shares and from `-calc_opt`
for everything else. The programs themselves and ASE must be installed; the path of a
program is read from `software_path.conf` (`name::path`).

| `-os` | `-func`, `-bs` | memory, processes | `software_path.conf` | `-calc_opt` |
|---|---|---|---|---|
| `gaussian` | method and basis of the route (`def2-SVP` is written `def2SVP`) | `-mem` as `%mem`, `-core` as `%nprocshared` | `gaussian::` the directory of the program (`GAUSS_EXEDIR`) or the program | `program=g09`, any route keyword (`empiricaldispersion=gd3bj`, `scrf=pcm,solvent=water`), `scf=...`, `symmetry=True`, `extra=...` (route text), `addsec=@FILE` (text after the structure), `chk=...`, `hessian_route=...` |
| `orca` | `! method basis` | `nprocs=N`, `maxcore=MB` (`-core` sets `nprocs`) | `orca::` the program or its directory | `simple="D3BJ RIJCOSX"` (more `!` keywords), `blocks="%scf maxiter 300 end"` or `blocks=@FILE`, `hessian_task=NumFreq`, `auto_fix_basis=False`, `heavy_atom_basis=...` |
| `gamessus` | `DFTTYP`; `$BASIS` keywords for STO-3G, Pople, and correlation consistent sets | `system.mwords=N` | `gamessus::` the USERSCR directory of `rungms` | `GROUP.KEY=` for every input group (`contrl.maxit=200`, `basis.gbasis=N31`, `scf.dirscf=True`, `pcm.solvnt=WATER`), `xc=None` (no DFT), `ecp=...`, `command=...` |
| `nwchem` | `xc`, `basis` (`-sub_bs` per element) | `-mem` | (the `nwchem` command of ASE) | `theory=mp2`, `BLOCK.KEY=` for every block (`dft.grid=fine`, `scf.maxiter=200`, `set.KEY=`), `command=...`, the other keywords of ASE's calculator |
| `mopac` | `-func` is the method (`PM7` by default) | `-core` as `THREADS=N` | `mopac::` the program or its directory (otherwise `mopac` on the search path, or ASE's configuration) | `task=...`, `keywords="EPS=78.4 PRECISE"` (written to the keyword line as they are), `threads=N`, `relscf=...`, `command=...` |
| `gpaw` | `xc` | | | every GPAW keyword (`mode=pw` or `mode.name=pw mode.ecut=400`, `kpts`, `h`, `basis`, `occupations.width=0.05`, `txt=gpaw.txt`, `symmetry.point_group=True`, ...); `vacuum=ANGSTROM` around a structure without a cell (6) |
| `espresso` | `input_dft` | | `espresso::` the `pw.x` command, `espresso_pseudo::` the pseudopotential directory | `pseudopotentials.Si=Si.UPF` (required, one per element), `kpts=(4,4,4)`, `input_data.SECTION.KEY=` (`input_data.system.ecutwfc=40`), `command="mpirun -n 4 pw.x"` |

What a program takes is listed by `python -m mopt_rs.calculators.programs orca` (all
programs without a name; `mopt_rs.calculators.describe_program("orca")` in Python). ORCA and
MOPAC take exactly the settings listed, so a misspelt key is an error that shows the
list; the other programs pass everything else on to their ASE calculators.

Settings that do not fit on a command line:

- `KEY=@FILE` gives the text of a file as the value (`-calc_opt blocks=@orca_blocks.inp`,
  `addsec=@gen_basis.txt`); `@@...` is a value that begins with `@`. This works for every
  calculator.
- Defaults of a calculator can be kept in a JSON file, so that they need not be repeated
  on every command line:

  ```json
  {"calculators": {
    "orca": {"simple": "RIJCOSX D3BJ TightSCF", "maxcore": 2000, "blocks": "@orca_blocks.inp"},
    "gamessus": {"contrl": {"maxit": 200}, "system.mwords": 400},
    "espresso": {"pseudopotentials": {"Si": "Si.pbe-n-rrkjus_psl.1.0.0.UPF"},
                 "input_data": {"system": {"ecutwfc": 40}}}
  }}
  ```

  The files are `~/.config/mopt-rs/calculator_options.json`, `mopt_calculator_options.json`
  in the working directory, and those named in `MOPT_CALCULATOR_OPTIONS` (separated like
  `PATH`), later ones ranking higher; `-calc_opt` overrides them key by key, and a
  `GROUP.KEY=` of the command line changes one entry of a group of the file. The name is
  what follows `-os` (any calculator, not only these programs: `pyscf`, `tblite`,
  `ase:module:Class`). The `Calculator:` line of the run log names the file and the keys
  taken from it. A `@FILE` inside such a file is looked up beside it.

- The charge and the multiplicity (`-elec`, `-spin`) reach each program in its own way: the
  charge and multiplicity line of Gaussian and ORCA, `$CONTRL ICHARG` / `MULT` (with
  `SCFTYP=UHF` for open shells) of GAMESS-US, `charge` and the `dft` block (`mult`, `odft`) of
  NWChem, `CHARGE=` and `DOUBLET UHF`, ... of MOPAC, `charge` of GPAW, `tot_charge` and
  `nspin = 2` with `tot_magnetization` of Quantum ESPRESSO (unless `input_data` sets them).
  For GPAW and Quantum ESPRESSO a multiplicity other than one also gives the atoms initial
  magnetic moments that add up to the unpaired electrons (the same on every atom), from
  which GPAW makes the calculation spin-polarized and ASE writes `starting_magnetization`.
- With `-fc N`, Gaussian runs a frequency job (`freq=noraman`) and ORCA a `Freq` job, and
  their force constants are read from the output (`Gaussian.log`, `orca.hess`); for methods
  without analytic second derivatives the programs do their own differences. The other
  programs get finite differences of the forces from mopt-rs.
- Gaussian runs with `nosymm`, so that forces and force constants are in the frame of the
  input coordinates. For elements beyond zinc with a Pople basis set, ORCA gets `def2-SVP`
  for those elements (a `%basis` block), as in MultiOptPy; `auto_fix_basis=False` makes this
  an error instead.
- The programs work in the current directory (`-calc_opt directory=...` for another) under
  fixed file names, one calculation at a time.
- GPAW and Quantum ESPRESSO need a cell. A structure without one is put in a box, `vacuum`
  away from the atoms on every side, and the box of the first structure is kept for the
  structures that follow: with a box that follows the atoms the grid changes from one
  structure to the next, and the energy with it (water, PBE, LCAO `dzp`, `h=0.2`: the
  energy scatters by up to 3 mHartree between boxes with 2 to 8 Angstrom of vacuum, without
  a trend; see the table below). When atoms come closer to a wall than half the vacuum, a
  new box is made and a warning says so; that distance is a guard, not a measured optimum.
- GPAW runs without the symmetry of the point group (`symmetry.point_group=True` turns it
  on): it keeps the symmetry of the first structure and stops when a later one breaks it.
- MOPAC is started with what it prints to the screen thrown away (a line per job); its
  results are read from `mopac.out`.

What was run (`tests/test_programs_run.py`, skipped where a program is missing; conda-forge
packages with ASE 3.29.0):

| Program | Run | What the runs check |
|---|---|---|
| MOPAC 23.2.5 | yes | energy and gradient of water against differences of energies (within 5e-5 Hartree/Bohr), the doublet cation (`DOUBLET UHF`), `keywords="EPS=78.4"`, `-func AM1`, an optimization with `optmain` with defaults from `mopt_calculator_options.json`, and the vibrations from differences of the forces against MOPAC's own `FORCE` calculation (1394.7, 2810.5, 2856.7 against 1394.0, 2810.0, 2856.8 cm^-1) |
| GPAW 26.7.0 | yes | water in a box (LCAO): gradient against differences of energies (within 2e-4), a structure that breaks the symmetry of the first, the fixed box and the new box near a wall; magnetic moments 1.00 and 2.00 for a doublet (OH) and a triplet (O2) from `-spin`; an optimization of H2 with `optmain`; silicon with plane waves and k-points (`mode.name=pw mode.ecut=250 kpts=(2,2,2)`): gradient, the stress against the energies of the scaled cell, and a variable-cell optimization (`optmain -vc`) |
| Quantum ESPRESSO 7.6 | yes | H2 in a box with a pseudopotential made by its own `ld1.x`: energy, gradient against differences of energies (within 2e-4); H2+ as a doublet (`tot_charge`, `nspin`, `tot_magnetization`, total magnetization 1.00); the stress of a crystal of H2 in the run of the energy and the forces (`tstress`), against the energies of the scaled cell |
| ORCA, Gaussian, GAMESS-US, NWChem | no | not run in these tests |

For the programs that were not run, and for every program in the other environments, the
tests check the input files that are written (ASE 3.26 and 3.29, `tests/test_programs.py`,
`tests/test_program_settings.py`) and the two Hessian readers on files written in the
layouts of the programs.

Energy of water against the vacuum of its box (GPAW, PBE, LCAO `dzp`, `h=0.2`), relative
to 8 Angstrom:

| vacuum / Angstrom | 8 | 6 | 5 | 4 | 3 | 2.5 | 2 |
|---|---|---|---|---|---|---|---|
| E - E(8) / mHartree | 0 | -0.13 | +1.96 | -0.51 | +2.76 | +0.33 | -2.81 |

## Any ASE calculator, without code

A program with an ASE calculator works as it is:

```bash
mopt-py optmain mol.xyz -os ase:pm6_rs.ase:PM6 -spin 2 -calc_opt method="pm6-d3h4x"
```

From Python, pass the class, a factory function, or an instance:

```python
from mopt_rs.calculators import create
from mopt_rs.optimize import Optimizer, run

calc = create(MyAseCalculator)              # or create(MyAseCalculator(...)), create("ase:pkg:Cls")
result = run(Optimizer(symbols, positions_bohr, multiplicity=2), calc)
```

mopt-rs finds how the calculator takes the charge and spin (keywords such as `charge`,
`multiplicity`, `uhf`, `spin` from its signature, ASE `default_parameters`, or documentation;
otherwise `atoms.info`) and in which convention, uses its Hessian where it has one (below) or
else finite differences of the forces, and switches to an unrestricted reference for open
shells when its default is restricted. To override the state, pass `StateMapping` to
`AseEngine`, or `-calc_opt charge_keyword=... spin_keyword=... spin_convention=unpaired`.

### Hessians of ASE calculators

ASE defines no Hessian, so most calculators get central differences of their forces (6N force
calls with the step `hessian_step`, 0.001 Bohr). A calculator that has one of its own is
found by itself (`hessian=auto`, the default). At the first Hessian these routes are tried in
this order, and the first that gives a usable matrix is kept for the run:

1. a method `get_hessian`, 2. a method `calculate_hessian`, 3. a method `hessian` (when it is
   callable), each called as `method(atoms)`, or as `method()` when it takes no argument (the
   calculator has just computed the energy and forces of that structure);
4. the `hessian` property, through `get_property("hessian", atoms)`, when the calculator lists
   it in `implemented_properties`;
5. an entry `results["hessian"]` that the calculation of the energy and forces left.

The order follows how the installed packages hand out Hessians: `get_hessian` is ASE's own
name (`ase.calculators.ff.ForceField.get_hessian`, and `ase.vibrations.VibrationsData.
get_hessian` for the result of a vibrational analysis), named like `get_forces` and
`get_stress`; the results are where `ForceField` keeps it. Methods are the public interface
and come first; the results are looked at last because they are a calculator's private store.
A method whose name says that it differentiates numerically (`get_numerical_hessian`, names
containing `numerical`, `finite`, `fd_`) is never called automatically: give its name to use it
(`hessian=get_numerical_hessian`); the engine then says it has a Hessian of its own that is not
analytic (`Capabilities.numerical_hessian`).

The arrays are taken in any of the layouts (3N, 3N), (N, 3, N, 3) (that of
`VibrationsData.get_hessian`), (3N, N, 3), and (N, N, 3, 3), rows and columns ordered atom by
atom; for three atoms, where (N, 3, N, 3) and (N, N, 3, 3) have the same shape, the reading
that is symmetric is taken. Plain arrays are in eV/Angstrom^2 (ASE's units) unless
`hessian_units` says otherwise: one of `eV/Angstrom^2`, `eV/Bohr^2`, `Hartree/Bohr^2`,
`Hartree/Angstrom^2`, `kcal/mol/Angstrom^2`, `kJ/mol/Angstrom^2` (anything else is an error),
converted with the CODATA 2018 constants of the package. A result object instead of an array
is read from an attribute whose name gives the unit (`hessian_ev_per_angstrom2`,
`hessian_hartree_per_bohr2`).

Before a matrix is used it must be finite and symmetric. Its asymmetry
`max |H - H^T| / max |H|` up to 1e-4 is rounding and the symmetric part is used; up to 0.1 the
symmetric part is used with a warning that gives the asymmetry; beyond that it is no Hessian in
a known layout and the route counts as failed. Measured on analytic Hessians of water (PySCF,
def2-SVP): 8e-9 for RHF and 7e-10 to 4e-6 for PBE and B3LYP on the grids 0 to 3, so the first
limit is 25 times the largest of them. The results of ASE 3.26's `ForceField` measure 1.0: its
`get_hessian` asks for a property it does not list, and the matrix it leaves in its results
has only the block of the first atom of each bond with the second (its loops over the atoms of
a term share one iterator), so for it both routes fail and differences are used.

`hessian_check=true` compares, at the first Hessian, `H v` along three random unit vectors `v`
with central differences of the forces (six more force calculations) and writes the largest
relative deviation to the run log. Above 0.05 it warns, gives the factor between the two, and
names the unit of the list that the factor matches (a matrix in Hartree/Bohr^2 read as
eV/Angstrom^2 deviates by a factor of 97.17), or says that no unit explains it (another layout,
a wrong Hessian). It is off by default because of the force calculations.

When the route found fails later, or a route named by the user fails, the engine warns once and
uses differences from then on; `hessian=analytic` makes a calculator without a usable Hessian an
error instead. The run log says what was taken (`Hessian from get_hessian(), layout (N, 3, N,
3), eV/Angstrom^2`, with the routes that failed before it) and the calculator line ends with
`Hessian: ...`. A Hessian of an ASE calculator is that of the positions: in a periodic cell it
has no part along the strain, so a variable-cell run (`-vc`) takes the columns of the cell
from differences of the stress and the forces and the block of the atoms from the calculator.

On the command line, for every engine of the Python side:

| `-calc_opt` key | Values | What |
|---|---|---|
| `hessian` | `auto` (default), `analytic`, `numerical`, and for ASE calculators the name of a method, `property`, `results` | where Hessians come from |
| `hessian_method` | as `hessian` | older spelling of `hessian` (both given must agree) |
| `hessian_units` | the units above | unit of the arrays of an ASE calculator |
| `hessian_check` | `true`, `false` | the check against differences of the forces |
| `hessian_differences` | `central` (default), `o1numhess` | how Hessians are made where the engine has none |
| `hessian_step` | Bohr (default 0.001) | step of those differences |

The built-in engines of the Rust side keep their own keys (`hessian=analytic|numerical` and
`hessian_step` of GFN0-xTB, GFN1-xTB, and GFN-FF).

```bash
mopt-py optmain ts.xyz -opt rsirfo_block_fsb -fc 5 -os ase:my_package.calc:MyCalculator \
    -calc_opt hessian=calculate_hessian hessian_units=Hartree/Bohr^2 hessian_check=true
```

```python
from mopt_rs.calculators import EngineCalculator
from mopt_rs.calculators.ase import AseEngine

engine = AseEngine(MyCalculator, hessian="auto", hessian_units="Hartree/Bohr^2",
                   hessian_check=True)
calc = EngineCalculator(engine)          # or create(MyCalculator, hessian_units=...)
print(engine.describe())                 # ...; Hessian: get_hessian() in Hartree/Bohr^2
```

In Python `CalculatorOptions(hessian_differences="o1numhess")` and
`EngineCalculator(engine, hessian_differences=...)` choose the differences; `hessian_method`
is the older name of both and still accepted.

## An engine of your own by where it is imported from (`-os engine:module:Class`)

A program without an ASE calculator is used through an engine: a class with a method
`compute(structure, hessian)` that returns the energy and the forces or the gradient in
units it names (`mopt_rs.calculators.Engine`; the engines of the package are examples).
On the command line such a class is named by its module, which must be on the search path
of Python:

```bash
mopt-py optmain mol.xyz -os engine:my_module:MyEngine -calc_opt level=2
```

`-calc_opt key=value` are the keywords of the class (or of a function that returns an
engine); the settings shared by all calculators (`-func`, `-bs`, `-core`, ...) are passed
as `options` when it has a parameter of that name. No registration and no change of the
package are needed, and the name works in every command and in the settings files.

## Several structures at once (`-parallel N`)

Where a command has several structures to calculate together, it calculates them at the
same time when they do not depend on each other:

| What | Calculated together |
|---|---|
| the images of a chain (`nebmain`, also with `-dmf`) in Cartesian coordinates | all images of an iteration |
| the two structures of a search between two structures (`ieipmain`, step 1 of `run_autots`) | both |
| the electronic states of a function of several states (`-mf`) at one structure | all states |

They are calculated one after the other where the order matters: followed excited states
(`-es_follow`), ONIOM with microiterations (the relaxation of the environment of an image
starts from the nearest one relaxed before), chains in internal coordinates (`-coord`),
chains on a Gaussian-process surface (`gp=on`), and chains on a function of several states.

`-core` is the number of threads of one calculation, as in MultiOptPy. `-parallel N` is
the number of calculations at the same time:

```bash
mopt-py nebmain path.xyz -os pyscf -func b3lyp -bs def2-SVP -core 2 -parallel 4
```

The calculators written in Rust (the built-in ones) are called from the threads of the
command: without `-parallel` there are as many as `-core` says, with it at least `N`. An
engine of the Python side is one object with one program behind it and takes one call at
a time, so with it the structures wait for each other unless there are copies of it:
`-parallel N` runs up to `N` copies of the calculator, each in a process of its own, and
gives every structure to a copy that is free (`mopt_rs.calculators.WorkerCalculator` in
Python; `-calc_opt workers=N` says the same for the calculator alone). What to know:

- `-core` and `-mem` are those of one copy: `N` times the threads of `-core` should not be
  more than the processors of the machine (`-core 1 -parallel 4`: four calculations of
  one thread each);
- a structure goes to the copy that last had the same structure, else to the free copy
  whose last structure is nearest, else to a new copy. In a chain every image so keeps
  its copy from iteration to iteration, and with it the orbitals its SCF starts from;
- a copy is created from the name of the calculator in a new process, so the calculator
  must be one that a name can create: a built-in name of the Python side, a program, a
  profile, `ase:module:Class`, `engine:module:Class`. An engine object made in a script
  cannot be copied;
- programs that work in a directory with fixed file names ("Programs run through files")
  work in `mopt_worker_<k>/` of their directory, one per copy;
- `-stability N` counts the evaluations of each copy;
- a copy that ends in the middle of a calculation (a program that crashes) is an error of
  that structure, which a chain treats like any failed image, and a new copy takes its
  place.

Tested with a model engine whose calls take a moment (three images inside the engine at
the same moment, the same energies as without workers). Measured with PySCF (HCN to HNC,
B3LYP/def2-SVP, 9 images, 4 iterations, 36 gradients, `-core 1`, on a machine with 8
cores that was running another job): 49.6 s with one calculator, 30.4 s with 2 copies,
21.8 s with 4, and 16.0 s with 9. The energies of the images agreed within 1e-8 Hartree
between the runs for three iterations (the copies start their SCF from other orbitals).
After the redistribution of the images before the fourth iteration they differed by up
to 3e-4 Hartree, in two groups (the runs with 2 and 4 copies agreed with each other, and
the run with 9 copies with the run with one calculator): where the redistribution
(`-adpred`) puts the images along the path made another choice on differences in the
eighth digit. Why it is that sensitive there has not been looked into. What is gained
in general depends on how the program scales with its own threads against how many
images there are.

## Threads of the built-in engines

The built-in engines (GFN0-xTB, GFN1-xTB, GFN-FF, D3, D4, EAM, MEAM, Tersoff, LJ) run in
the threads of the command (`-core`, or the pool of a job of `run_mapper` or of the images
of a chain) and never start threads of their own. Their energies, gradients, Hessians and
stresses are the same bits for any number of threads (`tests/thread_independence.rs`
evaluates them in pools of 1, 2, 4 and 8 threads):

- the dense linear algebra (products, LU and Cholesky factorizations, eigendecompositions)
  splits its work into a fixed number of tasks (`math::linalg::KERNEL_TASKS`, 8) and runs
  small problems on one thread (`PARALLEL_PRODUCT`, `PARALLEL_ORDER`); faer's default, which
  takes the size of the pool, gave other bits with 1 and 8 threads for every GFN0-xTB and
  GFN1-xTB calculation and for GFN-FF from about 300 atoms;
- the loops over atom pairs of GFN-FF run in fixed chunks of atoms whose sums are added in
  a fixed order (`pes::native::parallel`), in parallel from 256 atoms;
- the columns of the analytic Hessians of GFN0-xTB and GFN1-xTB (one perturbation per
  coordinate and per shell potential) and the displaced structures of the Hessians by
  central differences (`-calc_opt hessian=numerical`, D4, O1NumHess) are computed in
  parallel, each column on its own (`pes::native::fd_hessian`); the analytic Hessians hold
  at most as many columns at once as fit in half the free memory.

GFN1-xTB starts its SCC from the converged charges of the last structure it evaluated
(`scc_guess=previous`, the default). Where structures are evaluated at the same time (the
images of a chain, the jobs of `run_mapper` with one calculator), which structure was last
depends on the order in which the threads finish, and so the converged numbers within the
thresholds of the SCC: measured for caffeine and taxol from the charges of structures 0.02
to 0.1 Bohr away, the energy differs from the start of the superposition by at most 7e-13
Hartree and the gradient by at most 6.4e-11 Hartree/Bohr (`examples/bench_scc_start.rs`;
with the charges converged to 1e-12 instead of 1e-10, 3e-13 and 4e-13, at 6 to 8 more
iterations). `-calc_opt scc_guess=sad` starts every SCC from the superposition of atomic
densities: the results are then the same bits whatever was evaluated before and in whatever
order. The displaced structures of a GFN1-xTB Hessian by differences all start from the
charges of the structure itself and leave the start of the next SCC as it is. The SCC of
the lanthanide trichlorides needs 245 to 314 iterations from the superposition, near the
limit of 250 per pass (`Gfn1Xtb::max_iterations` in Rust); the official
program is the better choice where the SCC of a structure does not converge.

## Hessians from a few gradients (O1NumHess)

Central differences take 6N gradients. O1NumHess (Wang, Luo, Wang, Liu 2025) differentiates
the gradient along directions that excite all neighbourhoods of the molecule at once and fits
the Hessian to the result, so the number of gradients is that of the coordinates in the
largest neighbourhood (some tens for organic molecules) whatever the size of the system; the
accuracy is that of single-sided differences with a step of 0.005 Bohr.

- Command line: `-modelhess o1numhess -mfc N` asks for such a Hessian every `N` iterations,
  with any calculator (`-os ...`) and coordinate system. `-fc` (exact Hessians) takes
  precedence. The log has one line per Hessian with the number of gradients.
- Python: `create(name, CalculatorOptions(hessian_differences="o1numhess"))` (or
  `-calc_opt hessian_differences=o1numhess` for an engine of the Python side) or
  `EngineCalculator(engine, hessian_differences="o1numhess")` uses it whenever the engine has
  no Hessian of its own (`hessian_method` is the older name of the setting);
  `Optimizer(..., model_hessian="o1numhess", model_hessian_interval=N)` switches the
  calculators it is run with. `mopt_rs._core.O1NumHess` is the procedure by
  itself: `ask()` returns the displaced positions, `tell()` takes the gradients there.
- Rust: `pes::o1numhess::O1NumHessCalculator` wraps any `Calculator`; `O1NumHess` is the
  state machine, also for plain variables with a distance matrix of one's own.

If the Hessian has negative eigenvalues, a second round differentiates the gradient along
their eigenvectors, so the curvature of a reaction mode is measured rather than fitted.

New programs that are not ASE calculators: subclass `mopt_rs.calculators.Engine` (energy and
gradient in the program's units) and `register` it, or publish an entry point in the group
`mopt_rs.calculators`.

## Machine-learned potentials and other programs by name: profile files

A program with an ASE calculator gets a name of its own from a few lines of JSON, with no
code. Profile files are read from `~/.config/mopt-rs/profiles.json`, from
`mopt_profiles.json` in the working directory, and from the files named in the environment
variable `MOPT_PROFILES` (later ones replace earlier names):

```json
{
  "profiles": {
    "my-nnp": {
      "factory": "some_package.calculator:Calculator",
      "args": [
        {
          "factory": "some_package.pretrained:load",
          "kwargs": {
            "name": {"option": "model", "default": "small"},
            "device": {"option": "device", "default": "cpu"}
          }
        }
      ],
      "kwargs": {"device": {"option": "device", "default": "cpu"}},
      "state": "none",
      "description": "my potential"
    }
  }
}
```

```bash
mopt-py optmain mol.xyz -os my-nnp -calc_opt device=cuda model=large
```

- `factory` is the class or function that returns the ASE calculator; `args` and `kwargs`
  are its arguments.
- An argument written as `{"factory": ...}` is built first: this is the usual two-step
  pattern of such packages (load a model, hand it to the calculator). `"call": "name"` calls a
  method on the result (`model.ase()`).
- `{"option": "device", "default": "cpu"}` takes the value of `-calc_opt device=...`. Options a
  profile does not name go to the calculator as keywords, so every keyword of the calculator
  stays reachable.
- `{"software_path": true}` is the path given for the profile in `software_path.conf`
  (`my-nnp::/models/a.model`).
- `state` says how charge and spin are passed: `"none"` (default), `"charge"`,
  `"multiplicity"`, `"unpaired"`, or `{"charge": KEYWORD, "spin": KEYWORD, "convention": ...,
  "via": "kwargs" | "info"}`. `hessian` names the calculator's Hessian method (default:
  finite differences of the forces).

One calculator, and so one loaded model, serves every structure of a run. The built-in
profiles (`mace_mp`, `mace_off`, the `uma-*` models, the `*_rs` engines) work the same way and
can be replaced by a profile of the same name. `tests/test_profile_files.py` runs all of this
with a stand-in potential (`tests/fake_nnp.py`).

**AIMNet2** (package `aimnet`, MIT; `pip install "aimnet[ase]"` with PyTorch) has the profiles
`aimnet2` (wB97M-D3, closed shell), `aimnet2-nse` (open shell), `aimnet2-rxn` (reactions of
neutral molecules), `aimnet2-b973c`, and `aimnet2-pd`; the model is downloaded on first use and
checked against its SHA-256. Charge and multiplicity go to the calculator's `charge` and
`mult`, and its own Hessian (`get_hessian`) is used. `aimnet2` ignores the multiplicity (the
package says so), so an open shell needs `aimnet2-nse`. Tried with aimnet 0.2.5 and PyTorch
2.15 on the CPU (`tests/test_nnp_aimnet2.py`, skipped without the package):

| Check | Result |
|---|---|
| Energy and gradient against the calculator itself (water, hydroxide, the methyl radical) | 3e-7 to 6e-7 Hartree (8e-9 of the energy: the eV to Hartree factors of two CODATA versions), gradient 4e-10 Hartree/Bohr |
| Gradient against central differences of the energy (step 1e-3 Bohr) | 3.4e-5 Hartree/Bohr |
| `optmain` of ethanol | converged with 16 gradients (13 s) |
| `freqmain` of the optimized ethanol | 21 real frequencies, 219 to 3075 cm^-1 |
| `optmain -order 1 -fc 5 -freq` from a bent HCN | converged with 8 gradients and 3 Hessians of the model; one imaginary frequency, 2006i cm^-1 |
| The methyl radical with `aimnet2-nse` | the doublet and the quartet differ, without the warning of the closed-shell model |

## Parts of the library one by one from Python

Besides the optimizers, these parts can be called on their own (`mopt_rs._core`; positions in
Bohr as flat lists, atoms numbered from 1):

| Part | Python |
|---|---|
| coordinate systems (`cart`, `ric`, `dic`, `tric`, `tric-p`, `auto`): values, Wilson matrix, displacements, Cartesian step of a coordinate step, setting up anew | `Coordinates(name, symbols, positions)`, `coordinate_names()` |
| step methods (`-opt` names) and Hessian updates | `StepMethod(name, n)` |
| built-in calculators, model Hessians, dispersion | `evaluate_native`, `NativeEvaluator`, `model_hessian`, `dispersion` |
| bias potentials, terms of one's own, constraints | `evaluate_bias`, `check_term`, `check_bias`, `constraint_values` |
| superposition of structures, bond graph | `align`, `connectivity` |
| which minimum to sample next | `SamplingPriority(name, seed)`, `sampling_priority_names()` |
| electron-transfer rates (Marcus with Landau-Zener transmission) | `electron_transfer_rate` |
| electron transfer one part at a time (`mopt_rs.electron_transfer`): the inner sphere at a crossing, Scher-Holstein, the Marcus, Zhu-Nakamura, and Marcus-Levich-Jortner rates, Mulliken-Hush and its generalization, the charge-transfer index, the shifts of SAES, the apparent barrier of a rate, the transfer integral of the projection method, the four-point reorganization energy, the hopping rate, the fragment charge difference, the proton states and the rate of a proton-coupled electron transfer; `etmain` with its results (both methods) | `et_inner_sphere`, `et_scher_holstein`, `et_marcus_rate`, `et_zhu_nakamura_rate`, `et_marcus_levich_jortner`, `et_mulliken_hush`, `et_generalized_mulliken_hush`, `et_charge_transfer`, `et_solvent_shifts`, `et_apparent_barrier`, `et_transfer_integral`, `et_fock_from_orbitals`, `et_projection_coupling`, `et_four_point_reorganization`, `et_hopping_rate`, `et_fragment_charge_difference`, `et_proton_states`, `et_pcet_rate`, `run_etmain` |
| O1NumHess: a Hessian from a few gradients (`ask()` / `tell()`) | `O1NumHess(symbols, positions, gradient)` |
| objectives of plain variables | `evaluate_objective`, `objective_names()` |
| normal modes of a structure with its Hessian, point group, rotational constants, thermochemistry (the choices of `-freq` as `options`) | `normal_modes`, `point_group`, `rotational_constants`, `vibrational_analysis` |
| following orbitals between structures, natural reaction orbitals, reactive orbitals | `follow_orbitals`, `natural_reaction_orbitals`, `mass_weighted_displacement`, `reactive_orbitals`; the orbitals of a structure: `EngineCalculator.orbitals(structure)` |
| distances between structures, multidimensional scaling, principal components, dynamic modes of iterations | `structure_distances`, `classical_mds`, `path_variables`, `principal_components`, `koopman_observables`, `dynamic_modes` |

`tools/api/parity.json` lists every module of the Rust library with the Python names that
reach it, what exists on one side only and why (ASE and the other Python programs; the
infrastructure of the Rust side), and what Python cannot reach yet; `tests/test_api_parity.py`
and `crates/mopt-rs/tests/api_parity.rs` fail when a new module or a new Python name is
missing from it.
