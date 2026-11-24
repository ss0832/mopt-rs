# Parameter provenance

Every parameter table compiled into mopt-rs is listed here with its source. Entries are added in
the same change that introduces the parameters.

## Native calculators

### Lennard-Jones (`pes::native::lennard_jones`)

- UFF noble-gas parameters `x_i` (Angstrom) and `D_i` (kcal/mol) for He, Ne, Ar, Kr, Xe, Rn:
  A. K. Rappe, C. J. Casewit, K. S. Colwell, W. A. Goddard III, W. M. Skiff,
  *J. Am. Chem. Soc.* **1992**, 114, 10024-10035, doi:10.1021/ja00051a040.
  `sigma = x_i / 2^(1/6)`, `epsilon = D_i`, Lorentz-Berthelot combination, as in MultiOptPy's
  `lj_calculation_tools.py` (Rn is MultiOptPy's extrapolated value).

### Effective medium theory (`pes::native::emt`)

- Parameters `E0, s0, V0, eta2, kappa, lambda, n0` for Al, Cu, Ag, Au, Ni, Pd, Pt:
  K. W. Jacobsen, P. Stoltze, J. K. Norskov, *Surf. Sci.* **1996**, 366, 394-402,
  doi:10.1016/0039-6028(96)00816-3, as tabulated in ASE's `ase/calculators/emt.py`
  (LGPL-2.1-or-later). The H, C, N, O entries are ASE's non-production extras and are flagged as
  such in ASE.
- The cutoff construction (global cutoff from the largest `s0`, logistic smoothing reaching
  `1e-4` at the fourth-neighbor fcc distance, list cutoff `rc + 0.5` Angstrom), `beta = 1.809`,
  and the ASAP energy convention follow ASE. The parameter table is converted from Bohr with ASE's
  Bohr value (CODATA 2014, `0.5291772105638411` Angstrom) so that results reproduce ASE.
- The formulas are implemented independently in Rust (no ASE code is included); agreement with
  ASE is tested to `1e-15` Hartree (`tests/golden_ase.rs`).

### Tersoff (`pes::native::tersoff`)

- Single-element parameters for C, Si, Ge and the mixing rules (`A_ij = sqrt(A_i A_j)`,
  `B_ij = chi_ij sqrt(B_i B_j)`, arithmetic means of `lambda` and `mu`, geometric means of `R` and
  `S`) with `chi_SiC = 0.9776`, `chi_SiGe = 1.00061`:
  J. Tersoff, *Phys. Rev. B* **1989**, 39, 5566-5568, doi:10.1103/PhysRevB.39.5566.
- The Si set coincides with Tersoff's Si(C) (*Phys. Rev. B* **1988**, 38, 9902).
- The functional form and the triple-keyed parameter convention follow LAMMPS
  (`pair_style tersoff`) and ASE's `ase/calculators/tersoff.py`; LAMMPS-format files can be
  loaded with `Tersoff::from_lammps_str`.
- C-Ge has no parameters in Tersoff (1989) and is rejected rather than guessed. MultiOptPy's
  hetero tables (`Si-C`, `Si-Ge`, `C-Ge`) have no cited source and are not used.

### Embedded-atom method (`pes::native::eam`)

- Functional form and parameters for Cu, Ag, Au, Ni, Pd, Pt, Al, Pb, Fe, Mo, Ta, W, Mg, Co,
  Ti, Zr: X. W. Zhou, R. A. Johnson, H. N. G. Wadley, *Phys. Rev. B* **2004**, 69, 144113,
  doi:10.1103/PhysRevB.69.144113.
- The numbers were taken from the parameter database of the generator of tabulated EAM files
  written by X. W. Zhou and distributed with LAMMPS (`tools/eam_database`), which carries the
  table of the paper, the lower bound `0.25 rho_e` of the middle branch of the embedding
  function of Pt, and the cutoff `sqrt(10) / 2` lattice constants. Only the numbers and the
  formulas are used; the code is written for mopt-rs. Chromium, which that database adds
  from a later source, is not included.
- The transcription is checked by the cohesive energies and equilibrium distances of the
  sixteen crystals (see `docs/source/migration_surfaces.md`, "Embedded-atom method").

### Modified embedded-atom method (`pes::native::meam`)

- Method: M. I. Baskes, *Phys. Rev. B* **1992**, 46, 2727, doi:10.1103/PhysRevB.46.2727;
  screening: M. I. Baskes, J. E. Angelo, C. L. Bisson, *Modell. Simul. Mater. Sci. Eng.*
  **1994**, 2, 505; universal equation of state: J. H. Rose, J. R. Smith, F. Guinea,
  J. Ferrante, *Phys. Rev. B* **1984**, 29, 2963; nuclear repulsion at short distances:
  J. F. Ziegler, J. P. Biersack, U. Littmark, *The Stopping and Range of Ions in Matter*,
  1985.
- Parameters of Cu, Ag, Au, Ni, Pd, Pt, Al, Pb, Rh, Ir, Li, Na, K, V, Nb, Ta, Cr, Mo, W, Fe,
  Mg, Si, C, Sn: the first entry of each element in `potentials/library.meam` of LAMMPS
  ("meam data from vax files fcc,bcc,dia", contributed by Greg Wagner, citing Baskes 1992),
  the entry LAMMPS itself uses for these names.
- The conventions (which sums enter the partial densities, the `3/5` term and the raised
  `t1`, the forms of `G`, the cutoff function, the default screening limits, the blend
  with the nuclear repulsion, the default reference of unlike pairs) follow the MEAM
  package of LAMMPS, read as a description of the model the parameters belong to. Only
  numbers and formulas are used; the code is written for mopt-rs.
- The transcription is checked by the cohesive energies of the reference crystals and by
  the elastic constants of the cubic metals (see `docs/source/migration_surfaces.md`, "Modified
  embedded-atom method").

## Model Hessians (`model_hessian`, `parameters::model_hessian`)

- Lindh models (`lindh`, `lindh2007*`, `xtb_lindh*`) and the Swart model of xtb (`xtb_swart`,
  `swartd2`): the internal-coordinate traversal, the 1995 and 2007 `r_ref` / `alpha` / `d_ref`
  tables, the D2 term (C6 in xtb's units, radii, damping 20, `s6 = 20`), the default force
  constants, and the 2009 covalent radii with metals decreased by 10% follow xtb
  (`src/model_hessian/{internal,lindh,swart,shared}.f90`, `src/param/model_hessian.f90`,
  `src/setparam.f90`, `src/mctc/param/covalent_radii.f90`; grimme-lab/xtb, LGPL-3.0-or-later).
  The routines are reimplemented in Rust; the tables are copied.
  Original model: R. Lindh, A. Bernhardsson, G. Karlstrom, P.-A. Malmqvist, *Chem. Phys. Lett.*
  **1995**, 241, 423-428.
- O1NumHess Swart model (`swart`): B. Wang, S. Luo, Z. Wang, W. Liu, J. Chem. Theory Comput.
  2025, 21, 10893, arXiv:2508.07544 (Appendix B),
  reference implementation `O1NumHess_QC/Swart.py` (MIT, Copyright 2025 Luo Shaohang, Wang
  Zikuan, Shandong University), including its covalent radii; tested against the H2O2 Hessian
  printed there. Original model: M. Swart, F. M. Bickelhaupt, *Int. J. Quantum Chem.* **2006**,
  106, 2536.
- Schlegel (`schlegel`): H. B. Schlegel, *Theor. Chim. Acta* **1984**, 66, 333, with the stretch
  parameters of J. M. Wittbrodt, H. B. Schlegel, *J. Mol. Struct. (Theochem)* **1997**, 398-399,
  55 (as listed in Schrodinger Jaguar's `hess_guess` documentation).
- Fischer (`fischer`): T. H. Fischer, J. Almlof, *J. Phys. Chem.* **1992**, 96, 9768, in the form
  of pysisyphus' `guess_hessians.py`.

## DFT-D3(BJ) (`pes::dispersion::d3`, `parameters::d3_reference`)

- Reference systems, their coordination numbers, the reference C6 coefficients, and the
  `<r^4>/<r^2>` values are generated by `tools/params/gen_d3.py` from simple-dftd3
  (`src/dftd3/reference.f90`, `src/dftd3/data/r4r2.f90`; dftd3/simple-dftd3, LGPL-3.0-or-later).
  D3 covalent radii: 4/3 of the Pyykko-Atsumi 2009 radii with metals decreased by 10% (mctc-lib
  `covalent_rad_d3`, identical to the xtb table above). The counting function (`k = 16`),
  reference weighting (`w = 4`), cutoffs (40 / 60 Bohr), and rational damping follow
  simple-dftd3; the implementation (including the exact Hessian) is independent Rust code.
- S. Grimme, J. Antony, S. Ehrlich, H. Krieg, *J. Chem. Phys.* **2010**, 132, 154104;
  S. Grimme, S. Ehrlich, L. Goerigk, *J. Comput. Chem.* **2011**, 32, 1456. PBE parameters
  `s6 = 1`, `s8 = 0.7875`, `a1 = 0.4289`, `a2 = 4.4407`. Tested against simple-dftd3's reference
  energy for mstore MB16-43 #01 (`test_pbed3bj_mb01`).

## DFT-D4 and EEQ-2019 (`pes::dispersion::d4`, `parameters::d4_reference`)

- Reference systems (covalent CNs, EEQ charges, secondary systems, polarizabilities at 23
  imaginary frequencies), chemical hardnesses, effective nuclear charges, Pauling
  electronegativities, and `<r^4>/<r^2>` are generated by `tools/params/gen_d4.py` from dftd4
  (`src/dftd4/reference.inc`, `src/dftd4/data/*.f90`; dftd4/dftd4, LGPL-3.0-or-later); the
  EEQ-2019 parameters (`chi`, `eta`, `kappa`, `sigma`) from multicharge
  (`src/multicharge/param/eeq2019.f90`; grimme-lab/multicharge, LGPL-3.0-or-later). Entries
  missing from `reference.inc` are zero, as in dftd4's static arrays.
- Model constants (`ga = 3`, `gc = 2`, `wf = 6`, Gaussian-weight counts, the electronegativity
  factor `k4`, `k5`, `k6`, `kcn = 7.5`, the EEQ CN cap at 8, cutoffs 30 / 25 / 60 / 40 Bohr) follow
  dftd4, multicharge, and mctc-lib; the implementation (analytic gradient with the charge
  response by an adjoint solve) is independent Rust code.
- E. Caldeweyher, S. Ehlert, A. Hansen, H. Neugebauer, S. Spicher, C. Bannwarth, S. Grimme,
  *J. Chem. Phys.* **2019**, 150, 154122; EEQ: E. Caldeweyher, C. Bannwarth, S. Grimme,
  *J. Chem. Phys.* **2017**, 147, 034112. PBE parameters `s8 = 0.95948085`, `a1 = 0.38574991`,
  `a2 = 4.80688534`, `s9 = 1`, `alpha = 16`. Tested against dftd4's reference energy for mstore
  MB16-43 #01 (`test_pbed4_mb01`).

## MultiOptPy's legacy model Hessian tables

- `crates/mopt-rs/src/parameters/multioptpy_tables.rs` and `multioptpy_gfn.rs` are generated
  by `tools/params/gen_multioptpy_tables.py` and `gen_multioptpy_gfn.py` from MultiOptPy 1.21.2
  (`multioptpy/Parameters/*.py`, read with `ast`, never executed; MultiOptPy is by the same
  author, GPL-3.0-or-later). They serve only the `legacy_` models, which reproduce MultiOptPy.
- `multioptpy_gfn.rs` holds MultiOptPy's `GFNFFParameters` and `GFN0Parameters`: MultiOptPy's
  own approximate values for its GFN-FF-like and GFN0-like model Hessians. They are not the
  published GFN-FF (S. Spicher, S. Grimme, *Angew. Chem. Int. Ed.* **2020**, 59, 15665) or
  GFN0-xTB (P. Pracht, E. Caldeweyher, S. Ehlert, S. Grimme, ChemRxiv 2019,
  DOI 10.26434/chemrxiv.8326202.v1) parameter sets, although MultiOptPy labels some tables as
  taken from `gfnff_param.f90` / `gfn0_param.f90`. mopt-rs uses them for the elements of
  MultiOptPy's tables: H to Rn for the GFN-FF-like model, H to Xe (the `rad` and `en` tables)
  for the GFN0-like model.

## GFN0-xTB

- `crates/mopt-rs/src/parameters/gfn0_xtb.rs` is generated by `tools/params/gen_gfn0_xtb.py`
  for H to Rn from the upstream snapshots that gfn0-rs
  (github.com/ss0832/gfn0-rs, GPL-3.0-or-later) pins, with their SHA-256 in the file header:
  `param_gfn0-xtb.txt` (grimme-lab/xtb, LGPL-3.0-or-later; the fitted GFN0-xTB parameters),
  `src/approxrab.f90` and `src/disp/ncoord.f90` (grimme-lab/xtb, LGPL-3.0-or-later; the SRB
  data and the Pauling electronegativities), and `src/multicharge/param/eeq2019.f90`
  (grimme-lab/multicharge, Apache-2.0; the EEQ-2019 charges of the D4 term). The atomic and D3
  covalent radii of the elements are xtb's tables (`atomicrad.f90`, `covalentradd3.f90`),
  converted with xtb's legacy Bohr radius 0.52917726; the STO-nG tables are Stewart's as in
  xtb's `slater.f90`. The D4 reference data are those of mopt-rs's DFT-D4.
- Method: P. Pracht, E. Caldeweyher, S. Ehlert, S. Grimme, ChemRxiv 2019,
  DOI 10.26434/chemrxiv.8326202.v1. The implementation (`pes/native/gfn0/`) follows gfn0-rs,
  which reproduces xtb; checked against gfn0-rs by `crates/mopt-rs/tests/gfn0_xtb.rs`.

- The excited states of `-os gfn0xtb -es N` (`crates/mopt-rs/src/pes/native/gfn0/excited.rs`)
  are an addition of mopt-rs and not part of GFN0-xTB or of xtb. The form of the response
  matrix, the damped Coulomb interaction of Loewdin transition charges, and its exponent
  1.42 are those published for the simplified Tamm-Dancoff approximation (S. Grimme,
  *J. Chem. Phys.* **2013**, 138, 244104), taken without exact exchange; the atomic
  hardnesses are those of the charge model of GFN0-xTB above. It is written from the
  published formulas; no code of the sTDA program is used, and nothing is fitted.

## GFN1-xTB

- `crates/mopt-rs/src/pes/native/gfn1/elements.rs` is generated by
  `tools/params/gen_gfn1_xtb.py` for H to Rn from grimme-lab/xtb 6.7.1
  (LGPL-3.0-or-later), read as text, with the SHA-256 of each file in the header:
  `src/xtb/gfn1.f90` (shells, Slater exponents, self energies, shell polynomials, hardnesses
  and their shell scaling, third-order parameters, repulsion exponents and effective charges,
  halogen-bond strengths, the element kinds whose self energies shift with the coordination
  number) and `src/param/` (`paulingen.f90`, `atomicrad.f90`, `covalentradd3.f90`). The
  generator compares every value with the parameters that tblite exports
  (`tblite.library.export_gfn1_param`) and fails on a difference, including tblite's pair
  parameters (`kpair`) against the rule of `pair_scaling`; all agree.
  `crates/mopt-rs/src/pes/native/gfn1/params.rs` holds the global constants of `gfn1.f90`
  (`kshell`, `ksp`, `kdiff`, `enshell`, the H-H, B-H, N-H, B-P, N-Si, Ni-H, Re-H and Pt-H pair
  parameters and the scaling of pairs of d-block metals by their rows,
  `xbdamp`, `xbrad`, the D3(BJ) parameters). The STO-nG expansions are Stewart's (R. F. Stewart, *J. Chem.
  Phys.* **1970**, 52, 431) as xtb's `slater.f90` has them, except 4s and 4p, where tblite
  0.4.0 uses Stewart's original values and xtb older ones (tblite is followed). The D3
  reference C6 data are those of mopt-rs's DFT-D3 above.
- Where tblite 0.4.0 (LGPL-3.0-or-later) and xtb differ, tblite is followed (it is the
  reference of the tests): the Bohr radius 0.52917721090380 Angstrom for the parameters given
  in Angstrom, the Boltzmann constant 3.166808578545117e-6 Hartree/K (300 K), the cutoffs of
  the coordination number (25 Bohr) and of the dispersion (50 Bohr), and the 4s/4p sets.
- Method: S. Grimme, C. Bannwarth, P. Shushkov, *J. Chem. Theory Comput.* **2017**, 13, 1989.
  The implementation (`pes/native/gfn1/`) follows the formulas of the xtb code (written from a
  specification of it and a prototype checked against tblite, not translated line by line);
  `crates/mopt-rs/tests/gfn1_xtb.rs` compares it with tblite's energies, gradients, and
  charges of 37 systems (`tests/golden/gfn1_xtb.json`, made by `tools/golden/gfn1_xtb.py`; the
  first 18 from tblite 0.4.0, the others from tblite 0.8.0, which agrees with 0.4.0 to 1e-14
  Hartree for the light elements and moves the electronic energies of Br and I by up to 2e-5
  Hartree, so those records of 0.8.0 are compared in their classical terms). BF4- is not
  among them: tblite 0.8.0 converges it to B -5.8 e (1.66 Hartree below the physical
  solution, the third-order term being unbounded below); mopt-rs's test checks that its SCC
  stays on the physical solution.

## GFN-FF

- `crates/mopt-rs/src/parameters/gfnff.rs` is generated by `tools/params/gen_gfnff.py` for
  H to Rn (Z = 1 to 86) from grimme-lab/xtb (LGPL-3.0-or-later), read as text, with
  the SHA-256 of each file in the header. The source tree read is a later state than the
  6.7.1 release (it extends GFN-FF to the lanthanides and actinides and refits La to Lu), so
  the lanthanide entries of `chi`, `gam`, `cnf`, `alp` (`gfnff_param.f90`) and `r0`
  (`gfnff_rab.f`) are taken from the xtb 6.7.1 library, read as data and never executed (every
  other entry of those tables is checked to agree with the source); all other values are those
  of the source: `src/gfnff/gfnff_param.f90` (the `angewChem2020`
  element parameters and the fixed tables `en`, `rad`, `metal`, `group`, `normcn`, `repz`),
  `src/param/covalentradd3.f90` and `src/param/sqrtzr4r2.f90` (D3 covalent radii and
  `<r^4>/<r^2>`), `src/gfnff/gfnff_rab.f` (the CN-dependent bond-length estimate), and
  `include/param_ref.fh` (the D4 reference systems, from which the reference C6 coefficients of
  GFN-FF's dispersion model, xtb's `newD3Model`, are integrated for these elements). The
  global constants of `newGFNFFGenerator` and `gfnff_set_param` are in
  `pes/native/gfnff/constants.rs`.
- Method: S. Spicher, S. Grimme, *Angew. Chem. Int. Ed.* **2020**, 59, 15665
  (DOI 10.1002/anie.202004239). The implementation (`pes/native/gfnff/`) is a port of xtb's
  `src/gfnff` (`gfnff_ini.f90`, `gfnff_ini2.F90`, `gfnff_eg.f90`, `gdisp0.f90`, `gfnff_rab.f`;
  version `angewChem2020_2`, accuracy 0.1) for isolated molecules; xtb's own conventions are kept
  where they decide the topology (its legacy Bohr radius 0.52917726, the `real(sp)` topological
  distances, `NORM2` and `indexHeapSort` for orderings by distance). Checked term by term against
  the xtb executable (`tools/golden/gfnff.py`, `tests/golden/gfnff.json`) and xtb's
  formaldehyde-water unit-test reference by `crates/mopt-rs/tests/gfnff.rs`.

## Van der Waals radii (`parameters/vdw.rs`)

- S. Alvarez, "A cartography of the van der Waals territories", Dalton Trans. 2013, 42,
  8617-8636 (DOI: 10.1039/C3DT50599E), as tabulated in ASE (`ase/data/vdw_alvarez.py`,
  read on 2026-11-15), `Z = 1..=99`; elements without a value there have none. Used by the
  flat-bottom elastic network model (`path/interp/fbenm.rs`) for the nearest approach of
  atoms that are not bonded.

