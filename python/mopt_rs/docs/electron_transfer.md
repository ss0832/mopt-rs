# Electron transfer: EDEEL, SAES, barriers, and rate constants

`etmain` runs the whole chain of an electron-transfer rate constant from one structure: the
diabatic states of EDEEL from fragments that it builds itself, the equilibria of the initial
and final states, the crossings of the two states along the solvent coordinate, the
couplings, the barriers (electronic and free energies), and the rate constants by Marcus
theory with the nuclear tunnelling of Scher and Holstein, by Zhu-Nakamura theory, and by
Marcus-Levich-Jortner. Its parts are also there one by one, in Rust and in Python
(`mopt_rs.electron_transfer`).

```bash
# The radical anion of a donor-bridge-acceptor molecule, GFN1-xTB, a solvent of
# lambda_s = 0.6 eV for both states, the scan of the thesis (alpha_SA = 1, 0.9, ..., 0).
mopt etmain anion.xyz -os gfn1xtb -elec -1 -spin 2 \
    -et_donor 1-8 -et_acceptor 20-31 -et_lambda_s 0.6 0.6
```

## Methods and their sources

- **EDEEL between separate molecules**: A. Mutsuji, K. Saita, S. Maeda, RSC Adv. 2023, 13,
  32097, and chapter 3 of A. Mutsuji, PhD thesis, Hokkaido University, 2025
  (doi:10.14943/doctoral.k16373): `V11 = E^C_{n+m} - E^D_n + E^D_{n+1}`, `V22 = E^C_{n+m} -
  E^A_m + E^A_{m+1}` (`pes::combine::edeel_states`).
- **EDEEL within a molecule of donor, bridge, and acceptor**: chapter 4 of the thesis, eqn
  (4.3) and (4.4), with the donor-bridge and bridge-acceptor fragments, the bonds each cuts
  capped by hydrogens (`pes::combine::{EdeelLayout::bridged, edeel_with}`,
  `pes::combine::Caps`; the caps are those of ONIOM, so gradients and Hessians carry over).
  **Chapter 4 has not been published in a peer-reviewed journal**; the method here follows the
  thesis as written, and what the thesis leaves open is decided below and marked as such.
- **SAES**, the solvent activation energy shift (`kinetics::rate::solvent_activation`): thesis
  eqn (4.5)-(4.11), with the decomposition of the barrier into the solvent, the donor, and the
  acceptor (eqn (4.S1)-(4.S3)). Thesis only.
- **The inner sphere at a crossing and the tunnelling of Scher and Holstein**
  (`kinetics::rate::inner_sphere`): thesis eqn (4.12)-(4.17) and appendix 4.7 (eqn
  (4.S4)-(4.S9)), after H. Scher and T. Holstein, Philos. Mag. B 1981, 44, 343.
- **Couplings** (`kinetics::rate::couplings`, `workflow::electron_transfer`): the adiabatic
  one of EDEEL (`V12 = V11 - E^C_{n+m+1}`, RSC Adv. 2023), Mulliken-Hush (thesis eqn (4.18))
  with the charge-transfer index of T. Le Bahers, C. Adamo, I. Ciofini (J. Chem. Theory
  Comput. 2011, 7, 2498) for the distance and the threshold `q_CT >= 0.985` of the thesis for
  the charge-transfer state, the generalized Mulliken-Hush of R. J. Cave and M. D. Newton
  (Chem. Phys. Lett. 1996, 249, 15; `-et_coupling gmh`: `V12 = |mu_12| E / sqrt(dmu^2 + 4
  mu_12^2)` with the transition dipole and the change of the dipole projected on that
  change), and the fragment charge difference of A. A. Voityuk and N. Rosch (J. Chem. Phys.
  2002, 117, 5607; `-et_coupling fcd`: the same form with the differences of the electrons on
  the donor and the acceptor atoms, of the two states and of the transition density; a
  bridge counts for neither). GMH and FCD take the charge-transfer state as Mulliken-Hush
  does, and neither needs the two states to be localized: on a two-state model both give the
  coupling back exactly (`kinetics::rate::couplings` tests). A crossing where no state
  passes the threshold takes the mean of the couplings of the others, as the thesis does,
  and says so.
- **The excited states for the couplings**: the built-in GFN1-xTB gives the Tamm-Dancoff
  states of its charge response for a reference of any spin
  (`pes::native::unrestricted_tda`; the closed-shell singlets and triplets come out as a
  check), with transition dipoles from the dipole integrals of its basis, the changes of the
  atomic populations and of the dipole from the unrelaxed difference density, and the Mulliken
  populations of the transition density (they add up to zero; a triplet has none). PySCF does
  the same with its TDA (`calculators.pyscf`); a Python engine adds `"dipole_change"` and
  `"transition_population"` to its transitions. Other calculators give none, and
  `-et_coupling auto` then takes EDEEL's coupling. EDEEL's coupling is the difference of a diabatic energy and the
  adiabatic one, so it takes up the delocalization error of the calculator in the adiabatic
  state: with GFN1-xTB on the cofacial dimer cations of HAB11 it is about 1 eV and grows with
  the distance, where the references fall from 520 to 40 meV (see the HAB11 section below);
  a large EDEEL coupling with GFN-xTB or a semilocal functional is to be read with that in
  mind.
- **Rate constants**:
  - Marcus with the tunnelling factor (thesis eqn (4.1)), with the total reorganization
    energy `lambda = lambda_in + (lambda_s^i + lambda_s^f) / 2` (eqn (4.17));
  - Zhu-Nakamura at the crossing (new in mopt-rs): the thermal transmission of a peaked
    (normal region; tunnelling below the crossing included) or sloped (inverted region)
    crossing of `kinetics::rate::{nonadiabatic_tunnelling, zhu_nakamura}` along the
    mass-weighted difference gradient, `k = kappa (k_B T / h) exp(-dE / k_B T) / Q_n` with the
    electronic barrier, `Q_n` the partition function of the mode in the initial well (which
    the electronic barrier leaves out and the Franck-Condon factor of the Marcus rate has;
    without it the rate came out `Q_n` times too large, 1.2 for a mode of 175 1/cm and 3.4 for
    60 1/cm at room temperature), and `k = kappa (k_B T / h) exp(-dG / k_B T)` with the free
    energy of activation, whose partition functions have the mode. At weak coupling in the
    normal region it is the Marcus rate with the tunnelling factor of the mode (the tests:
    within 10 per cent of Scher-Holstein's for 175 1/cm, where that factor is 1.45). **In the
    inverted region (a sloped crossing) it is low at weak coupling**: the energies below the
    crossing are not built for that type (as for the crossing edges of `run_kinetics`), and
    the Zhu-Nakamura formula above the crossing (Lehto, arXiv:1304.3331, eqn (29), with
    `0.4 a^2 + 0.7`) departs from Landau-Zener up to `b^4 ~ 0.4 a^2`, which at a coupling of
    `1e-6` Hartree reaches 12 `k_B T` above the crossing; there the rate is 4.5 times below
    Marcus's. The energies below the crossing of the Landau-Zener type (Ishida, Nanbu,
    Nakamura, Int. Rev. Phys. Chem. 2017, 36, 229, eqns (2.41)-(2.53)) are planned;
  - Marcus-Levich-Jortner (J. Jortner, J. Chem. Phys. 1976, 64, 4860; with the mode in its
    ground state, J. Ulstrup, J. Jortner, J. Chem. Phys. 1975, 63, 4358): the mode of the inner
    sphere as one quantum mode with its thermal populations and the solvent classical, from the
    same inputs as the Scher-Holstein factor; it needs a solvent (`-et_lambda_s`). The thermal
    Franck-Condon weights of the mode are integrals of its generating function on the circle
    through the saddle point, where they do not oscillate; a soft mode gives the classical
    Marcus rate of `lambda_s + lambda_in`, a stiff one at low temperature the sum over the quanta
    of its ground state;
  - the Landau-Zener transmission of the RSC Adv. paper (eqn (1)-(5)) as a part
    (`kinetics::rate::electron_transfer::edeel_rate`).

## Barriers

Each crossing reports:

- **`dE`** (`barrier_eV`): the electronic barrier with the solvent activation, `V11(SX) -
  V11(EQ_i) + G_SA^i`, as in the thesis.
- **`dG`** (`free_barrier_eV`; `-et_free_energy on`, the default): the free energy of
  activation `G(SX) - G(EQ_i)`, the harmonic free energies (rigid rotor, the quasi-harmonic
  correction and the other settings of `-freq`) of the initial state at its equilibrium
  from its normal modes and of the crossing from the vibrations within the seam of the two
  states (`vib::seam_analysis`: the direction across the seam left out, as for the crossing
  points of nonadiabatic transition state theory), with the solvent shift at the crossing.
  It needs the Hessians of both states there; `-et_free_energy off` leaves them out.
  Imaginary frequencies within the seam (a crossing that is not a minimum of it) are counted
  (`seam_imaginary`), left out, and warned about. The RSC Adv. paper works with free energies
  of this kind; the thesis with electronic energies, so both are given.
- **The apparent free energy of activation** of each rate constant,
  `-k_B T ln(k h / k_B T)` (`apparent_barrier_marcus_eV`, `apparent_barrier_zn_eV`; in the
  summary for the largest rate constants, in eV and kcal/mol): the barrier that the Eyring
  equation gives the rate constant, for comparison with measurements.

The rate constants are given with `dE` (`log10_k_marcus`, `log10_k_zn`) and with `dG`
(`log10_k_marcus_G`, `log10_k_zn_G`).

## What the thesis leaves open, and how it is decided here

- **The solvent reaction energy `dG_s0` of eqn (4.8)** is drawn in Fig. 4.3 but not defined in
  the text. The thesis says that at `alpha_SA = 1` the inner sphere is not activated
  (appendix 4.7); that puts the crossing at the equilibrium of the initial state, which holds
  when `dG_s0 = V22 - V11` there. That is the definition taken.
- **Eqn (4.8)** is used in the form `x_s = (lambda_s^f + dG_s0) / (lambda_s^f +
  sqrt(lambda_s^i lambda_s^f - (lambda_s^f - lambda_s^i) dG_s0))`, the same number without the
  cancellation near `lambda_s^i = lambda_s^f`.
- **The exponent of eqn (4.14)** lost its brackets in the thesis's typesetting; it is
  `kappa_n = exp(-dG_na(T) / k_B T)` with `dG_na(T)` of eqn (4.13), which gives eqn (4.15) at
  high temperature and the energy-gap law of a single quantum mode at low temperature (both
  checked in the tests).
- **The maximization along `alpha_SA`** (thesis step 6): the grid of the thesis (1.0 to 0.0 in
  steps of 0.1, each crossing optimized from the last), the largest rate taken, refined by the
  parabola through it and its two neighbours in `log k` when it is inside the grid (as the
  thesis does along `r_DA` in chapter 3).

## Zhu-Nakamura in kinetic networks

An electron-transfer step of `run_kinetics` takes `"transfer": "zhu_nakamura"` next to
`nonadiabatic` and `adiabatic`, with `"transfer_wavenumber_cm1"` (the mode along which the
two parabolas `lambda x^2` and `dG0 + lambda (1 - x)^2` cross, e.g. `nu_in_cm1` of `etmain`):
the free energy of its bottleneck is `dE_X - k_B T ln kappa + k_B T ln Q_n` above the first
state, with the thermal Zhu-Nakamura transmission `kappa` of that crossing and the partition
function `Q_n` of the mode in the well (the free energies of the states have the mode, the
parabolas do not), so that the two directions balance (`k_f / k_b = exp(-dG0 / k_B T)`), and
its slopes (for the derivatives) include those of `kappa`. The tests check that it is the
Marcus-Levich rate at weak coupling, and at strong coupling transition state theory over the
lower surface with the mode in the well (`adiabatic` has `k_B T / h` alone, the convention of
the RSC Adv. paper, which is larger by `Q_n`).

## The workflow (`etmain`)

1. **Fragments.** Donor and acceptor atoms are given (`-et_donor`, `-et_acceptor`). With a
   bond between them, the other atoms are the bridge (or `-et_bridge`), and the fragments
   donor-bridge and bridge-acceptor have the bonds they cut capped by hydrogens. Separate
   molecules (no bond between them) are refused for now: chapter 3 of the thesis takes their
   initial and final states as the isolated molecules and the crossings at fixed
   donor-acceptor distances, which is not built yet (an equilibrium of the complex on `V11`
   would let two molecules of like charge fly apart). `-et_charges` are the charges of the two fragments without the
   moving electron; the multiplicities are the lowest ones unless `-et_mult` gives them. The
   input structure is the system with the moving electron.
2. **Equilibria** of the initial and final states on `V11` and `V22` (`-et_final` gives a
   start for the final one). The optimizations (these and the crossings) start from Lindh's
   model Hessian unless `-modelhess` names another or `-fc N` asks for exact Hessians (those
   of the diabatic states with their caps, and at the crossings those of both states, every
   `N` iterations from the first; `-ih` is refused, as one Hessian file cannot serve the
   several optimizations): on the radical anion of 1,4-pentadiene with GFN1-xTB and the
   normal criteria, from the structure of `tests/cli_etmain.rs` in the gas phase, the
   equilibria and the crossing took 57, 58, and 46 iterations from Lindh's model (98 s) and
   68, 63, and 44 from the exact Hessian at the start (100 s), with barriers within 0.1 meV
   (`dE` 0.420 eV, `dG` 0.238 eV). From an earlier start structure, and before the largest
   trust radius of the crossing searches (step 4), they had taken 124, 117, and 114
   iterations from the unit matrix (226 s) against 50, 48, and 42 from Lindh's (89 s).
   The criteria are those of `optmain`; loose ones (`-lcc`) leave the crossing short of a
   stationary point of the seam, which the slopes of the inner sphere and the vibrations
   within the seam assume.
3. **Solvent.** The solvent reorganization energies are given (`-et_lambda_s`, eV); without
   them there is one crossing without solvent activation.
4. **Crossings.** The minimum-energy crossings of the shifted states along `-et_alpha`
   (`1,0,0.1`: from 1 to 0 in steps of 0.1), each from the last; at `alpha_SA = 1` the
   crossing is the initial equilibrium, which needs no search. The searches on the gradient
   projection with RS-I-RFO have a largest trust radius of 0.1 Angstrom unless `-tr` gives
   one, as `optmain -mf opt_mesx` has (next to a seam the difference gradient turns within a
   short distance). On the molecule of step 2 with `-et_lambda_s 0.6 0.6` and
   `-et_alpha 1,0,0.5`, the crossings at 0.5 and 0 took 22 and 46 iterations from Lindh's
   model (97 s for the whole run) and 58 and 51 with exact Hessians every 50 iterations
   (`-fc 50`, 102 s), with barriers within 0.6 meV. Before the derivatives below a
   combination were taken at the shifted energies (see the migration notes, gradient
   projection), the crossing at 0.5 did not converge in 400 iterations from Lindh's model.
5. **At each crossing:** the coupling (`-et_coupling`: `auto`, `mulliken_hush`, `gmh`, `fcd`, `edeel`, or a
   value in eV), the inner sphere along the difference gradient, the tunnelling factor, the
   barriers, and the rate constants; the barrier decomposed into solvent, donor, and
   acceptor.
6. **Result:** `et_scan.csv` (one row per crossing), `et_summary.json` (the largest rate
   constants with their apparent barriers), `eq_initial.xyz`, `eq_final.xyz`,
   `crossings.xyz`, and `mopt.log` with the counts of the evaluations. `--restart DIR` goes on
   from the journal of the evaluations of a run.

## Checks

- The rate constants of the thesis's tables from their tabulated parameters (Tables 4.3 and
  4.S3-4.S7: fourteen rows in three solvents, within 0.2 in `log10 k`, the rounding of the
  printed parameters).
- SAES: the shifted parabolas cross at `x_s`; eqn (4.9) is the limit of eqn (4.8).
- Scher-Holstein: its high-temperature limit, the activationless case, the energy-gap law at
  low temperature, and the exponent at the saddle point of the single-mode golden-rule
  integral computed numerically.
- The inner sphere recovers the frequency, reaction energy, and reorganization energy of two
  displaced harmonic states exactly; Zhu-Nakamura at a crossing is the Marcus rate at weak
  coupling.
- Marcus-Levich-Jortner: Marcus of the solvent alone without an inner sphere, and with the
  reaction energy raised by `lambda_in` for a mode too soft to resolve.
- The open-shell TDA of GFN1-xTB: the singlets of the closed-shell TDA and the triplets come
  out for a closed shell; no electrons are made; the dipole integrals of the basis against
  quadrature.
- A whole run on the radical anion of 1,4-pentadiene (donor, CH2 bridge, acceptor) with the
  built-in GFN1-xTB (`tests/cli_etmain.rs`): the two states at their equilibria have the same
  energy by symmetry, the inner reorganization is four times the barrier of the self-exchange,
  and at `alpha_SA = 1` the crossing is the initial equilibrium.

## Charge hopping: the projection method on HAB11

`kinetics::rate::transfer_integral` has the parts of a hop between two molecules:

- **The transfer integral by the projection method** (E. F. Valeev, V. Coropceanu, D. A. da
  Silva Filho, S. Salman, J.-L. Bredas, J. Am. Chem. Soc. 2006, 128, 9882; B. Baumeier,
  J. Kirkpatrick, D. Andrienko, Phys. Chem. Chem. Phys. 2010, 12, 11103): the frontier
  orbitals of the two neutral molecules, each computed alone at its place in the pair, are
  written in the atomic orbitals of the pair, and with the Fock matrix of the pair from its
  orbitals, `F = S C diag(e) C^T S`, give `J`, the site energies, the overlap `S_ab`, and
  after the symmetric orthogonalization `J_eff = (J - (e_a + e_b) S_ab / 2) / (1 - S_ab^2)`.
  Holes hop between the highest occupied orbitals, electrons between the lowest empty ones.
  Where the frontier orbital is degenerate (within `1e-5` Hartree, as the pi orbitals of
  acetylene), the combinations of the two degenerate sets are taken that couple most (the
  singular vectors of the largest singular value of their matrix of `J_eff`): they do not
  depend on the rotation within each set and are the pair that splits most in the dimer.
- **The four-point reorganization energy** (S. F. Nelsen, S. C. Blackstock, Y. Kim, J. Am.
  Chem. Soc. 1987, 109, 677) and **the Marcus rate of a hop**.

The orbitals must say which atom each atomic orbital is on: the built-in GFN0-xTB and
GFN1-xTB and PySCF (`calculators.pyscf`, from `Mole.aoslice_by_atom`) do; a Python engine adds
`"atom_of_ao"` to its orbitals.

**The workflow** (`etmain -et_method hopping`, `workflow::hopping`): the input is the pair in
the charges of its two molecules without the carrier (`-et_charges`, neutral by default);
`-et_donor` is the molecule the carrier (`-et_carrier hole` or `electron`) starts on and
`-et_acceptor` the other, together every atom of the pair.

```bash
mopt etmain dimer.xyz -os gfn1xtb -et_method hopping -et_carrier hole \
  -et_donor 1,4-8 -et_acceptor 2,3,9-12
```

1. The transfer integral at the structure as given (a pair in a crystal or a snapshot), from
   the orbitals of the pair and of each molecule alone at its place.
2. Each molecule optimized without and with the carrier (Lindh's model Hessian, as for the
   EDEEL workflow), and each charge at both equilibria: the four-point reorganization
   energies of the hop both ways, and its reaction energy (electronic, each molecule alone).
   For two different molecules the two reorganization energies differ; both rates take their
   mean, so that the rates of the two directions keep detailed balance, and both are reported.
3. The Marcus rates both ways with their apparent free energies of activation.

The run writes `mopt.log`, `hop_summary.json`, and the four equilibria
`molecule_{a,b}_{without,with}.xyz`; `run_etmain` returns the same from Python. On the ethylene
dimer of HAB11 at 4.0 Angstrom with GFN1-xTB (`tests/cli_etmain.rs`): `J_eff` 35.2 meV, the
reorganization energy 0.337 eV both ways, no reaction energy, and 1.3e12 1/s both ways, from
28 gradients. With the GFN couplings of the table above the rates are lower bounds.

**HAB11** (the hole-transfer couplings of ten cofacial pi dimers at 3.5, 4.0, 4.5, and
5.0 Angstrom; A. Kubas, F. Hoffmann, A. Heck, H. Oberhofer, M. Elstner, J. Blumberger,
J. Chem. Phys. 2014, 140, 104105, the structures of its supplementary material and the
references as listed in Table I of S. Manna et al., arXiv:1806.01808; benzene is not listed
there and is left out). `tests/hab11_couplings.rs` measures `|J_eff|` with the built-in
calculators against the references:

| calculator | mean ratio at 3.5 / 4.0 / 4.5 / 5.0 Angstrom | MUE (meV) | decay of `ln J` (1/Angstrom) |
|---|---|---|---|
| GFN1-xTB | 0.256 / 0.155 / 0.087 / 0.046 | 171 | 2.66 |
| GFN0-xTB | 0.177 / 0.088 / 0.041 / 0.019 | 185 | 3.40 |
| reference | 1 | | 1.43 |

Every coupling of both is below its reference and falls with the distance. The minimal bases
of the GFN methods give a quarter (GFN1) or a sixth (GFN0) of the coupling at 3.5 Angstrom and
decay about twice as fast, so that at 5 Angstrom GFN1 gives a twentieth. This is the
Hamiltonian, not the projection: for ethylene at 3.5 Angstrom the HOMO pair of the GFN1 dimer
is split by 237 meV, `2 |J_eff|` within 1 meV, where the reference coupling makes 1040 meV.
Hopping rates from GFN couplings are therefore lower bounds by one to three orders of
magnitude (the rate goes with `J^2`); PySCF with a hybrid functional is the route for
couplings to compare with experiment.

**EDEEL's coupling on HAB11** (GFN1-xTB, ethylene, thiophene, and phenol;
`tests/hab11_couplings.rs`) is not a coupling here. Written for the electron that moves from
the neutral molecule to the cation, the complex without it is the dication, and `V12 = V11 -
E(A2+)` holds the Coulomb repulsion of the two cations: 3.4 to 5.0 eV, 10 to 70 times the
references. Written for the hole (the neutral dimer as the complex without it, `V11 = E(A B) -
E(A) + E(A+)`), it is 0.9 to 1.6 eV and grows with the distance: the delocalization error of
GFN1 in the dimer cation, whose hole spread over both molecules lies too low the farther they
are apart.

## Proton-coupled electron transfer: the parts

`kinetics::rate::pcet` (Python: `proton_states`, `pcet_rate`) has the parts of a rate in the
nonadiabatic vibronic picture; the workflow that would make the proton potentials (electron-
proton diabatic states, scans of the proton) is not built, and is postponed until after
v2.0.0 ([below](#postponed-until-after-v200-the-pcet-workflow)).

- **The proton states** on a uniform grid by the discrete variable representation of D. T.
  Colbert and W. H. Miller (J. Chem. Phys. 1992, 96, 1982), for the potential of each
  electron-proton diabatic state, with the masses of the proton and the deuteron for isotope
  effects; the overlaps of the states of two potentials on one grid are dot products.
- **The rate**, the golden rule with a classical solvent and quantized proton states (R. I.
  Cukier, J. Phys. Chem. 1996, 100, 15428; S. Hammes-Schiffer, A. A. Stuchebrukhov, Chem.
  Rev. 2010, 110, 6939, at a fixed proton donor-acceptor distance):
  `k = sum_mu P_mu sum_nu (2 pi / hbar) |V_el S_mu,nu|^2 (4 pi lambda k_B T)^(-1/2)
  exp(-(dG_mu,nu + lambda)^2 / (4 lambda k_B T))`, the Boltzmann populations `P_mu` of the
  reactant's proton states, `dG_mu,nu` the difference of the energies of the proton states in
  their potentials (which carry the electronic energies) plus the free energy the potentials do
  not hold; summed in logarithms, with the share of each pair of states.

Checks (`kinetics::rate::pcet` tests, `tests/test_electron_transfer.py`): the harmonic ladder
of a proton to 1e-9 Hartree, the Franck-Condon factors `e^-S S^n / n!` of displaced
oscillators to 1e-10, the Marcus rate of the electronic coupling when both potentials are the
same, and a deuteron slower than a proton when the proton moves.

## Planned

- Separate molecules by thesis chapter 3: the isolated molecules as the initial and final
  states (with their free energies), the crossings at fixed donor-acceptor distances (the
  centroids, or two given atoms such as the metals), the rate constants of eqn (3.1)-(3.5)
  with the standard state of a bimolecular step, and the largest along the distance by the
  parabola through three of them.
- The solvent reorganization energies from a calculator with nonequilibrium solvation (eqn
  (4.10), (4.11); `kinetics::rate::solvent_activation::solvent_reorganization` is the part).
- Thermal corrections of the reaction energy of a hop, and hopping networks of molecular
  crystals (the pairs of a cell with periodic boundaries and the kinetics of
  `kinetics::rate`).

### Postponed until after v2.0.0: the PCET workflow

Decided on 2026-11-20: v2.0.0 comes first. The parts of
[above](#proton-coupled-electron-transfer-the-parts) (`kinetics::rate::pcet`) stay as they
are, and nothing more of PCET is built before v2.0.0. What the workflow would need, and what is
still to be decided, so that the work can start from here later:

- **The steps.** The electron donor and acceptor as in EDEEL, with the proton donor, the
  proton acceptor, and the moving hydrogen; the proton potentials from scans of that hydrogen
  on `V11` and on `V22` (the four electron-proton diabatic states); the proton states and
  their overlaps from the parts; `V_el`, `lambda` (SAES and the four-point scheme), and `dG0`
  from the parts of `etmain`; the rate and the H/D isotope effect from the parts.
- **The choices still open.**
  - Concerted PCET alone, or the stepwise paths too (proton first or electron first), as
    edges of a kinetic network next to the electron-transfer and proton-transfer steps.
  - Where it is evaluated: the crossing of the heavy atoms (EDEEL with SAES), or the
    equilibrium of the reactant with `lambda` added.
  - The proton coordinate: the line of the proton donor and acceptor, the IRC of the proton
    transfer, or a grid in three dimensions.
  - The fluctuation of the proton donor-acceptor distance (the R mode): the formula of the
    original papers (not checked there yet), or an average over a scan of that distance with
    Boltzmann weights.
  - A check that the proton transfer is vibronically nonadiabatic, which the rate of the
    parts assumes, with a warning when it is not.
- **A system to check it against**, with measured rates and isotope effects in its original
  papers.

## Electron-transfer rate constants (`etmain`)

`etmain STRUCTURE.xyz -et_donor ATOMS -et_acceptor ATOMS` (the command of the Python package,
or `mopt etmain` of the binary) computes the rate constant of an electron transfer from one
structure, the system with the moving electron on the donor: the diabatic states of EDEEL on
the donor-bridge and bridge-acceptor fragments it builds, with hydrogen caps on the bonds
they cut (donor and acceptor in separate molecules are refused for this method: the
procedure of chapter 3 of the thesis is not built), the
equilibria of the two states, their crossings along the solvent coordinate of SAES
(`-et_lambda_s LI LF` in eV, `-et_alpha START,END,STEP`; one crossing in the gas phase), and at
each crossing the coupling (`-et_coupling auto | mulliken_hush | gmh | fcd | edeel | VALUE_EV`), the
barriers (`dE` of the electronic energies with the solvent shift; `dG` from the harmonic free
energies at the initial equilibrium and within the seam at the crossing, `-et_free_energy
on`), and the rate constants of Marcus with Scher-Holstein tunnelling, Zhu-Nakamura, and
Marcus-Levich-Jortner, with the apparent free energy of activation of each. The calculator
and the optimizer are those of `optmain`, its optimizations starting from Lindh's model
Hessian unless `-modelhess` names another or `-fc N` asks for exact Hessians (every `N`
iterations from the first); `-et_temperature` (298.15 K) is the temperature of
the rates and of the free energies (`-press`, `-freq_corr`, and the other settings of `-freq`
apply to the latter). It writes `et_scan.csv`,
`et_summary.json`, `eq_initial.xyz`, `eq_final.xyz`, `crossings.xyz`, and `mopt.log`. The
methods, their sources (chapter 4 of the thesis it follows is not peer reviewed), the
choices where the thesis is open, and the checks are in
[Electron transfer](electron_transfer.md).

```bash
mopt etmain pentadiene_anion.xyz -os gfn1xtb -elec -1 -spin 2 \
    -et_donor 1,2,6-8 -et_acceptor 4,5,11-13 -et_lambda_s 0.6 0.6
```

`-et_method hopping` computes the hop of a charge (`-et_carrier hole` or `electron`) between two
molecules instead: the input is the pair in the charges of its molecules without the carrier,
`-et_donor` the molecule the carrier starts on and `-et_acceptor` the other. It takes the
transfer integral of the projection method at the structure as given, the four-point
reorganization energy from the equilibria of both molecules in both charges, and the Marcus
rates both ways, and writes `hop_summary.json`, the four equilibria, and `mopt.log`. The
calculator must say which atom each atomic orbital is on (the built-in GFN0-xTB and GFN1-xTB,
PySCF); the GFN couplings are a fraction of those of the HAB11 references (Electron transfer,
the HAB11 section).

```bash
mopt etmain dimer.xyz -os gfn1xtb -et_method hopping -et_carrier hole \
    -et_donor 1,4-8 -et_acceptor 2,3,9-12
```
