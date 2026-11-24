# Kinetics of reaction networks (`run_kinetics`, `run_rcmc`)

`run_kinetics` computes the kinetics of a reaction network: rate constants for every step from
the stationary points of the network, the master equation of the populations of the minima, and
its solution at the times you ask for, with the superstates of the rate constant matrix
contraction (RCMC) and diagnostics that say how far the answer can be trusted. RCMC is one of
the solvers, not the name of the whole: the default solver contracts the fast states and then
propagates the superstates exactly. The design, the assumptions, and the theory behind each
choice are in [kinetics_beyond_tst.md](kinetics_beyond_tst.md).

```console
run_kinetics mapper_output                       # the run directory of run_mapper
run_kinetics reaction_network.json -T 350 --time-grid 1e-12 1e-3 10 --validate
run_kinetics network.json --solver rcmc --time-rule eigen -t 1e-6
run_kinetics network.json --rate-theory tst --annotations freqs.json --corrections eckart
run_kinetics network.json --rate-theory rrkm --model egme --collision-model exponential_down \
    --energy-transfer 200 --bath 3.47:114:39.95 --lennard-jones 4.5:400 --pressure 1e4
run_kinetics network.json --refine 3 --uncertainty 1.5 --times 1e-6 1e-3
run_rcmc network.json -T 300 -t 1.0 --start-node 0 --output-dir rcmc_out   # MultiOptPy's command
```

`run_rcmc` is MultiOptPy's command with its options (`json_file`, `-T`, `-t`, `--start-node`,
`--output-dir`, `--energy-type`, `--log-level`). It runs the same code with the settings that
reproduce MultiOptPy's numbers (Eyring rates with MultiOptPy's clamp, the solver
`legacy_rcmc`), prints MultiOptPy's table, and writes `rcmc_K_contracted.csv` as MultiOptPy does
(with the other files below) when `--output-dir` is given. The differences of MultiOptPy's
algorithm from the published method are listed in
[migration.md](migration_workflows.md#kinetics-run_rcmc-run_kinetics-and-the-rcmc-priority).

## What it does

1. **The network** is read: the `reaction_network.json` of `run_mapper`, its database
   `reaction_network.sqlite` (`-map_opt store=sqlite`; read as far as it goes, with what was
   left out in the notes), or its run directory (the JSON file, or the database when the
   directory has only that), or MultiOptPy's network file (`nodes[].node_id`,
   `energy_hartree`, `free_energy_hartree`; `edges[].node_id_1`, `node_id_2`,
   `ts_energy_hartree`, `ts_free_energy_hartree`). The states are the minima and the
   dissociation channels (of level 1, `--level`), the bottlenecks the transition-state edges
   between two different states, the crossing edges with both sides known, and the
   barrierless dissociations. What is left out (a
   degenerate rearrangement that joins a minimum with itself, a self-loop, a state or edge
   without the energy the rate theory needs, a transition state whose IRC has an end that is
   not known, `open_ts_edges`) is listed on the screen, in `mopt.log`, and in
   `diagnostics.json`; nothing is guessed, with one exception said there too. A transition
   state whose IRC runs into another transition state (a TS–TS link of the mapper, where a
   reaction bifurcates past a valley-ridge inflection) becomes one step from its other end to
   every state beyond the link, each with an equal share `w = 1/n` of its flux. The share is a
   free energy `-RT ln w` on the bottleneck, so it scales both directions and detailed balance
   holds, and the total rate out of the start is that of the transition state. The split
   itself is assumed: the network does not tell it, only trajectories from the transition
   state would. An annotation of such a transition state applies to all of its steps.
2. **Rate constants** in flux form (`kinetics_beyond_tst.md`, section 3.2): every bottleneck is
   one symmetric equilibrium flux `ln F = ln(k_B T / h) - (G_ts - G_ref) / RT + sum ln f_c`,
   every state a weight `ln pi = -(G_j - G_ref) / RT`, and `K[b][a] = F / pi[a]`. Detailed balance
   therefore holds by construction; parallel bottlenecks between the same pair add (log-sum-exp).
3. **The model** `canonical`: thermalized minima, first-order steps, `dp/dt = K p`, with
   bimolecular steps pseudo-first order at the concentrations of their species; or
   `mass_action`: the states and the species as concentrations, non-linear (see
   [Bimolecular steps](#bimolecular-steps-concentrations-and-the-standard-state)); or `auto`,
   which judges the order of every step and takes one of the two (below); or one of
   the models of [Beyond the canonical model](#beyond-the-canonical-model): `egme` (energy
   grains and collisions), `two_grain` (hot intermediates), `semi_markov` (residence times
   that are not exponential).
4. **The solver** gives the populations at the output times (`-t`, `--times`, `--time-grid`)
   from the start population (`--start-node`, `--initial ID:SHARE ...`; by default the node of
   the first input of a mapper run, node 0 of a MultiOptPy network).
5. **Checks and reports**: conservation, the detailed-balance residual of the superstate
   generator, the separation ratio of the cut, `--validate`, and on request degrees of rate
   control, energy errors, the relaxation spectrum, species, and the exploration ranking.

## Rate theories and corrections

| Name | What | Needs |
|---|---|---|
| `eyring` (default) | `k = (k_B T / h) exp(-(G_ts - G_min) / RT)` with one energy per stationary point: electronic (`--energy-type electronic`, MultiOptPy's default) or the Gibbs energies of the file (`free`) | energies |
| `tst` | free energies from partition functions (`vib::thermochemistry`, the thermochemistry of `freqmain`): ideal-gas translation at `--pressure` or `--standard-concentration`, rigid rotors with the rotational symmetry number of each structure (`--symmetry auto`: from the point group, tolerance 1e-2 Bohr as `freqmain`), harmonic vibrations with Grimme's quasi-RRHO (`--qrrho grimme`, cutoff 100 cm^-1, the defaults of `freqmain`), the multiplicity | frequencies and structures of the states and saddle points (networks of mopt-rs with frequencies, or `--annotations`) |
| `wigner` | `kappa = 1 + u^2 / 24`, `u = h nu_im / k_B T` | imaginary frequency |
| `bell` | Bell's parabolic barrier: the infinite parabola `kappa = (u/2) / sin(u/2)` (`u < 2 pi`), or the parabola truncated at the barrier, the thermal average of `P(E) = 1 / (1 + exp(-2 pi E / h nu))` cut at `-V` (`--bell-truncation auto`: the infinite form where the truncation changes `kappa` by less than 1e-8, the truncated one near and past `u = 2 pi` and for low barriers; `on`, `off`). The cut is the barrier from the higher minimum (`--bell-barrier higher`, the same factor from both sides), or the forward or the reverse barrier | imaginary frequency; the barriers for the truncation |
| `eckart` | thermal average of the transmission of the unsymmetrical Eckart barrier fitted to both barriers and the imaginary frequency, referred to the top of the barrier (the same factor from both sides) | imaginary frequency, both barriers > 0 |
| `skodje_truhlar` | the closed form of Skodje and Truhlar with the barrier from the higher minimum | imaginary frequency |
| `kappa` | a given transmission coefficient (`--kappa`, or per step in the annotations) | the value |
| `landau_zener` | crossing points: thermal average of the double-passage Landau-Zener probability, `1 - P^2` for a peaked crossing and `2 P (1 - P)` for a sloped one (see below); without a coupling a crossing keeps the probability 1 (the adiabatic limit) | coupling, gradient difference, reduced mass; the product of the slopes for the sloped form |
| `zhu_nakamura` | crossing points: for a sloped one the thermal average of `2 p (1 - p)` with the Zhu-Nakamura probability `p` of one passage, over the energies above the crossing; for a peaked one the thermal average of the overall transmission in the three zones of the nonadiabatic-tunnelling type, from the higher minimum up, tunnelling below the crossing included (see below) | coupling, gradient difference, product of the slopes, reduced mass (or `diabatic_crossings`); the barriers for a peaked crossing |
| `cvt` (also `variational`) | canonical variational TST of the steps with a reaction path: the flux times `exp(-(max_s G_GT(s, T) - G_GT(s_saddle, T)) / k_B T)`; steps without a path keep TST (`cvt_path=0` in `rates.csv`) | the path of `freqmain -path` (see below) |
| `rrkm` (theory) | the Boltzmann average of the RRKM rate constants `k(E) = W_ts(E - E0) / (h rho(E))` from directly counted states (Beyer-Swinehart in cells of `--rrkm-cell`, 1 cm^-1, convolved with the classical rotors; translations and spin as `tst`): every state gets `-kT ln sum N_j exp(-E_j / kT)`, every saddle point `-kT ln((delta / kT) sum W_j exp(-E_j / kT))`, so the canonical rates are those of the `k(E)` that `egme` uses (its high-pressure limit) | frequencies and structures, as `tst` |
| `small_curvature`, `zero_curvature` | tunnelling through the vibrationally adiabatic ground-state barrier `V_a^G(s)` of a path, with the effective mass of small-curvature tunnelling (Liu et al., J. Am. Chem. Soc. 1993, 115, 2408) or without it, at the temperature of the run (`vib::transmission`; threshold at the top of `V_a^G`, the same factor from both sides; with `cvt` the CVT/SCT coefficient) | `path_profile.csv` of `freqmain -path` (`"sct_profile_file"`: `arc_length_sqrt_amu_bohr`, `adiabatic_energy_hartree`, `sct_mass_ratio`) |
| `kramers` | Kramers' spatial-diffusion transmission coefficient `kappa = sqrt(1 + (gamma / 2 omega_b)^2) - gamma / (2 omega_b)` (computed as `exp(-asinh(gamma / 2 omega_b))`; Physica 1940, 7, 284; Hanggi, Talkner, Borkovec, Rev. Mod. Phys. 1990, 62, 251) | imaginary frequency; the friction `gamma` (1/s) of the mass-weighted reaction coordinate (`--friction`, or `"friction_s-1"` per step) |
| `grote_hynes` | `kappa = lambda_r / omega_b` with the root of `lambda^2 + lambda zeta_hat(lambda) = omega_b^2` (Grote, Hynes, J. Chem. Phys. 1980, 73, 2715) for an exponential kernel (`"friction_s-1"` and `"friction_memory_s"`; Kramers without the memory) or a tabulated one (`"friction_kernel_file"`, columns `time_s,zeta_s-2`, transformed by the trapezoidal rule; e.g. from `workflow::kinetics_dynamics::friction_kernel`) | imaginary frequency, the kernel |
| `reactive_flux` | the transmission coefficient of the dividing surface at the saddle point measured by trajectories (Bennett-Chandler, see [Trajectories from a saddle point](#trajectories-from-a-saddle-point)) | `"reactive_flux_kappa"` (and `"reactive_flux_error"`) |
| `collins_kimball` | the diffusion limit of a bimolecular step `a + S -> b` in a solution: the flux times `k_D / (k_D + k_act)` (both directions; `1 / F = 1 / F_act + 1 / F_diff`), `k_act` (L/mol/s) from the theory with the other corrections, Smoluchowski's `k_D = 4 pi R (D_a + D_S) N_A` with the Stokes-Einstein coefficients of the van der Waals radii (`kinetics::rate::encounter`); `k_act_over_k_d` in `rates.csv` | `--viscosity`, the structures of `a` and of the species |
| `weak_coupling` | crossings: the Airy form of the weak-coupling probability `P(E) = 4 pi^2 H12^2 (2 mu / (F dF))^(2/3) Ai^2(-(E - E_X) (2 mu dF^2 / F^4)^(1/3))` (Delos's result as quoted by Harvey, Phys. Chem. Chem. Phys. 2007, 9, 331, and Lykhin et al., Int. J. Quantum Chem. 2016, 116, 750; `F = sqrt(|F1 F2|)`, `dF = |F1 - F2|`), averaged from the higher minimum (at most the crossing) upwards: tunnelling below the crossing included, and from all energies the closed form `(C / 2) sqrt(p / pi) exp(p^3 / 12)`, the thermal weak limit of Landau-Zener times `exp(p^3 / 12)` | coupling, gradient difference, product of the slopes, reduced mass |

**Bell and Skodje-Truhlar.** The truncated parabola is the exact integral that Skodje and
Truhlar's closed form approximates: below `u = 2 pi` its series
`kappa = pi r / sin(pi r) - r sum_n (-1)^(n-1) e^{-(n - r) a} / (n - r)` (`r = u / 2 pi`,
`a = 2 pi V / h nu`) has their formula as its first two terms (checked: the difference is the
next term, within twice the term after it, at `r = 0.8`, `a = 5`), and above `2 pi` their
formula is its leading behaviour (within 1e-5 in `ln kappa` at `r = 1.5`, `a = 30` and at
`r = 3`, `a = 20`). The truncated form is
finite and smooth through `u = 2 pi`, where `kappa = a + ln(1 + e^-a)` exactly (checked to
1e-12). For small `u` both forms are `1 + u^2/24 + 7 u^4/5760 + ...`, Wigner to `O(u^4)`. The
sources of Bell (Trans. Faraday Soc. 1959, 55, 1; The Tunnel Effect in Chemistry, 1980) were
not checked here; the formulas are derived in `kinetics::rate::tunneling` and checked against
the parabolic limit of the Eckart barrier and these identities.

The barriers of the tunnelling corrections are the electronic ones, with the zero-point
energies where the frequencies of the saddle point and both minima are known (the diagnostic
`eckart_zero_point` says which). Every correction reports `u` and `T / T_c` in `rates.csv`; at
`u` above about 3 Wigner is outside its range. The corrections are symmetric in the two
directions, so they keep detailed balance.

**The clamp** (`--clamp`): a saddle point below one of its minima (possible with free energies
or mixed levels) gets `max(E_ts, E_a, E_b)` for both directions (`consistent`, the default),
which keeps `k_ab / k_ba = exp(-(E_b - E_a) / RT)`; `legacy` is MultiOptPy's clamp of each
direction by its own minimum (no detailed balance); `none` leaves it.

**Crossings: peaked and sloped.** At a peaked crossing the slopes of the two states along
the hopping coordinate have opposite signs and the two minima lie on the two sides; the step
is made unless both passages (in, and back out) keep the diabatic state: `1 - P_LZ^2` with
`P_LZ` the probability that a passage keeps it. At a sloped crossing the slopes have the
same sign and both minima lie on one side; the step needs exactly one of the two passages to
change the diabatic state: `2 P (1 - P)` with the phase of the two passages averaged, at most
1/2, and 0 for a strong coupling. Which one a crossing is comes from the product of the two
slopes (`slope_product`, Hartree^2/Bohr^2); without it a crossing is taken as peaked, as
before.

**Zhu-Nakamura.** With the coupling `V`, the slopes `F1`, `F2`, the reduced mass `mu`, and
the energy `E` above the crossing energy `E_X` (atomic units):
`a^2 = sqrt(F1 F2) |F1 - F2| / (2 mu (2 V)^3)`, `b^2 = (E - E_X) |F1 - F2| / (2 sqrt(F1 F2) V)`,
and for `E >= E_X` the probability that one passage changes the adiabatic state is
`p = exp(-(pi / 4a) (2 / (b^2 + sqrt(b^4 + 0.4 a^2 + 0.7)))^(1/2))`, which is Landau-Zener's
`exp(-pi / 4ab)` far above the crossing and stays finite at it, where Landau-Zener diverges.
`zhu_nakamura` averages `2 p (1 - p)` over `E >= E_X`. The same `a` and `b` follow from the
adiabatic energies alone (the gap at its smallest, `2V`; the points `t1`, `t2` where the lower
and the upper energies reach `E_X`; `d^2` from the gaps there), which is how
`freqmain -diabatize` makes the `zhu_nakamura` rows of `diabatic_crossings.csv`: for a
crossing of two states of the same spin no coupling has to be computed. For a crossing of two
spin states the coupling is the spin-orbit coupling, which the user gives
(`coupling_hartree`). The formulas are those quoted by Lykhin and Varganov (Phys. Chem. Chem.
Phys. 2020, 22, 5500, eqs. 12-17) and Lehto (arXiv:1304.3331, eq. 29) from Zhu and Nakamura;
the two forms of `a` and `b` were checked against each other for straight states. The same
probabilities of one passage decide the hops of surface hopping in `mdmain`
(`-md_opt hopping_states=`, see `molecular_dynamics.md`).

**Zhu-Nakamura at a peaked crossing** (the nonadiabatic-tunnelling type, slopes `F1 > 0 > F2`,
`F = sqrt(|F1 F2|)`): `a^2 = F (F1 - F2) / (16 mu V^3)` and `b^2 = (E - E_X) (F1 - F2) / (2 F V)`,
so that `b^2 = -1` at the top `E_t` of the lower adiabatic energy and `1` at the bottom `E_b`
of the upper one. The probability of a step is the overall transmission `P12` from one side to
the other, in three zones:

- `E >= E_b`: `p = exp(-(pi / 4a) (2 / (b^2 + sqrt(b^4 - 0.72 + 0.62 a^1.43)))^(1/2))` for one
  passage and `P12 = 4 cos^2 phi / (4 cos^2 phi + p^2 / (1 - p))`, used with the phase
  averaged, `2 (1 - p) / (2 - p)` (the phase of a straight model would put the resonances of a
  well that a molecule does not have);
- `E_t <= E <= E_b`: `P12 = W^2 / (1 + W^2)` with `W` an Airy-type integral with three
  empirical factors, taken along a contour in the complex plane;
- `E <= E_t` (tunnelling under the barrier of the lower energy):
  `P12 = B e^(-2 delta) / ((1 + a / (2 (1 + a)) B e^(-2 delta))^2 + B e^(-2 delta))` with
  `B = B(sigma_c / pi)`, the closed form of `sigma`, and `delta` the action under the barrier
  of the straight two-state model (it depends on how unequal the slopes are).

`zhu_nakamura` averages `P12` from the higher of the two minima (the lower barrier) upwards,
`kappa = (1 / k_B T) int P12(E) exp(-(E - E_X) / k_B T) dE`, so that the factor is the same
from both sides; it can exceed 1 where tunnelling dominates, and `rates.csv` gives the share
from below the crossing (`below_crossing`). The limits are tested: far above the crossing `p`
is Landau-Zener's one passage and the averaged `P12` its double passage `1 - P^2` to
`O(1 - p)`; for a weak coupling the same holds at energies well above the crossing on the
energy scale of the straight model (the argument `b^2 / a^(2/3)` of its Airy function much
larger than 1). Within a few units of it Landau-Zener does not hold, and the empirical factor
of `p` makes its exponent a few per cent smaller (measured: 2 % to 5 % at
`b^2 / a^(2/3) = 2.3` for couplings from 1e-4 to 1e-7 Hartree). Thermal averages agree with
Landau-Zener to 1 % when `k_B T` is far above that scale, and differ by up to some 10 % at
room temperature, where tunnelling and the Airy region count. The formulas are those of
Ishida, Nanbu, Nakamura (Int. Rev. Phys. Chem. 2017, 36, 229) and of the supporting
information of Shchepanovska, Shannon, Curchod, Glowacki (J. Phys. Chem. A 2021, 125, 3473),
which agree; the phase average, the `delta` of the straight model and the contour of `W` are
derived in the code and checked in its tests. `freqmain -diabatize` writes the model of a
peaked crossing from the adiabatic energies alone (the top `E_t` at `R_t`, the bottom `E_b` at
`R_b`, `gamma = (E_b - E_t) / (E2 - E1)` midway:
`a^2 = (1 - gamma^2) / (mu (R_b - R_t)^2 (E_b - E_t))`, or from the curvatures when
`R_b = R_t`). Not built: the energies below a sloped crossing (its tunnelling), for which the
same text gives the formulas.

**Crossings from `freqmain -diabatize`.** `diabatic_crossings.csv` of a `-diabatize` run lists
its crossings (`crossing`: two diabatic states cross; `gap`: the straight model fitted to the
smallest gap; `zhu_nakamura`: the model of the adiabatic energies above, sloped or
peaked), with the coupling,
the two slopes along the mass-weighted path, their difference and product, and the reduced
mass 1 amu. `"diabatic_crossings": "run/diabatic_crossings.csv"` in the annotations of an edge
takes the coupling, the slopes and the reduced mass from it (`"diabatic_crossing": N`, from 1,
chooses the row when there are several); a coupling of the edge itself (a spin-orbit coupling)
is kept, and slopes or a reduced mass of the edge together with the file are refused.

Refused by name, with the reason: `marcus` as a correction of every step (an electron transfer
is a property of its edge, below) and `symmetry` (part of `tst` and `rrkm`).

**The weak-coupling factor** is checked against the closed form at high temperature (1e-8),
against Landau-Zener's thermal weak limit (1e-3, the order of the tunnelling factor), and by
direct quadrature against the closed Laplace transform of `Ai^2` where `p >= 1` (1e-7);
`Ai` itself against the values of Cephes (1e-10 relative, 5e-9 next to `x = -7`, where the
series and the asymptotic expansion meet; `math::special`). **Kramers** has its limits 1 and
`omega_b / gamma` (1e-12); **Grote-Hynes** is Kramers without memory (1e-12), tends to 1 for a
frozen solvent, and agrees for a tabulated exponential kernel (1e-4). **Curvature tunnelling**
of a parabolic adiabatic barrier is Bell's `(u / 2) / sin(u / 2)` (5e-3, the semiclassical
action on the grid) and a smaller effective mass tunnels more.

### The rate theory of each edge

Every step keeps a forward and a backward rate constant from one bottleneck free energy (and
the free energies of its two sides), so steps of different kinds mix in one network and one
solver, and detailed balance holds for each (checked on a network with a saddle point under
Bell's correction, a saddle point with a variational path, a non-adiabatic and an adiabatic
electron transfer, and a crossing point under Landau-Zener: residual below 1e-12, `propagate`
equal to the matrix exponential to 1e-8, the derivatives equal to central differences).
`rates.csv` names the theory of every step (`step_rate`) and the factors with their
diagnostics.

| Edge | Rate | Data | Where it comes from |
|---|---|---|---|
| saddle point | `eyring` or `tst`, with `wigner`, `bell`, `eckart`, `skodje_truhlar`, `kappa` | energies (`eyring`); frequencies and structures (`tst`); the imaginary frequency and the barriers (tunnelling) | the network of `run_mapper` (energies, structures, imaginary frequencies), the annotations (frequencies, free energies) |
| saddle point with a path | the above times `cvt` | the arc length, the energy, and the transverse frequencies along the path, or the generalized free-energy profile at the temperature of the run | `freqmain -path` (or `ircmain -path_analysis`): `free_energy_profile.csv` (`"cvt_profile_file"`, its temperature from `variational_tst.txt` next to it or `"cvt_temperature"`), or `"cvt_path": {"arc_length", "energy_hartree", "frequencies_cm1"}` (any temperature) |
| electron transfer | `marcus_nonadiabatic`, `marcus_adiabatic`, or `zhu_nakamura_transfer` (`"transfer"`: `nonadiabatic`, `adiabatic`, `zhu_nakamura`) | the reorganization energy `lambda`, the coupling `H_ab`; `dG0` is the difference of the free energies of the two states; for `zhu_nakamura` the wavenumber of the mode along which the parabolas cross (`"transfer_wavenumber_cm1"`, e.g. `nu_in_cm1` of `etmain`), whose crossing gives the thermal Zhu-Nakamura transmission with tunnelling (the Marcus rate at weak coupling, the adiabatic one at strong coupling) | `"reorganization_hartree"` or `"reorganization_kcal_mol"` (EDEEL: eqn (10) of `kinetics::rate::electron_transfer`, `reorganization_energy`), `"coupling_hartree"` or `"coupling_cm1"` (the diabatic coupling of `freqmain -diabatize`, `analysis::diabatic`, or the EDEEL coupling); `"delta_g0_hartree"` is checked against the states |
| crossing point | `eyring`/`tst` with `landau_zener` or `zhu_nakamura` (probability 1 without a coupling) | coupling, gradient difference, reduced mass; the product of the slopes for sloped crossings | the crossing edges of the mapper, the annotations, `diabatic_crossings.csv` of `freqmain -diabatize` |
| barrierless dissociation | `capture` (the gas) or `collins_kimball` (`--viscosity`): the association of the two fragments, the dissociation by detailed balance | the structures (radii, masses) of the two fragments | the dissociation edges of the mapper (a complex of two molecules and its channel) |

**Dissociation channels.** A dissociation channel of the mapper (`DC`) is a state whose free
energy is the sum of those of its fragments (each optimized on its own, with its own
frequencies, structure and multiplicity for `tst`); every fragment beyond the first is one more
free molecule, converted from the standard state of the free energies to `c°` (1 mol/L) as a
species is (none for electronic energies). Transition states whose IRC ends at a channel are
ordinary steps. The barrierless step between a complex of two molecules and its channel takes
its rate from the association (`kinetics::rate::encounter`): from the channel
`k = k_a c°`, from the complex by detailed balance, written as the free energy of a bottleneck
that follows the channel. In the gas `k_a` is the capture of hard spheres on contact,
`pi R^2 sqrt(8 k_B T / (pi mu))`. With `--viscosity` (Pa s) it is Collins and Kimball's
encounter by diffusion and capture in series, `1 / k_a = 1 / k_D + 1 / k_c`, with
Smoluchowski's `k_D = 4 pi R (D_A + D_B)` and the Stokes-Einstein coefficients (equal radii:
`8RT / 3 eta`, 7.4e9 L/mol/s in water at 25 C, checked). The radius of a fragment is that of the
sphere of its van der Waals volume; the forms are those of the textbooks (the texts of
Smoluchowski 1917 and Collins and Kimball 1949 were not read here). A channel of three or more
fragments has no barrierless step.

**Electron transfer.** With the parabolas `lambda x^2` and `dG0 + lambda (1 - x)^2` of the two
diabatic states:

- `nonadiabatic` (Marcus-Levich, eqn (5) of the EDEEL scheme of Mutsuji, Saita, Maeda, RSC Adv.
  2023, 13, 32097): `k = (2 |H_ab|^2 / h) (pi^3 / (lambda k_B T))^(1/2) exp(-(lambda + dG0)^2 /
  (4 lambda k_B T))`, written as the bottleneck `G_a + (lambda + dG0)^2 / (4 lambda)` with that
  prefactor, so `k_f / k_b = exp(-dG0 / k_B T)`. Activationless at `dG0 = -lambda` (the rate is
  the prefactor), symmetric about it in the inverted region (`dG0 = -2 lambda` is as fast as
  `dG0 = 0`, checked to 1e-10);
- `adiabatic`: TST (`k_B T / h`) on the lower surface of the two-state model (eqn (9)), its
  highest point between the minima found numerically: `lambda / 4 - H_ab` for `dG0 = 0` and
  `H_ab < lambda / 2` (checked to 1e-12). Refused in the inverted region `|dG0| > lambda`, where
  one state is not a minimum of the lower surface.

The functions are those of the EDEEL analysis (formerly `analysis::electron_transfer`, now
`kinetics::rate::electron_transfer`, re-exported at the old path). The derivatives move the
bottleneck with the states through the slopes of the parabola (`1 - x_c`, `x_c`) or of the lower
surface (envelope).

**Variational TST.** The generalized free energy is the energy plus the vibrational free energy
of the transverse modes (the thermochemistry of `freqmain`, `--qrrho`), and the maximum is that
of the parabola through the highest point and its neighbours, as `vib::vtst`: on the model of
`vib::vtst` the factor equals its `recrossing` to 1e-12 at 200, 298.15, and 600 K, from the
frequencies and from the profile. A profile given at one temperature is used within 1 K of it
(and refused farther away); its free energies are then held as they are, so `dp/dT` of such a
step lacks the change of the profile with `T`.

## Bimolecular steps, concentrations, and the standard state

A step `a + S <-> b` takes up a species `S` that is not a state of the network (a substrate, a
reagent, a product released). The network files do not say which steps these are, so they are
given: the species under `"species"` of `--annotations` (electronic energy, free energy,
frequencies, structure as XYZ text, its own association treatment) or `--species
NAME:E_HARTREE[:G_HARTREE]`, and the edge by `"partner": "S", "partner_free_state": "EQ1"` (the
state of the edge where `S` is free) or `--partner TS4:S:EQ1`. The energy of the state on the
free side is that of the state alone; `S` is added to it.

**Pseudo-first order** (the model `canonical`). `--concentration S:VALUE[UNIT]` holds `S` at a
concentration (mol/L by default; `M`, `mM`, `uM`, `nM`, `pM`, `mol/m3`, or a partial pressure of
an ideal gas in `bar`, `atm`, `Pa`, as `c = p / RT`). Its chemical potential
`mu_S = G°_S + RT ln([S] / c°)` enters the free side of its steps, so that

```text
k(a -> b) = k_2 [S],   k_2 = (k_B T / h) exp(-(G_ts - G_a - G°_S) / RT) / c°   (L/mol/s)
k(b -> a) = (k_B T / h) exp(-(G_ts - G_b) / RT)                               (1/s)
```

Both directions come from the same free energies, so every step balances its reverse at the
concentration given. A state with `n` free molecules of `S` (relative to the other states of
its component, found by a walk over the steps) has the free energy `G + n mu_S`, and the rates
are in flux form. A network in which a cycle of steps converts species, such as the catalytic
cycle `E + S -> ES -> EP -> E + P`, has no such potential: its rate constants are directed (each
step still balances its own reverse, and the product of the rate constants around the cycle
over those against it is `exp((mu_S - mu_P) / RT)`, checked to 1e-10), the generator claims no
detailed balance, and the solvers and the derivatives handle it as any directed generator; the
screen and `diagnostics.json` (`"potential": false`) say so. A concentration must be positive;
a species that is only released belongs to `mass_action`. The consistent clamp acts on the
free energies at the concentration given.

**The standard state.** The free energies of the rate theories are at the standard state of
`vib::thermochemistry`, the one of `freqmain`: the ideal gas at `-press` (`--pressure`, 101325 Pa)
or, with `-conc` (`--standard-concentration`), a solution at that concentration. The standard
concentration of the bimolecular steps is `c° = -conc` when given and 1 mol/L otherwise, and
every free energy of the gas is converted to it by

```text
G°(c°) = G°(p) + RT ln(c° R T / p)     per molecule
```

which is `RT ln 24.47 = 1.894` kcal/mol at 298.15 K and 1 atm (`RT ln 24.79 = 1.900` kcal/mol at
1 bar). Only the number of molecules on each side matters, so the conversion changes the
bimolecular steps alone: `k_2` from free energies of the gas is 24.47 times larger at 1 mol/L
than the same numbers without it (checked to 1e-10), and `k [S] / c°` uses the same `c°`.
With `-conc` the free energies are there already and nothing is converted. The free energies of
a network file (`--energy-type free`) are taken at the standard state of the conditions in the
same way. Electronic energies have no standard state and are not converted. The screen,
`diagnostics.json` (`standard_concentration_mol_L`, `standard_state_shift_kcal_mol_per_molecule`),
and the Python function `kinetics.standard_state_shift` report the conversion.

**Association in solution** (`--association`, `none` by default; per species `"association"` of
the annotations). Optional empirical treatments of the entropy that an association loses:

| Treatment | What | Source |
|---|---|---|
| `wertz:F` | every structure (state, saddle point, species) loses the fraction `F` of its translational and rotational entropy at `c°`, `G += F T (S_trans + S_rot)`, so that the loss on association is scaled by `1 - F`; needs the structure of every state, saddle point, and species of the model | Wertz-type scaling (D. H. Wertz, J. Am. Chem. Soc. 1980, 102, 5316). The paper was not checked here, so no value of `F` is built in |
| `free_volume:M:V`, `free_volume:SOLVENT` | the translational entropy of a species in the free volume of a solvent of molarity `M` (mol/L) and molecular volume `V` (Angstrom^3): `v_free = 8 ((1/(M N_A))^(1/3) - V^(1/3))^3` per solvent molecule, the free fraction `phi = v_free M N_A`, per free molecule `-RT ln phi` | as GoodVibes computes it (`get_free_space`, which GoodVibes marks as deprecated and experimental; read, not run), with its table (`h2o` 55.6 mol/L and 27.944 A^3: `phi` = 8.4e-5, 5.6 kcal/mol per molecule at 298 K; `toluene`, `dmf`, `acoh`, `chloroform`). The source GoodVibes cites (Mammen, Shakhnovich, Whitesides) was not checked here |
| `effective:C` | an effective concentration `C` (mol/L) in place of `c°` in the conversion, per free molecule `RT ln(C / c°)` | a user's value |
| `factor:F` | the association constant and `k_2` multiplied by `F` (the same as `effective` with `C = F c°`) | a user's value |

Every treatment changes free energies, never one direction alone, so detailed balance holds
(checked for each); `rates.csv` and `diagnostics.json` record the treatment of every
bimolecular edge.

**The order of every step, judged** (`--model auto`, `kinetics::order`). With the states
together at `--total-concentration`, a species `S` held by `n_S(g)` free molecules in the state
`g` (the composition of the rate assembly) changes by at most `c_tot (max_g n_S - min_g n_S)`,
whatever the kinetics does; a bimolecular step whose partner stays within `--order-tolerance`
(1e-2, "a few per cent", section 2.6 of the design) of its concentration is pseudo-first order,
one whose partner may change more, starts at zero (a product released), or is turned over by a
cycle of steps is of second order. When every step is of first or pseudo-first order the run
takes the linear model `canonical` (RCMC and its contraction apply), otherwise `mass_action`
with the solver `contract` (unless `--solver` gives `rosenbrock` or `windows`); the judgment of
every bimolecular step is on the screen and in the notes of `diagnostics.json`, whose `model`
is the model that ran. Without `--total-concentration` the orders cannot be judged and the
steps are taken as pseudo-first order. Checked: `A + B <-> C -> D` with the states at 1 mM is
`canonical` with `B` at 1 M and `mass_action` with `B` at 1 mM (`tests/cli_kinetics_models.rs`).

**Mass action** (`--model mass_action`, solved by `rosenbrock`, contracted by `contract`, or by the pseudo-first-order `windows`). The states of the network and
the species of its bimolecular steps are concentrations:

```text
a + S -> b :  rate = k_2 [a] [S]       b -> a + S :  rate = k(b -> a) [b]
a -> b     :  rate = k(a -> b) [a]
```

with the rate constants at `c°` (every species at the standard state). `--total-concentration`
(mol/L) is the concentration of the states together at the start, shared by the initial
population; the species start at their `--concentration` (0 for one not given). The equations
are integrated by ROS2 (Verwer, Spee, Blom, Hundsdorfer, SIAM J. Sci. Comput. 1999, 20, 1456;
two stages, second order, L-stable, the embedded first-order solution for the step size). The
totals of the states of every component and of every species, free and bound, are conserved:
one row of every stage is replaced by its invariant, so they hold to rounding (1e-15 of the
largest concentration) also for steps far beyond the fastest time scale. What rounding still
limits is the slow evolution next to a fast equilibrium (the net rate is a difference of two
large fluxes, a relative error of about 1e-16 times the ratio of the rates), which the linear
solvers do not have. Outputs: `concentrations.csv` (mol/L, the states and then the species),
`yields.csv` (initial and final concentration and the yield of every entity: of a state its
share of the states together, of a species what is left of it), `concentrations.svg`. The
derivatives, `--validate`, `--uncertainty`, `--priority`, and `--spectrum` are for the linear
model and are refused with `mass_action`.

**Checks** (`tests/cli_kinetics.rs`, `kinetics::species`, `kinetics::mass_action`): `A + B <-> C`
under mass action against the analytic solution
`x(t) = r_1 (1 - e^(-a t)) / (1 - (r_1 / r_2) e^(-a t))` (`a = k_2 (r_2 - r_1)`, `r_1,2` the
roots of `k_2 (A0 - x)(B0 - x) - k_r x`) to 2e-10 of `[A]0`, with `A + C` and `B + C` conserved
to 1e-15; pseudo-first order against its own analytic curve (1e-10) and against mass action
with `[B] / [A] = 1e6` (1.4e-7, of the order of `[A] / [B]`); the equilibrium
`[C] / [A] = K [B]`; the conversion `RT ln 24.46` at 298.15 K and 1 atm; the treatments and
detailed balance; the catalytic cycle; the command lines.

## Solvers

| Name | What |
|---|---|
| `propagate` (default) | contraction to superstates, then the exact propagation of the superstate generator `G = D diag(1^T M)^-1` (closure A; `--closure b`: `D M^-1`): `p_T(t) = diag(1^T M)^-1 exp(G t) (p_T + B p_S)`, `p_S = C p_T` |
| `rcmc` | the published RCMC of Type A (Sumiya et al. 2015; Iwata, Oki, Sakaue 2026): the snapshot `q = V p` after the last step whose reference time does not exceed `t` |
| `legacy_rcmc` | MultiOptPy's `rcmc.py` as it is (for comparison; its differences are reproduced on purpose) |
| `expm` | `exp(t K) p0`, the reference |
| `spectral` | the eigen-decomposition of the symmetrized generator (detailed balance needed; small populations are lost to cancellation, so it is a diagnostic) |
| `steady` | the stationary distribution reached from `p0` (`t -> infinity`) |
| `rosenbrock` | the L-stable Rosenbrock method ROS2 with step-size control (`--ode-tolerance`, relative, 1e-7): the integrator of `mass_action`, and of `canonical` as a check |
| `windows` | mass action by pseudo-first-order windows (`kinetics_beyond_tst.md`, section 3.8, item 1): the states propagated by `propagate` (RCMC's contraction) with the species held over a window, the species moved by the stoichiometry of the fluxes of the window (their time integrals from the block exponential, so the totals free and bound are kept to rounding), a window as long as no species changes by more than `--window-tolerance` (1e-3) of its concentration; first order in the tolerance (`A + B -> C` with `[A]0 = [B]0`: the error falls tenfold per decade of the tolerance against the analytic solution; with `B` in excess one window and the exponential decay to 1e-5) |
| `kmc` | kinetic Monte Carlo by Gillespie's direct method (`--kmc-trajectories`, 10000; `--seed`): the shares of the trajectories in every state with their standard errors in the diagnostics; for `semi_markov` the residence times are drawn from their gamma distributions (Marsaglia-Tsang). A reference for small non-stiff models: a run that needs more than `--kmc-max-events` (1e8) events is refused with the advice to contract |

| `contract` | the contraction of mass action of any reaction order, RCMC's extension to non-linear kinetics (`kinetics_beyond_tst.md`, section 3.8.1): for every output time the species whose effective loss rate `-D_jj` (`D = J_TT - J_TS J_SS^-1 J_ST` of the Jacobian at the composition) times `t` exceeds `--contract-theta` (100) are put on their quasi-steady manifold `f_S = 0` (Newton), fastest first while the eliminated block relaxes as a whole (`-max Re eig(J_SS) t` above the threshold); the start is moved onto the manifold along `[I B]` (every conserved total kept), and `(I + B h') dc_T/dt = f_T` is integrated by the Dormand-Prince pair 5(4); nothing to eliminate: ROS2. For a first-order network the blocks are those of RCMC |

Refused by name: `rcmc_b` (the snapshot of Type B needs a projection onto the simplex in the
`pi`-norm) and `ode` (the integrator is `rosenbrock`).

**`contract`** reproduces the checks of `tools/kinetics/check_nonlinear_contraction.py`
(`kinetics::solve::contract_mass`): for a first-order network `D` and `M = I + B h'` are those
of `kinetics::contract` (1e-12) and the populations those of the matrix exponential; for `A + B
<-> X -> P` the error against ROS2 falls at second order in the separation (below `0.3 /
fast^2` for `k_diss + k_2` = 1e1, 1e2, 1e3, by a factor of more than 50 per decade) and both
totals are kept to 1e-12; dimerization `2 A <-> D -> P`, the enzyme `E + S <-> ES -> E + P`, and
the third-order `2 A + B <-> T -> P` agree with ROS2 within 1e-5 (the table of section 3.8.1:
8e-7, 6e-7, 8e-7); `run_kinetics --model mass_action --solver contract` agrees with
`--solver rosenbrock` within 1e-5 of the concentrations (`tests/cli_kinetics_models.rs`).

**`kmc`** agrees with the matrix exponential within three standard errors on a three-state
network (20000 trajectories), samples gamma draws with their mean and variance, refuses a
stiff network, and the semi-Markov chain reaches the populations of its Markov limit
(`kinetics::solve::kmc`, `kinetics::semi_markov`, `tests/cli_kinetics_models.rs`).

**The greedy order** eliminates the state with the largest `|D_jj|` first (ties: the lower
index), by a subtraction-free elimination (`kinetics::contract`): every update adds
non-negative numbers, the diagonal is rebuilt from the column, and the triangular factors of
`-K_SS` are kept, so that the bounds of the reference times need two triangular solves with
additions only for every prefix. `K_SS` is never factorized by a general LU.

**Reference times of `rcmc`** (`--time-rule`): `gershgorin` (the default, MultiOptPy's and the
one of the experiments of Iwata et al.): `ln 2 / sqrt(sigma_hat(K_SS) rho_hat(D))` with
`rho_hat(A) = min(max row sum, max column sum of |A|)` and `sigma_hat = 1 / rho_hat(K_SS^-1)`;
`eigen`: the same with the eigenvalues of the symmetrized matrices (detailed balance needed;
`O(n^4)`); `diag`: `1 / |D_jj|` of the eliminated state (the original papers). A step after
which nothing is left to react (`rho(D) = 0`) has an infinite reference time under `gershgorin`
and `eigen`, as in the published formula, so the snapshot of a connected network never reaches
its full equilibrium; `propagate` does.

**The cut of `propagate`** is chosen per output time after the greedy order: among the
prefixes whose eliminated states have relaxed (`sigma_hat t >= ln(1 / tol)`), the one with the
smallest separation ratio `eps_hat = rho_hat(D) / sigma_hat(K_SS)`; if that ratio exceeds `tol`,
nothing is eliminated and the answer is `exp(t K) p0`. The default `tol = 1e-8` (`--tolerance`)
keeps the error, of the order of `eps + exp(-sigma t)`, below the accuracy that `--validate`
compares with (`1e-10` to `1e-8`); measured: on the golden network `random7` the populations
at 1 s and 100 s agreed with a 90-digit reference to 4.7e-13, and on four-state models with a
gap from 1e-1 to 1e-8 and on random ten-state networks the error stayed below `eps_hat` and
below 1e-8 (`tests/cli_kinetics.rs`). The superstate generator is small and not stiff on the
scale of `t`.

**The matrix exponential.** `expm`, the residence times, and the degrees of rate control use an
exponential that multiplies non-negative numbers only: with `lambda = 2 max |K_jj|` and
`P = I + K / lambda >= 0`, `exp(tau K) = sum_k exp(-x) x^k / k! P^k` (`x = lambda tau <= 1`),
then squarings; after every squaring a diagonal entry that holds most of its column is rebuilt
as 1 minus the sum of the off-diagonal entries, so that the slow outflow of a state is carried
by off-diagonal entries with a relative error of a few hundred rounding errors. Measured on the
golden network `random7` (rates from 1e-2 to 1e9 1/s) against the 90-digit reference: 1.4e-15
at 1 s and 1.1e-15 at 100 s; the scaling and squaring of a Pade approximant (Higham 2005) gave
3.4e-9 and 5.4e-8, and the non-negative series without the rebuilt diagonals 1.1e-8. The Pade
form is kept for matrices with negative off-diagonal entries (closure B).

## Diagnostics and `--validate`

Every output time reports (screen, `diagnostics.json`): the number of eliminated states,
`eps_hat`, `sigma_hat t` (the initial transient is `exp(-sigma t)`), `rho_hat t`, conservation
`|1^T p - 1|`, the most negative population, and the detailed-balance residual of the
superstate generator. `--validate [TOL]` solves the same model with `expm` and reports the
largest difference against `TOL` (default: `max(eps_hat, exp(-sigma_hat t), 1e-10)` for a
contraction, `1e-10` otherwise); a failed check is a warning and `"passed": false`.

## Exploration priorities and the mapper

`--priority NAME` ranks the states (`priority.csv`), and `run_mapper -map_opt
priority=kinetics` (or MultiOptPy's `--use_rcmc`) explores the minimum with the highest score
in the network as it stands after every job:

| Score | Of state `i` |
|---|---|
| `population` (default) | `p_i(t)`, MultiOptPy's RCMC queue |
| `max_population` | the largest `p_i` for `0 <= t' <= t` (eight points per decade from `0.01 / k_max`): short-lived intermediates that are passed early |
| `residence` | `int_0^t p_i dt`: exact from the block exponential with `propagate` and `expm`, by Gauss-Legendre panels on a logarithmic time axis (two per decade over sixteen decades) with the other solvers |
| `discovery_flux` | `tau_i k_guess / (1 + n_i)` with the visits `n_i` and the largest rate constant of the network as the optimistic rate of an exit not found yet (a proposal of the design) |

The settings of the mapper (`mapper_settings` or `-map_opt`): `kinetics_score`,
`kinetics_solver`, `kinetics_time` (s, 1; `rcmc_reaction_time_s`, `--rcmc_time`),
`kinetics_temperature` (the temperature of the run; `rcmc_temperature_K`,
`--rcmc_temperature`), `kinetics_start` (the node of the first input; `rcmc_start_node_id`,
`--rcmc_start_node`), `kinetics_energy`, `kinetics_clamp`, `kinetics_time_rule`,
`kinetics_tolerance`, `kinetics_rate_theory`, `kinetics_corrections` (`,` between them),
`kinetics_kappa`. The command line goes over `kinetics_*`, which goes over the `rcmc_*` keys of
the file. A network without a transition state yet gives no kinetics; the lowest minimum is
explored then, as in MultiOptPy.

## Sensitivity, uncertainty, spectrum, species

- `--sensitivity [NODE ...]`: the derivatives of the yields (populations) of these nodes
  (default: every state) at every output time and at `t -> infinity`, with respect to every
  parameter of the rates (`sensitivity.csv`, `rate_control.csv`; see
  [Derivatives of the yields](#derivatives-of-the-yields)).
- `--uncertainty SIGMA --samples N --seed S`: independent normal errors of standard deviation
  `SIGMA` (kcal/mol) on the energy of every state and bottleneck, `N` draws solved like the
  model (in parallel, the same result for any number of threads): mean, 5 %, 50 %, 95 %
  quantiles of every population and the share of draws in which a state is the most
  populated (`uncertainty.csv`). At 298 K an error of 1 kcal/mol changes a rate constant by a
  factor 5.4.
- `--spectrum`: the relaxation rates of the generator (the eigenvalues of the symmetrized
  generator) and the ratios of consecutive ones, where the time scales separate
  (`spectrum.csv`).
- `--lump-conformers`: the populations of states with the same bond graph added
  (`species.csv`; networks of mopt-rs, which have the structures).

## Derivatives of the yields

`--sensitivity` (Rust: `kinetics::derivative::derivatives`; Python:
`mopt_rs.kinetics.derivatives`) differentiates the populations `p_i(t)` at every output time
and the final yields `p_i(inf)` (the stationary distribution reached from the start: the
equilibrium of each closed class) with respect to:

| Parameter | `parameter_kind` | Derivative | Unit | `normalized` |
|---|---|---|---|---|
| the energy of a minimum `j` (the energy the rate theory uses: electronic, or the free energy) | `state_energy` | `dp_i / dE_j` | 1/Hartree | `d ln p_i / d(-E_j / RT)` (thermodynamic degree of rate control) |
| the energy of a transition state or crossing point `e` | `bottleneck_energy` | `dp_i / dE_e` | 1/Hartree | `d ln p_i / d(-E_e / RT)` |
| `ln k` of the step `e`, both directions together | `ln_k` | `dp_i / d ln k_e` | 1 | `d ln p_i / d ln k_e` (degree of rate control) |
| the temperature | `temperature` | `dp_i / dT` | 1/K | `d ln p_i / d ln T` |
| `ln [S]` of a species held at a concentration | `ln_concentration` | `dp_i / d ln [S]` | 1 | `d ln p_i / d ln [S]` (the apparent reaction order of the yield) |
| the standard free energy `G°_S` of that species | `species_energy` | `dp_i / dG°_S = (1 / k_B T) dp_i / d ln [S]` | 1/Hartree | `d ln p_i / d(-G°_S / RT)` |

With corrections, the energy of a stationary point moves the corrections too (the barriers of
tunnelling, ...): their tangent is a central difference of the rate assembly with the
stationary point shifted by `1e-4 k_B T` (its electronic and free energies together), added to
the exact part, and the bottlenecks then get passes of their own.

Detailed balance holds for every perturbation: the rates are in flux form, so moving the energy
of a minimum changes its weight `pi_j` and thereby every rate constant out of it, moving a
saddle point changes its flux `F_e` and both of its rate constants, and `ln k_e` scales both
directions by the same factor (the equilibrium constant is kept). A saddle point that the
consistent clamp has put at the level of a minimum moves with that minimum, and its own energy
has no effect. The derivatives with respect to `T` include the prefactor `k_B T / h`, the
Boltzmann factors, the free energies of `tst`, and the corrections (their temperature
dependence is taken by a central difference of the rate assembly with the relative step 1e-5;
the rest is exact).

`ln [S]` moves `mu_S` by `k_B T` on the free side of the steps of `S`: the step from that side
scales with `[S]`, its reverse stays. Every tangent is that of `ln k` of both directions of a
step from the free energies of its two sides and its bottleneck, so a network without a
potential (a catalytic cycle) is differentiated as well. The apparent orders are checked
against central differences in `ln [S]` (1e-7) for `expm` and `propagate`, for `A + B <-> C`
and for the catalytic cycle; the order of the yield of `C` is 1 at the start and `p_A` at the
equilibrium (`p_C = K [B] / (1 + K [B])`), as it should be.

**Method.** Forward-mode dual numbers through the code of the solver that the run uses
(`propagate` with closure A, `rcmc`, `expm`, `steady`; the elimination, the contraction, and the
uniformized exponential are written for any scalar): one pass per parameter, each pass giving
the derivatives of every population at every time; the energies of the bottlenecks follow from
the passes of `ln k` (`dp / dE_e = -(1/k_B T) dp / d ln k_e` unless the clamp is active), so a
network of `n` states and `m` steps needs `n + m + 1` passes, which run in parallel. This was
chosen over an adjoint because it differentiates exactly what the solver reports, with its
greedy order and cut (taken from the real parts, the same in every pass), for all states at
once, and because the passes are independent; the adjoint of the exact exponential (Van
Loan's block integral, `kinetics::sensitivity::rate_control`) is kept and agrees with it to
1e-9. `legacy_rcmc` and `spectral` have no derivatives (refused by name).

**Identities and checks** (`tests/cli_kinetics.rs`): on a four-state network and on a stiff one
(rate constants from 1e-3 to 1e9 1/s), every derivative of `expm` and `propagate` at every time
and of the final yields agrees with central differences of the exact populations (Richardson's
extrapolation of two steps) to 1.1e-8 (the four-state network) and 2.2e-7 (the stiff one) of
its largest entry; the final yields have
`dp_i / dE_j = -(1/k_B T) p_i (delta_ij - p_j)` and no dependence on the saddle points; the
degrees of rate control of a yield satisfy `sum_e d ln p / d ln k_e = t (dp/dt) / p` (scaling
every rate constant scales time; it is 1 only where the yield grows linearly in time) and
`sum_e d ln p / d ln k_e + sum_j d ln p / d ln pi_j = 0`.

**Effective rate constants.** Where the solver contracts, the off-diagonal entries of the
superstate generator `G` (the rate constants between superstates, 1/s) are differentiated too
(`rate_control.csv`: `time_s, from, to, k_eff_s-1, step, degree_of_rate_control`, with a `(sum)`
row for every pair). They are homogeneous of degree one in the rate constants, so their
degrees of rate control sum to 1 exactly, as Campbell's for a rate: for `A <-> I <-> P` with
a short-lived `I`, `k_eff = k_1 k_2 / (k_-1 + k_2)` and the degrees of rate control are
`k_2 / (k_-1 + k_2)` and `k_-1 / (k_-1 + k_2)` (checked to 1e-7; the sum to 1e-12).

**Cost.** One pass costs about one solution in dual arithmetic (about three real operations per
operation, with matrix products written for the scalar instead of those of `faer`). Measured
in the test build (unoptimized): a random network of 30 states and 41 steps, three output
times, 113 parameters in 72 passes, 0.36 s with `propagate` and with `expm`; the four-state
networks take 2 to 6 ms. The log says it for every run (`Derivatives: N parameters in M
forward passes ...`).

`sensitivity.csv` has the columns `time_s` (`inf` for the final yield), `state`, `population`,
`parameter_kind`, `parameter` (label, `T` for the temperature), `parameter_value` (Hartree for
energies, after the clamp for a saddle point; `ln k` of the step from its first to its second
state, k in 1/s; K), `derivative`, `unit`, `normalized` (empty for a population of 0).

## Beyond the canonical model

### The energy-grained master equation (`--model egme`)

The minima with frequencies and structures are divided into grains of `--egme-grain` (50
cm^-1) on one energy axis from the lowest well up to `--egme-above` (40) `k_B T` above the
highest saddle point; dissociation channels and states without frequencies stay thermal
states. The states are counted as `rrkm` counts them, every quantity is a sum over the cells of
a grain, and the generator is in flux form (`kinetics_beyond_tst.md`, section 3.7):

```text
pi[(w,a)]      = exp(-(G_trans + G_elec)_w / kT) sum_{cells of w in a} N_j exp(-E_j / kT)
F[(v,a)][(w,a)] = exp(-(G_trans + G_elec)_ts / kT) (1 / h) sum_{cells of ts in a} W_j delta exp(-E_j / kT)
F[(w,a)][(w,b)] = omega P(a <- b) pi[(w,b)]
```

so that the sums over the grains are the canonical quantities of `rrkm` (the shares the grains
hold are checked and reported) and `omega -> infinity` is `rrkm`. It needs `--rate-theory
rrkm`. Collisions (`--collision-model`): `exponential_down` with the mean energy transferred
downwards (`--energy-transfer`, cm^-1), the upward probabilities from detailed balance,
normalized from the top grain down (the lowest grains, where the upward probabilities add to
more than 1, have their downward part alone normalized and are counted in a note), or
`strong`. The collision frequency is `--collision-frequency` (1/s) or that of Lennard-Jones
collisions at `--pressure` with `--bath SIGMA:EPS:MASS` and `--lennard-jones SIGMA:EPS` (or
`lennard_jones` of a state), with Troe's reduced collision integral. The corrections of a step
multiply its flux at every energy; steps without counted states at their bottleneck (crossings,
electron transfers, barrierless steps) join the grains with their canonical flux. The
populations are reported on the states; `--sensitivity`, `--refine`, and `--priority` are for
the canonical model.

**Phenomenological rate constants** (`phenomenological.csv`): by contraction (the grains
eliminated in the greedy order but the last of every species; closures A and B), by the
eigenpairs of the symmetrized generator (Bartis-Widom, `k = Z Lambda Z^-1`), and the
high-pressure ones of `rrkm` with `k / k_inf`; the gap between the fastest chemical eigenvalue
and the slowest relaxation of the energies says whether the species are defined. Where the
weights of the species span so many orders that the slow eigenvectors are not resolved in
double precision (their components do not sum to zero as the conservation requires; e.g. a
molecule and a dissociation channel 0.4 Hartree below), the eigenpairs give no constants and the
note says so: the contraction, which needs no subtraction, is the result.

**Checks** (`tests/kinetics_models.rs`): with strong collisions the decay of a molecule into a
channel far below has the slowest eigenvalue that solves the secular equation
`sum_a omega f_a / (omega + k_a - lambda) = 1` (1e-6 at `omega` = 1e10, 1e12, 1e14 1/s) and the
contraction agrees within 2 %; at `omega` = 1e22 the contraction is `rrkm` to 1e-6 (both
closures), at 1e4 and 2e4 1/s the rate constant is proportional to `omega` (2e-3), and in
between it rises monotonically below `k_inf`; exponential down is normalized and balanced
(residual below 1e-10); for an isomerization of two wells of similar energy the closures A and
B bracket the eigenpair constants, the deviation falls as the gap grows (below 1 % at
`omega` = 1e15), and the eigenpair constants keep detailed balance (1e-4). A detail found on the
way: the eigen-decomposition of `faer` returned eigenpairs with residuals of the order of the
norm for these matrices of rate constants (norms near 1e14 1/s); the kinetics scales every
symmetric matrix by a power of two before the decomposition (`generator::scaled_symmetric_eigen`;
also used by `spectral`, `--spectrum`, and the time rule `eigen`).

### Hot intermediates (`--model two_grain`)

An intermediate `I` formed from `a` over the saddle point `e` holds the energy `E* = E0(e) +
eps - E0(I)` (`eps` the mean energy of the canonical flux above the saddle point, from the
counted states). With the cooling rate `--cooling-rate` (1/s) the diagnostic is `H =
max_x k_x(E*) / omega` over the other exits `x` of `I` (RRKM at `E*`). Where `H` exceeds
`--hot-threshold` (0.1), a hot grain `I*` is added that the entrance feeds, with the weight
`pi* = F_e / k_e(E*)` taken out of `I`, the cooling `omega pi* (1 - pi* / pi_I)`, the hot exits
`k_x(E*) pi*`, and the thermal exits reduced by what the hot ones carry: lumping `I*` with `I`
gives back the canonical model exactly (checked: the folded populations equal the canonical
ones to 1e-8 with fast cooling), and slow cooling leaves the chemically activated branching
`k_x(E*) / sum_y k_y(E*)`. The screen lists `E*` and `H` of every entrance.

### Semi-Markov chains (`--model semi_markov`)

The mean residence times and the jumps are those of the rate constants (the Markov limit `K =
(P - I) diag(1 / tau)`), the residence time of a state is gamma-distributed with the shape
`residence_shape` of the annotations (1: exponential). The linear solvers use the Markov limit
(exact stationary populations); `--solver kmc` samples the chain itself; `memory.csv` has the
first-order memory correction `K (I - K1)^-1`, `K1 = K diag(tau (cv^2 - 1) / 2)`
(`kinetics_beyond_tst.md`, section 3.6). With `--md-trajectory FILE --md-time-step FS`, the
residence times, their spreads, and the jumps come from a trajectory instead: every frame is
assigned to the minimum with its bond graph (several: the smallest RMSD), frames in no minimum
keep the last one (core sets), the first and the last visit are not counted; `msm.csv` has the
visits, the mean residence times, `cv^2`, and the implied time scales of the reversible Markov
state model at `--md-lag` frames (Prinz et al., J. Chem. Phys. 2011, 134, 174105). The
contraction of memory kernels (`kinetics::semi_markov::kernel_contraction`,
`D(0) (I - D'(0))^-1`) is checked to give closure B for a Markov generator and the expansion
`D(s) = D(0) + s D'(0)` of the resolvent.

### Trajectories from a saddle point

`workflow::kinetics_dynamics` runs pairs of trajectories from the dividing surface of a saddle
point (Wigner sampling of the transverse modes and a Boltzmann momentum along the imaginary
one for a structure on any calculator, `molecule_pairs`; classical sampling for an objective of
plain variables, `objective_pairs`), each forwards and with the velocities reversed, until it
commits to a minimum (bond graph and RMSD for structures, a function of the caller for model
surfaces). `kinetics::branching` turns the pairs into the Bennett-Chandler transmission
coefficient `kappa = sum |J_AB| / F_TST` and the branching fractions of the thermal flux
`phi_AB = |J_AB| / sum |J|` with their standard errors (flux-weighted or counted), and
`kinetics_dynamics::annotations` writes them for `run_kinetics` (`reactive_flux_kappa`,
`outcomes`, `outcome_errors`). `friction_kernel` gives the kernel of `grote_hynes` from dynamics
with the imaginary mode held. On the model surface `V = -b x^2/2 + c x^4/4 + (k0 - k1 x) y^2/2 +
d y^4/4` (a valley-ridge inflection at `x = k0 / k1`), 300 pairs at 300 K split 50:50 within
three binomial standard errors and a tilt of the product valleys favours its side
(`workflow::kinetics_dynamics::tests`). From Python, `kinetics.saddle_trajectories` runs the same
pairs on any surface of `mopt_rs.md.run_md`.

The measured fractions enter through `"outcomes": {"from": STATE, "to": {STATE: SHARE}}` (and
`"outcome_errors"`) of an edge: the bottleneck (with the steps of its N-furcation) is replaced
by steps from that state to every outcome with its share, so detailed balance holds and the
total rate is that of the saddle point. A geometric model of branching fractions is planned
([Planned](#planned)); until then the fractions come from the user or from trajectories.

### Refinement tasks (`--refine [NODE ...]`)

The tasks are ranked by the first-order change of `ln y` that a better value of a parameter
could cause, `|d ln y / d theta| sigma_theta` (`kinetics::refine`), for the population `y` of the
nodes at the last output time (default: the most populated state): `refine_energy` (the energy
of a minimum or saddle point; `sigma` from `uncertainty_kcal_mol` of the annotations or
`--uncertainty`, 1 kcal/mol without either), `branching` (the share of a step of an
N-furcation, the others changed in proportion; `sigma` the standard error of a measured share,
`1 / sqrt(12)` for an assumed one), `coupling` (a crossing without a coupling; `sigma = ln 1e3`).
The derivatives are those of `--sensitivity`. `refinement.csv` lists them; checked on a network
where the saddle point controls the early yield and an assumed split the late one (`d ln y /
d phi = 1 / phi`, 1e-4).

## Planned

Not built yet (each is a later step of `kinetics_beyond_tst.md`, section 7):

- the re-examination of the eliminated set of `contract` along a trajectory (3.8.1, "changing
  composition"): the set is chosen at the start, as in the check script;
- `rcmc_b`, the snapshot of Type B, with its projection onto the simplex in the `pi`-norm;
- the geometric model of branching fractions (Lee and Goodman, J. Am. Chem. Soc. 2020, 142,
  9210; Org. Biomol. Chem. 2021, 19, 3940), validated against the trajectories of model
  surfaces before any use;
- the refinement tasks as tasks of the mapper's queue (3.11; `--refine` ranks them now);
- a command-line route for the trajectories from a saddle point and the friction kernel
  (`workflow::kinetics_dynamics`, from Rust and Python now).

## Annotations

`--annotations FILE` adds data that the network file lacks, by label (`EQ3`, `TS12`) or number:

```json
{
  "nodes": {"EQ3": {"frequencies_cm1": [120.5, 340.2], "symmetry_number": 2}},
  "edges": {"TS12": {"kappa": 0.5, "imaginary_wavenumber_cm1": -1200,
                     "coupling_cm1": 50, "gradient_difference": 0.02, "reduced_mass_amu": 10}}
}
```

Node keys: `energy_hartree`, `free_energy_hartree`, `frequencies_cm1`, `symmetry_number`,
`multiplicity`. Edge keys: the same, `imaginary_wavenumber_cm1`, `kappa`, `coupling_hartree`
or `coupling_cm1`, `gradient_difference` (Hartree/Bohr), `reduced_mass_amu`, `partner` and
`partner_free_state` (a bimolecular edge), `transfer`, `reorganization_hartree` or
`reorganization_kcal_mol`, `delta_g0_hartree`, `transfer_wavenumber_cm1` (an electron
transfer), `cvt_profile_file`,
`cvt_temperature`, `cvt_path` (a path for `cvt`; files relative to the annotations file),
`xyz` (the structure as XYZ text), `friction_s-1`, `friction_memory_s`, `friction_kernel_file`
(`kramers`, `grote_hynes`), `reactive_flux_kappa`, `reactive_flux_error`, `sct_profile_file`
(`small_curvature`), `uncertainty_kcal_mol` (`--refine`), `outcomes`, `outcome_errors` (measured
branching fractions). Further node keys: `xyz`, `residence_shape` (`semi_markov`),
`lennard_jones` (`[SIGMA_ANGSTROM, EPSILON_K]`, `egme`), `uncertainty_kcal_mol`.
Species keys (`"species": {"S": {...}}`):
`energy_hartree`, `free_energy_hartree`, `frequencies_cm1`, `symmetry_number`, `multiplicity`,
`xyz` (the structure as XYZ text, for `tst` and `wertz`), `association`. The mapper does not
fill the frequencies and free energies of its nodes yet, so `tst` needs them here.

## Files

| File | Content |
|---|---|
| `populations.csv` | the population of every state at every output time (`canonical`) |
| `concentrations.csv`, `yields.csv`, `concentrations.svg` | `mass_action`: the concentrations of the states and species (mol/L), the initial and final concentrations and yields |
| `rates.csv` | every step: its states, the energy of the bottleneck (after the clamp), the barriers from both sides (as the step sees them: a free partner counted on its side), `ln k` and `k` both ways, `ln F` of the theory, the factors and their diagnostics, and for a bimolecular step its partner, the free state, the association treatment, and `k_2` (L/mol/s) |
| `rcmc_K_contracted.csv` | MultiOptPy's layout for the last time: the superstate matrix (the header names it: `D` for `legacy_rcmc`, `G = D diag(1^T M)^-1` otherwise), the members, and the populations |
| `rcmc_superstates.json` | the superstates of every time: members, `1^T M`, composition (`C`), entering weights (`B`), the matrix, lifetimes |
| `diagnostics.json` | the model, what was left out, the species (concentrations, `G°`, treatments, the standard state) and the bimolecular steps, whether the network has a potential, and the diagnostics of every time with the comparison of `--validate` |
| `populations.svg` | populations against `log10(t)` (several output times) |
| `sensitivity.csv`, `rate_control.csv` | `--sensitivity`: the derivatives of the yields, the degrees of rate control of the effective rate constants |
| `uncertainty.csv`, `priority.csv`, `spectrum.csv`, `species.csv` | when asked for |
| `phenomenological.csv` | `egme`: the rate constants between the species by contraction (closures A and B), by the eigenpairs, the high-pressure ones, `k / k_inf`, the gap and the slowest eigenvalues |
| `memory.csv`, `msm.csv` | `semi_markov`: the Markov limit and its first-order memory correction with the shapes; from `--md-trajectory` the visits, residence times, `cv^2`, and the implied time scales |
| `refinement.csv` | `--refine`: the tasks of every observable with their sensitivity, `sigma`, and score |
| `mopt.log`, `settings.json` | the log and the options in effect (`--config` reads them) |

## Example: the network of LJ7

`run_mapper` on the seven-atom argon cluster with the built-in `lj` (the settings of
[mapper.md](mapper.md), `-map_opt priority=tabu`, 30 jobs, 41 s) found four minima (EQ0, EQ1,
EQ4, EQ7 at 0, 0.135, 0.216, 0.230 kcal/mol) and six transition states; one of them joins EQ1
with itself and is left out, five join two minima (barriers 0.06 to 0.35 kcal/mol).
`run_kinetics mapper_output -T 40 --time-grid 1e-13 1e-3 11` from EQ0:

| t (s) | EQ0 | EQ1 | EQ4 | EQ7 |
|---|---|---|---|---|
| 1e-13 | 0.9945 | 0.0035 | 0.0010 | 0.0010 |
| 1e-12 | 0.9504 | 0.0309 | 0.0096 | 0.0091 |
| 1e-11 | 0.7885 | 0.1275 | 0.0455 | 0.0385 |
| 1e-10 and later | 0.7671 | 0.1401 | 0.0505 | 0.0422 |

The relaxation times are 4.2, 1.8 and 1.2 ps: no separation of time scales at 40 K. `propagate`
therefore eliminates nothing before 1e-10 s (the matrix exponential) and everything after (one
superstate, the equilibrium); `--validate` found differences below 2e-14. The published `rcmc`
snapshot is off by up to 0.2 here (its own error estimate is 0.6), `legacy_rcmc` by about 0.02
up to 1e-11 s.
At 8 K the slowest relaxation (9.0 ns) is 26 times slower than the next (0.35 ns); the
equilibrium (EQ0 0.99980, EQ1 2.03e-4, EQ4 1.24e-6, EQ7 5.07e-7) is reached by 1e-7 s;
`propagate` agrees with the matrix exponential to 1e-16 at every time, while the `rcmc`
snapshot leaves EQ1 empty (its last step, after which nothing is left to react, has an infinite
reference time), an error of 2e-4.

## Numbers

Energies are differences of Hartree values; `k_B` (1.380649e-23 J/K, 3.1668115635e-6 Hartree/K)
and `h` are the CODATA 2018 values (see Running calculations, Physical constants); MultiOptPy's
`k_B` of 3.166811563e-6 Hartree/K differs by 1.4e-10, so its rate constants agree to rounding. Everything is kept
in logarithms until a solver needs a rate constant; entries more than `exp(-700)` below the
largest rate are dropped and counted. The elimination needs only `+`, `*`, `/` of non-negative
numbers, so every quantity has a high relative accuracy regardless of the dynamic range: the
stationary distribution of a random network with energies spread over 500 `RT` agrees with
`exp(-G / RT) / Z` to 1e-12 per state. Storage is dense (`n^2` per matrix; networks up to a
few thousand states).

## Verification

`tests/cli_kinetics.rs` and the unit tests of `kinetics`: the Eyring rates, both clamps and
parallel saddle points (1e-13); `A <-> B`, `A -> B -> C` (Bateman), and the pre-equilibrium
`k_eff = k_2 K / (1 + K)`; the identities `L K = [D 0]`, `K_SS C + K_ST = 0`, `1^T G = 0`, and
the conservation after unfolding on random generators with and without detailed balance; the
snapshot equals `q = V p` of Iwata et al. computed by dense solves; the models of section 1.4;
the error bound of `propagate`; MultiOptPy's populations, `D`, the members, and
`rcmc_K_contracted.csv` for `legacy_rcmc` (twelve cases of `tests/golden/rcmc_legacy.json`,
made by `tools/golden/rcmc_reference.py` with MultiOptPy's `rcmc.py`); the Eckart transmission
against a Numerov solution of the Schroedinger equation (1e-6), its parabolic limit
`(u/2)/sin(u/2)` and Wigner limit, its symmetry; Skodje-Truhlar's continuity at `beta =
alpha`; the Landau-Zener limits; `tst` against the thermochemistry and the closed-form
vibrational free energy; degrees of rate control against finite differences; the residence
time of `A <-> B`; relabelling invariance of the scores; the command lines and the mapper with
`--use_rcmc`.

## Python

```python
import numpy as np
from mopt_rs import kinetics

K = np.array([[0.0, 2.0, 0.0], [5.0, 0.0, 1e-3], [0.0, 7.0, 0.0]])   # K[i][j]: j -> i, 1/s
sol = kinetics.solve(K, [1.0, 0.0, 0.0], [1e-3, 1.0], solver="propagate")
c = kinetics.contract(K, [1])                 # B, C, D, 1^T M with state 1 eliminated
order = kinetics.greedy_order(K, time_rule="gershgorin")
score = kinetics.scores(K, [1.0, 0.0, 0.0], 1.0, score="residence")
kinetics.run_kinetics("mapper_output", "-T", "300", "--validate")

# Bimolecular steps (species and partners in the annotations): apparent orders and mass action.
d = kinetics.derivatives("network.json", [1e-3, 1.0], annotations="species.json",
                         concentrations={"S": 1e-3})
d.apparent_orders("EQ1")                      # {"S": d ln p / d ln [S]}
m = kinetics.mass_action("network.json", [1e-3, 1.0, 1e3], total_concentration=1e-4,
                         concentrations={"S": 1e-3}, annotations="species.json")
m.final_yields()
kinetics.standard_state_shift(298.15)         # 1.894 kcal/mol

# Single edges: tunnelling, electron transfer, variational TST.
kinetics.tunnelling("bell", 1500.0, (0.03, 0.02), 200.0)          # (kappa, diagnostics)
kinetics.transfer_rate("nonadiabatic", 0.03, 2e-4, -0.01, 298.15) # (k_f, k_b, top)
kinetics.cvt_factor(s, energies, transverse_wavenumbers, 298.15)  # (factor, s*, shift)

# Beyond the canonical model.
m = kinetics.network_model("network.json", [1e-9, 1e-6], model="egme", annotations="ann.json",
                           temperature=1500.0, collision="strong", collision_frequency=1e12)
m["contraction_a"], m["eigen"], m["high_pressure"], m["gap"]
kinetics.refinement_tasks("network.json", [1e-3, 1.0], [2], sigma=1.5)
kinetics.kramers(gamma, omega_b); kinetics.grote_hynes(omega_b, gamma=g, memory=1e-13)
kinetics.weak_coupling(h12, dF, F1F2, mu, 300.0, barrier); kinetics.beyer_swinehart(w, 1.0, 5000)
kinetics.curvature_tunnelling(s, v_adiabatic, mass_ratio, 300.0)   # (sct, zct)
pairs = kinetics.saddle_trajectories(surface, x_ts, hessian, classify, pairs=200)
b = kinetics.branching(pairs, "flux"); b.annotations("TS3", labels)
kinetics.markov_state_model(dtrajs, 3, lag=10, lag_time=1e-12); kinetics.dwell_times(dtrajs, 3, 1e-15)
kinetics.kernel_contraction(K0, K1, [1])
kinetics.solve(K, p0, [1.0], solver="kmc", trajectories=20000, seed=3)
```

## Adding a rate theory, a correction, or a solver

Every part is a trait with a registry in `KineticsParts`; `cli::run_kinetics::run_with_parts`
takes them.

```rust
use std::sync::Arc;
use mopt_rs::kinetics::rate::{Diagnostic, RateCorrection, RateSettings, StepContext};
use mopt_rs::kinetics::{Bottleneck, Conditions, KineticNetwork, KineticsParts};

struct Recrossing(f64);

impl RateCorrection for Recrossing {
    fn name(&self) -> &str {
        "recrossing"
    }
    fn ln_factor(
        &self,
        _b: &Bottleneck,
        _step: &StepContext,
        _net: &KineticNetwork,
        _c: &Conditions,
    ) -> mopt_rs::error::MoptResult<Option<(f64, Vec<Diagnostic>)>> {
        // The same factor in both directions keeps detailed balance.
        Ok(Some((self.0.ln(), Vec::new())))
    }
}

let mut parts = KineticsParts::default();
parts.corrections.register(
    "recrossing",
    Arc::new(|_: &RateSettings| Ok(Arc::new(Recrossing(0.5)) as Arc<dyn RateCorrection>)),
);
// run_with_parts(&config, base, console, &parts) with --corrections recrossing
```

- **Rate theory**: implement `RateTheory` (`state_energy`, `bottleneck_energy`, optionally
  `ln_prefactor`) and register it in `parts.theories` (`--rate-theory NAME`). Return `None` for
  a stationary point that lacks the data: it is left out and listed.
- **Solver**: implement `KineticSolver` (`solve` returns the populations at the times with
  diagnostics; `residence` has a default by quadrature on a logarithmic time axis) and register
  it in `parts.solvers` (`--solver NAME`; the mapper's `kinetics_solver` uses the built-in
  registry).
- **Model**: implement `KineticModel` (`assemble` returns the evolution law) in `parts.models`.
- **Mapper priority**: `ExplorationPriorityRegistry::register_with_options` takes a priority
  with settings of its own; `KineticPriority` is the built-in one.
