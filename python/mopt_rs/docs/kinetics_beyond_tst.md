# Kinetics beyond transition state theory: a framework that contains RCMC

Status: design document. Stages 1 to 4 of section 7 are implemented except the items listed
there as planned (the geometric model of branching fractions and a few extensions); what is
implemented, how it was checked, and what is refused by name are in [kinetics.md](kinetics.md).

Rate constant matrix contraction (RCMC) is used in two places: `run_rcmc` analyses a finished
reaction network, and the reaction-network mapper uses it to decide which minimum to explore
next. RCMC as usually applied rests on one specific kinetic picture (thermalized minima,
first-order steps, Eyring rate constants). This document

1. states every assumption of that picture, separating the assumptions of the contraction
   itself from the assumptions about how rate constants are obtained (section 1);
2. surveys the established theory beyond each assumption, with the diagnostic that tells
   when it matters, the input it needs, and its cost (section 2);
3. constructs one framework in which RCMC is a special case and the extensions are choices
   (section 3), with the numerical formulation (section 4);
4. turns the framework into a module design (section 5), verification tests with known
   answers (section 6), and a staged plan including what not to implement (section 7).

Statements marked **Proposal** are not taken from the literature; each comes with its argument
and a numerical test against direct integration. Numbers quoted as "checked numerically" were
obtained with small NumPy/SciPy scripts against `exp(tK)`; they are to be turned into the unit
tests of section 6.

Notation: `p` is a column vector of populations, `dp/dt = K p`, and `K[i][j]` (i != j) is
the rate constant from state `j` to state `i`, so the columns of `K` sum to zero. This is the
convention of MultiOptPy's `rcmc.py` and of refs [4, 5].

---

## 1. What RCMC assumes

### 1.1 The method as published

RCMC was introduced for multistep organic reaction profiles [1], extended to branching ratios
of unimolecular decomposition [2], reviewed together with its use for steering automated
searches ("kinetics-based navigation") [3, 6, 7], and later reformulated and analysed as a
matrix algorithm [4], with a faster implementation [5]. In the matrix form of ref [4]:

- Input: a rate constant matrix `K` with (RCM1) non-negative off-diagonal entries, (RCM2)
  zero column sums, (RCM3) detailed balance `K[i][j] pi[j] = K[j][i] pi[i]`; an initial
  population `p`; an end time `t_max`.
- States are moved one at a time from the "transient" set `T` to the "steady" set `S`; the next
  one is the state with the largest `|D[j][j]|` (shortest lifetime), where
  `D = K_TT - K_TS K_SS^-1 K_ST` is the Schur complement of `K_SS`. The original papers pick
  the largest off-diagonal element instead.
- Each step `k` has a reference time: `t_diag = 1 / |D[j][j]|` (original) or
  `t_eigen = ln 2 / sqrt(sigma(K_SS) rho(D))` (ref [4]; `sigma`, `rho` are the smallest and
  largest absolute eigenvalues, or cheap bounds on them).
- Output for step `k` (only while `t(k) <= t_max`): `q = V p`, an approximation of
  `exp(t(k) K) p`, with

  ```text
  q_T = V_TT (p_T - K_TS K_SS^-1 p_S)        q_S = -K_SS^-1 K_ST q_T
  M    = I + K_TS K_SS^-2 K_ST
  V_TT = diag(1^T M)^-1   (Type A = the original RCMC)      V_TT = M^-1   (Type B)
  ```

RCMC therefore returns a snapshot: states in `S` are fully equilibrated with their
neighbourhood, states in `T` have not yet reacted. No differential equation is integrated.
Ref [4] proves an error bound in terms of the eigenvalues of `K`, `rho(D)` and `sigma(K_SS)`
and names the extension to non-linear (bimolecular) kinetics as future work.

### 1.2 What MultiOptPy implements

`multioptpy/Utils/rcmc.py` (`RCMCQueue.pop`, `main`; entry point `run_rcmc` in
`Entrypoints/core.py`) and its use in `Wrapper/mapper.py` / `Entrypoints/mapper.py`:

- Rate constants: `k = (k_B T / h) exp(-(E_TS - E_min) / k_B T)` for both directions of every
  edge, with `E` the electronic energy by default and the Gibbs energy with
  `--energy-type free`; `E_TS` is replaced by `max(E_TS, E_min)` separately per direction.
  Parallel edges between the same pair add up.
- Contraction: largest `|D[j][j]|` first, Schur-complement update with the diagonal rebuilt from
  the column sums, reference time `ln 2 / sqrt(sigma rho)` with Gershgorin-type bounds, stop
  when the reference time exceeds the requested reaction time.
- Populations: the initial population (one start node) is projected on `T`, the contracted
  equation `dp_T/dt = D p_T` is integrated with RK4 for the requested time, the eliminated
  states are filled in by `q_S = -K_SS^-1 K_ST q_T`, and everything is renormalized.
- Mapper: with `--use_rcmc` the queue recomputes these populations at every `pop()` and
  returns a task of the node with the largest population; all candidates are accepted
  (no Boltzmann rejection). Options: `--use_rcmc`, `--rcmc_temperature`, `--rcmc_time`,
  `--rcmc_start_node` (settings keys `use_rcmc`, `rcmc_temperature_K`,
  `rcmc_reaction_time_s`, `rcmc_start_node_id`). `run_rcmc` takes `json_file`, `-T`, `-t`,
  `--start-node`, `--output-dir`, `--energy-type`, `--log-level`. `interface.py` has no RCMC
  options.

### 1.3 The assumptions

"Contraction" marks assumptions of RCMC itself; "rates" marks assumptions that only concern how
the entries of `K` are obtained; "MultiOptPy" marks choices of that implementation.

| # | Assumption | Belongs to | Where it is made |
|---|---|---|---|
| A1 | The state of the system is a population vector over minima (EQs); nothing else is remembered (Markov property, exponential residence times) | contraction | master equation of refs [1-5] |
| A2 | Linear, first-order kinetics with a time-independent `K`: unimolecular or pseudo-first-order steps only | contraction | ref [4] ("monomolecular"); bimolecular steps named as future work |
| A3 | `K` is a generator: non-negative rates, zero column sums (closed system, population conserved) | contraction | (RCM1), (RCM2) |
| A4 | Detailed balance with a positive stationary distribution | error analysis of ref [4]; not needed by the contraction formulas (section 3.4) | (RCM3) |
| A5 | Time-scale separation: at the time of interest, the eliminated states are quasi-steady and the kept states have not reacted | contraction | QSSA; error bound of ref [4] |
| A6 | Greedy choice of the eliminated states by lifetime | contraction (heuristic) | refs [1, 4] |
| A7 | Fixed temperature; no pressure, concentration, or time dependence of `K` | contraction (through A2) and rates | all |
| A8 | Each minimum is internally thermalized (canonical ensemble), or all states share one total energy (microcanonical variant of ref [2]) | rates | eq. (3) of ref [4] |
| A9 | Rate constants from conventional TST in Eyring form with a transmission coefficient (set to 1 in practice): no recrossing, classical motion along the reaction coordinate, no tunnelling | rates | eq. (3) of ref [4]; `kappa = 1` in its experiments |
| A10 | Rigid-rotor / harmonic-oscillator free energies (or bare electronic energies) | rates | Gibbs energies at 300 K in ref [4]; MultiOptPy default is the electronic energy with the prefactor `k_B T / h` |
| A11 | One saddle point connects exactly two minima; the products of a step are decided at the saddle point | rates / network model | network of EQ nodes and TS edges |
| A12 | One electronic state, adiabatic (Born-Oppenheimer) motion | rates | implicit |
| A13 | The energies are exact: no uncertainty is propagated | rates | implicit |
| A14 | One start node carrying all population; nodes or edges without the needed energy are dropped | MultiOptPy | `rcmc.py` |

Several saddle points between the same pair of minima are not excluded by RCMC: their rate
constants add. Conformers are not excluded either: when every conformer is a node, the
contraction itself produces the conformational-entropy correction, which was the point of
ref [1].

### 1.4 Differences between MultiOptPy's `rcmc.py` and the published method

Checked numerically against `exp(tK) p` (energies in units of `RT` at 300 K):

| Model | Exact | `rcmc.py` | Published Type A | Contract and propagate (3.4) |
|---|---|---|---|---|
| Start state with a 100 s lifetime next to a fast pair, t = 1 s: population left in the start state | 0.989 | 0.006 | 1.000 | 0.989 |
| Start state decaying with 1 s^-1 into ten degenerate conformers, t = 0.4 s: population left | 0.670 | 0.169 | 1.000 | 0.670 |
| Two states, rate 1e-9 s^-1, t = 1 s: population left | 1.000 | 0.119 | 1.000 | 1.000 |

The causes (line numbers of `multioptpy/Utils/rcmc.py`, version 1.21.2):

1. Lines 411-498: the stop test `time_current > reaction_time_s` is made after `S`, `T`, `D`
   were already updated, so the contraction whose reference time exceeds the requested time
   is kept. Refs [1, 4] discard it. A state whose lifetime is far longer than the requested
   time is then treated as equilibrated (row 1 of the table).
2. Lines 484-485: when one state is left, `rho(D) = 0` and the reference time is not
   updated, so the last contraction is never tested (row 3: any two-state network returns the
   equilibrium distribution at any time).
3. Lines 572-618: the vector integrated with `D` is the superstate population
   `p_T - K_TS K_SS^-1 p_S`, but `D` acts on the population of the representative states.
   The factor `V_TT = diag(1^T M)^-1` of the published method (or `M^-1`) is missing, and the
   global renormalization at lines 612-618 does not replace it. Superstates with many members
   lose and gain population too fast by the factor `1 + sum_j C[j][t]` (row 2).
4. Lines 595-603 (fallback when RK4 would need too many steps): the weights are computed as
   `1 - K_TS K_SS^-1 1_S`; the published Type A uses `1_T^T M = 1 - 1_S^T K_SS^-1 K_ST`
   (the transposed product). The comment says the stationary distribution would be needed;
   it is not.
5. Lines 367-371: clamping `E_TS` to `max(E_TS, E_u)` and `max(E_TS, E_v)` separately breaks
   detailed balance when the saddle point lies below the higher minimum (possible with Gibbs
   energies or mixed levels of theory): `k_uv / k_vu` is then not `exp(-(E_v - E_u)/RT)`.
   A consistent clamp uses `max(E_TS, E_u, E_v)` for both directions.
6. Lines 389-394 against 544-571: when the start node is absent (`None` is passed by
   `Wrapper/mapper.py` line 2104 if the setting is missing), the population vector starts at
   the lowest-energy node but the contracted initial vector is uniform over `T`.
7. `Entrypoints/mapper.py` lines 480-486 and `Wrapper/mapper.py` lines 2101-2106 never pass
   `use_free_energy`: inside the mapper RCMC always uses electronic energies, although the
   Boltzmann queue and the reference energy switch to Gibbs energies when available.
8. Lines 93-168: the file `rcmc_K_contracted.csv` is described as the superstate rate matrix
   but contains the Schur complement `D`. The superstate matrix of the original method is
   `D diag(1^T M)^-1` (section 3.4).
9. Lines 467-473 and 513: `K_SS` is refactorized by LU after every contraction (cost
   `O(n^4)` per `pop()`), and LU on a matrix whose entries span tens of orders of magnitude
   is not guaranteed to keep relative accuracy; refs [4, 5] update a triangular factor
   incrementally without subtractions.

Recommendation: the default of mopt-rs follows the published method and section 3.4; the
MultiOptPy numbers stay reproducible under the name `legacy_rcmc` for comparison, and the
differences are recorded in [migration_workflows.md](migration_workflows.md) when the code is written.

---

## 2. Beyond each assumption

For every topic: the theory that relaxes the assumption, when it matters (the diagnostic to
compute), the input, and the cost relative to one transition-state optimization with a
frequency calculation.

### 2.1 Better TST rate constants (A9, A10)

| Theory | Relaxes | Matters when (diagnostic) | Input | Cost |
|---|---|---|---|---|
| Quasi-RRHO, hindered rotors [16-18] | harmonic low-frequency modes | modes below about 100 cm^-1; flexible or associated species | frequencies (already available) | none |
| Symmetry numbers [22] | missing rotational symmetry / path degeneracy | point groups of minimum and saddle point differ | point group (`vib::symmetry`) | none |
| Wigner, Eckart, Skodje-Truhlar tunnelling [12-14] | classical barrier crossing | `u = h nu_im / k_B T > 1`; H transfer; `T` near or below `T_c = h nu_im / (2 pi k_B)` | imaginary frequency, forward and reverse barriers | none |
| Small-curvature tunnelling [15] | one-dimensional tunnelling path | `u > 3`, strongly curved paths | IRC with Hessians along the path | 10-50x |
| Canonical / microcanonical variational TST [8-10] | dividing surface fixed at the saddle point | low or broad barriers, loose saddle points, large entropy change along the path | IRC with projected frequencies at several points | 5-20x |
| Variable-reaction-coordinate TST [11] | no saddle point (barrierless association) | radical-radical, ion-molecule association | interaction potential sampled over orientations (10^3-10^5 energies, often multireference) | very high |
| Multi-structural TST [19, 20], Curtin-Hammett [21] | one structure per stationary point | several conformers within a few `RT` of the lowest one | conformer ensembles of minima and saddle points | conformer search |
| Semiclassical TST, instanton, ring-polymer rates [55-57] | separable, shallow tunnelling | deep tunnelling (`T < T_c`) | anharmonic constants or path-integral sampling | very high |
| Energetic span model [23] | (a limit, not an extension) | steady-state turnover of one catalytic cycle | Gibbs energies of the cycle | none |

Notes.

- Eyring form with corrections: `k = kappa_tun gamma_rec (sigma_rxn) (k_B T / h) (c0)^(1-m)
  exp(-dG_act / RT)`, `m` the molecularity and `c0` the standard-state concentration. The
  change of standard state from 1 atm to 1 mol/L is `RT ln(24.46)` = 1.89 kcal/mol at 298 K
  per association step and is as large as most corrections in this table.
- Wigner: `kappa = 1 + u^2 / 24`. For an imaginary frequency of 1000 cm^-1 at 298 K,
  `u = 4.8`, `kappa = 2.0`, and `T_c = 229 K`: Wigner is already outside its range and Eckart
  or Skodje-Truhlar should be used. The diagnostic to print is `u` and `T / T_c`.
- Conformers. Two exact statements connect the multi-structural picture to the network
  picture. (i) If all conformers of a species interconvert much faster than they react, the
  species is one superstate with partition function `sum_c Q_c exp(-E_c/RT)` and the rate
  constant out of it is the population-weighted sum of the conformer rate constants
  (multi-structural TST); the product ratio depends only on the saddle-point free energies
  (Curtin-Hammett). (ii) Contracting a network in which every conformer is a node gives
  exactly this when the separation holds and remains correct when it does not. The
  diagnostic is the separation ratio `eps` of section 3.5 for the conformer block.
- Energetic span: the steady-state turnover frequency of a single cycle is a ratio of
  sums of exponentials of energy differences; the energetic span keeps the largest term. In
  the framework it is the stationary flux of a driven generator (section 3.9), which needs
  no approximation; the energetic span is the report of which term dominates.

### 2.2 Recrossing and dynamics in solution (A9)

| Theory | Relaxes | Matters when (diagnostic) | Input | Cost |
|---|---|---|---|---|
| Reactive flux (Bennett-Chandler) [24, 25] | no recrossing of the dividing surface | flat barrier tops, strong solvent coupling, ion pairs, charge transfer | free-energy profile along a coordinate + 10^2-10^3 short trajectories from the dividing surface | 10^4-10^6 gradients |
| Kramers [26, 28] | no friction | `gamma / omega_b` not small; viscosity dependence of the rate | barrier frequency `omega_b`, friction `gamma` | none once `gamma` is known |
| Grote-Hynes [27] | Markovian friction | sharp barriers (`omega_b` above solvent relaxation rates) | friction kernel from constrained MD at the barrier top | one MD run |
| Smoluchowski / Collins-Kimball [29, 30] | reaction-limited encounter | `k_act` within two orders of the diffusion limit `k_D` | diffusion coefficients or viscosity, contact radius | none |

Formulas: Kramers (moderate to high friction) `kappa = sqrt(1 + (gamma / 2 omega_b)^2) -
gamma / (2 omega_b)`; Grote-Hynes `kappa = lambda_r / omega_b` with `lambda_r = omega_b^2 /
(lambda_r + zeta_hat(lambda_r) / mu)`; Collins-Kimball `k = k_D k_act / (k_D + k_act)`,
`k_D = 4 pi (D_A + D_B) R N_A`, about `8RT / (3 eta)` = 7.4e9 L mol^-1 s^-1 in water at
298 K. The cheap diagnostic for every bimolecular step is `k_act / k_D`: above 0.01 the
Eyring value must be capped.

Recrossing factors are typically between 0.1 and 1, i.e. worth less than 1.4 kcal/mol at
298 K. They rarely change a mechanism; they matter for absolute rates and for steps that are
compared with diffusion.

### 2.3 Non-statistical behaviour: where the assumptions of TST fail (A8, A11)

Here the population picture itself breaks down: where the system goes is not decided by
free energies of stationary points.

| Phenomenon | What fails | Diagnostic | Prediction of branching | Cost |
|---|---|---|---|---|
| Post-transition-state bifurcation (valley-ridge inflection, VRI) [31-36] | A11: one saddle point feeds two or more products; TST gives the total rate but no product ratio | along the IRC, the lowest eigenvalue of the Hessian projected on the space orthogonal to the gradient changes sign before a minimum is reached; a second saddle point connects the two products | quasi-classical trajectories from the saddle point [43-45]; cheaper: geometric models from the two saddle points and products [35, 36], correlations with partial bond lengths at the saddle point [34] | 10^2-10^3 trajectories of 0.2-1 ps, or none for the models |
| Dynamic matching [37, 38] | A8: momentum at the saddle point carries the system across a shallow intermediate into one exit | intermediate well shallower than the kinetic energy released into the reaction coordinate; exit direction collinear with the entrance | trajectories; momentum-direction models | as above |
| Hot (chemically activated) intermediates [39-42] | A8: the intermediate reacts before it has cooled | `H = k_next(E*) / omega_cool` at the energy `E*` with which the intermediate is formed; non-thermal when `H > 0.1` | RRKM `k(E)` + master equation with energy transfer (2.4); in solution a cooling rate of the order of 10^11-10^12 s^-1 | density of states: none; cooling rate: estimate or MD |
| Incomplete intramolecular vibrational energy redistribution [38, 41] | RRKM itself: energy is not randomized within the intermediate | lifetime at `E*` below a few ps; mode-specific exits | trajectories only | high |
| Roaming [46, 47] | the reaction avoids the saddle-point region altogether | near-dissociation plateaus at long range | global surface + trajectories, or a statistical roaming model | very high |

Notes.

- Bifurcation and dynamic matching have the same structure for a kinetic model: one
  dynamical bottleneck with several outcomes. The framework represents both by branching
  fractions attached to the bottleneck (section 3.3). Fractions come, in order of cost, from
  the user, from a geometric model, or from trajectories started at the saddle point with
  quasi-classical normal-mode sampling at the temperature of interest. With `N`
  trajectories the standard error of a fraction `phi` is `sqrt(phi (1 - phi) / N)`:
  100 trajectories fix a 50:50 ratio to +-5 %.
- Hot intermediates stay inside statistical theory as long as energy is randomized within
  the intermediate; then an energy-resolved master equation is sufficient, and the alkene
  hydroboration case that was first explained by non-statistical dynamics [40] was
  reproduced that way [39]. The diagnostic `H` separates the three regimes: `H << 1`
  thermal (RCMC valid), `H` of order one energy-resolved model needed, lifetime below the
  redistribution time trajectories needed.

### 2.4 Energy- and pressure-resolved kinetics (A7, A8)

- RRKM: `k(E) = sigma W_ts(E - E0) / (h rho(E))`, sums and densities of states by direct
  count [48, 49]; tunnelling enters by convolving `W_ts` with a transmission probability
  [50]. The canonical TST rate constant is the Boltzmann average of `k(E)`.
- Energy-grained master equation (EGME) [51-54, 58]: the states are (well, energy grain);
  collisions with a bath move population between grains of one well (frequency `omega`,
  exponential-down model with the upward rates fixed by detailed balance), reaction moves
  population between wells at fixed energy with `k(E)`, bimolecular channels are sources
  and sinks.
- Phenomenological rate constants: the EGME generator has as many slow ("chemically
  significant") eigenvalues as there are chemical species, separated from the fast
  internal-energy-relaxation eigenvalues; the slow eigenpairs define `k(T, p)` between
  wells, including direct "well-skipping" rates (Bartis-Widom analysis) [51-53]. When a
  slow eigenvalue merges with the relaxation continuum, two species have to be merged.
- Limits: `omega -> infinity` gives canonical TST for every step (the RCMC picture);
  `omega -> 0` gives rates proportional to pressure; the strong-collision model gives
  `k_uni = integral of omega k(E) f(E) / (omega + k(E)) dE`.
- When it matters: gas phase below the high-pressure limit (diagnostic `k_uni / k_inf`),
  chemically activated association, and hot intermediates in solution (2.3). In solution
  at thermal energies with barriers above about 10 kcal/mol the high-pressure limit holds.
- Input: frequencies and rotational constants of wells and saddle points (available),
  collision model (Lennard-Jones parameters and mean energy transferred per collision, or
  a cooling rate). Cost: none in electronic-structure terms; the linear algebra is the
  same as for a network with a few thousand states.

Section 3.7 shows that the contraction applies to the EGME unchanged and how it relates to
the eigenvalue analysis.

### 2.5 Non-adiabatic steps (A12)

| Theory | Applies to | Diagnostic | Input | Cost |
|---|---|---|---|---|
| Non-adiabatic TST at the minimum-energy crossing point (MECP) [59-61] | spin-forbidden steps, weakly coupled surface crossings | reactant and product of a step have different multiplicity; two spin states within a few kcal/mol at a minimum | MECP geometry and energy, effective Hessian on the seam, gradient difference, coupling (spin-orbit) | MECP optimization (exists in `pes/combine`) + coupling |
| Landau-Zener, weak-coupling (Airy) probability [61, 62] | hopping probability at the seam | `P_LZ` between 0.01 and 0.99 means neither limit applies | coupling `H12`, gradients of both states, reduced mass along the hopping coordinate | none |
| Marcus theory and extensions [63-65] | electron transfer, outer-sphere and intramolecular | charge localization changes between two minima without bond rearrangement | reorganization energy, driving force, coupling | available from the EDEEL analysis [66] |
| Surface hopping, conical-intersection dynamics [67] | photochemistry, ultrafast decay through conical intersections | excited-state minima and minimum-energy conical intersections in the network | excited-state gradients and couplings, swarms of trajectories | very high |

Formulas: in canonical form `k = (k_B T / h) (Q_MECP / Q_R) exp(-E_MECP / k_B T) <p_sh>`,
with the double-passage Landau-Zener probability `p_sh = 1 - P_LZ^2`, `P_LZ = exp(-2 pi
H12^2 / (hbar dF v))`, averaged over the energy in the hopping coordinate. In the
weak-coupling limit `<p_sh> = 4 pi^(3/2) H12^2 / (hbar dF) sqrt(mu / (2 k_B T))`. With
the small spin-orbit couplings of light elements the factor is often 10^-4 to 10^-1, i.e.
1.4 to 5.5 kcal/mol at 298 K: a
spin-forbidden step is easily misjudged by more than the barrier-height error if the factor
is left out.

### 2.6 Non-Markovian and non-linear kinetics, open systems (A1, A2, A3, A4, A7)

- Memory. Projecting any dynamics on a set of states gives a generalized master equation
  with a memory kernel [68, 69]; it reduces to a rate matrix when the kernel decays faster
  than the populations change. Markov state models estimate the rate matrix (or transition
  matrix at a lag time) from MD and validate the Markov property by the convergence of
  implied time scales [70, 71]; memory-kernel variants keep the kernel instead of
  lengthening the lag time [72]. Optimal reductions of a Markov model to fewer states are
  known, with and without memory [73]. Diagnostic: residence-time distributions that are
  not exponential; implied time scales that depend on the lag time.
- Elimination of states in Markov chains is the same algebra under several names:
  stochastic complementation [77], graph transformation for mean first-passage times and
  committors [74-76], lumping [78], and the Grassmann-Taksar-Heyman algorithm for
  stationary distributions, which avoids subtractions [79]. Coarse-graining of first-order
  networks that preserves the slow time scales without assuming internal equilibrium was
  given in ref [80].
- Bimolecular steps. Mass-action kinetics is non-linear: `dc/dt = N v(c)`. The established
  tools are microkinetic modelling with stiff integrators [84-86], quasi-steady-state and
  partial-equilibrium reductions with error estimates [81-83], and Jacobian-based
  time-scale analysis (computational singular perturbation, intrinsic low-dimensional
  manifolds) [87-89]. Diagnostic: any step whose rate depends on a concentration that
  changes by more than a few per cent over the time of interest.
- Stochastic kinetics: kinetic Monte Carlo (Gillespie / BKL) [90-92] samples the same
  master equation; it is exact for small copy numbers and for non-exponential waiting
  times, and unusable for stiff networks (the number of events scales with the fastest
  rate times the simulated time).
- Open and driven systems: flow (sources and sinks), photochemical steps (a rate set by
  the photon flux, not by temperature), electrochemical steps (potential-dependent rates;
  the computational hydrogen electrode [95] shifts free energies, Marcus-Hush-Chidsey gives
  the rate [96]). Such networks have non-equilibrium steady states with cycle fluxes [93,
  94]; detailed balance (A4) does not hold. Diagnostic: a cycle whose product of forward
  rate constants differs from the product of backward ones.
- Exploration driven by kinetics exists for non-linear models too: rate-based enlargement
  of mechanisms [97, 98] and concentration-flux steering [99, 100].

### 2.7 Uncertainty (A13)

At 298 K an error of 1 kcal/mol in a barrier changes the rate constant by a factor 5.4;
typical density-functional barrier errors are 1-3 kcal/mol. This dominates over recrossing,
shallow tunnelling, and symmetry factors, and is comparable to conformational and
standard-state effects.

- Local sensitivity: derivative of an observable (yield, turnover frequency, population at
  time `t`) with respect to the free energy of each minimum and saddle point; the degree of
  rate control is the normalized derivative with respect to saddle-point free energies
  [101, 102]. Derivatives of RCMC yields were used to find controlling steps in large
  networks [103].
- Global: sample correlated energy errors, propagate each sample through the kinetic model,
  report quantiles and the probability that a ranking changes [104-108].
- Cost: none in electronic-structure terms; `N` kinetic solutions for `N` samples.

### 2.8 Typical sizes of the corrections at 298 K

| Effect | Typical factor on `k` | Equivalent energy |
|---|---|---|
| Barrier-height error of the electronic-structure method | 5-150 | 1-3 kcal/mol |
| Standard state (1 atm to 1 mol/L, per association) | 24 | 1.9 kcal/mol |
| Low-frequency modes (harmonic vs quasi-RRHO), flexible systems | 2-100 | 0.5-3 kcal/mol |
| Conformational entropy (`n` equivalent conformers) | `n` | `RT ln n` |
| Spin-forbidden step (Landau-Zener factor) | 10^-4 - 10^-1 | 1.4-5.5 kcal/mol |
| Tunnelling, H transfer, `u` = 3-6 | 1.5-10 | 0.2-1.4 kcal/mol |
| Symmetry numbers | 1-12 | up to 1.5 kcal/mol |
| Recrossing in solution | 0.1-1 | up to 1.4 kcal/mol |
| Variational effect, tight saddle point | 0.5-1 | up to 0.4 kcal/mol |
| Bifurcation (product ratio not given by TST at all) | ratio anywhere in 0-1 | not an energy |

Consequence for priorities: uncertainty propagation, standard states, quasi-RRHO, conformer
ensembles, symmetry, and the spin-crossing factor come first; cheap tunnelling next;
branching at bifurcations wherever a VRI is detected; everything that needs dynamics only
on demand.

---

## 3. The general framework

### 3.1 The kinetic model

A kinetic model consists of

- a set of states with a population (or concentration) each;
- a rule for the time derivative: a generator matrix, a kernel, or a rate law;
- conserved quantities;
- optionally a reference measure `pi` with respect to which detailed balance holds.

| Model | States | Evolution | Conserved | Detailed balance |
|---|---|---|---|---|
| M0 canonical rate matrix (RCMC) | minima | `dp/dt = K p`, `K` from Eyring | `1^T p` | yes, `pi_i = exp(-G_i/RT)` |
| M1 corrected rate matrix | minima (and conformers) | `K` with `k = kappa gamma sigma k_TST` | `1^T p` | yes if the correction multiplies both directions (3.2) |
| M2 multi-outcome bottlenecks | minima | `K` assembled from bottlenecks with branching fractions | `1^T p` | yes if the fractions are those of the thermal flux (3.3); otherwise stated as not imposed |
| M3 energy-grained master equation | (well, energy grain) | `dp/dt = K p` with collisions and `k(E)` | `1^T p` (plus sinks) | yes, `pi = rho(E) exp(-E/k_B T)` |
| M4 mass-action kinetics | species | `dc/dt = N v(c)` | element / moiety totals `W^T c` | thermodynamic consistency of `k+ / k-` |
| M5 generalized master equation | minima | `dp/dt = integral K(t - s) p(s) ds` | `1^T p` | on the integrated kernel |
| M6 driven / open | any of the above | generator plus sources, sinks, non-thermal rates | none or partial | no |

M0 to M3 and the linear cases of M6 share one data structure, a sparse generator, and one
set of solvers. M5 reduces to a generator with a correction (3.6). M4 needs a different
solver but reuses rate constants, conditions, sensitivities, and priorities.

### 3.2 Generators in flux form; corrected rate constants

Every detailed-balanced generator can be written with one symmetric non-negative matrix of
equilibrium one-way fluxes `F[i][j] = F[j][i]` and the reference weights `pi`:

```text
K[i][j] = F[i][j] / pi[j]   (i != j),      K[j][j] = -sum_i K[i][j]
ln pi[j]   = -G_j / RT
ln F[i][j] = ln(k_B T / h) - G_ts(ij) / RT + sum_c ln f_c(ij)
```

`G_j` and `G_ts` are Gibbs energies (or electronic energies in the MultiOptPy-compatible
mode), and `f_c` are the correction factors of the step. A bottleneck is thus a weight
`F`, not a pair of rate constants; both directions follow from it, and detailed balance
cannot be broken by rounding, clamping, or inconsistent corrections.

Which corrections are symmetric, i.e. belong in `F`:

| Correction | Symmetric? | Remark |
|---|---|---|
| tunnelling (Wigner, Eckart, Skodje-Truhlar, small curvature) | yes | the transmission probability at a given total energy is the same from both sides, so the thermal factor referred to the saddle-point energy is the same in both directions; formulas with an endoergicity term (Skodje-Truhlar) must be evaluated with it to keep this |
| recrossing (reactive flux, Kramers, Grote-Hynes) | yes | a property of the dividing surface |
| variational TST | yes | replaces `G_ts` by the maximum of the free-energy profile |
| symmetry numbers, conformer ensembles | yes when put into `G_j`, `G_ts` | `G` then includes `-RT ln` of the degeneracy |
| non-adiabatic probability at an MECP, Marcus | yes | the seam is the bottleneck |
| diffusion limit (Collins-Kimball) | yes | harmonic combination of two weights: `1/F = 1/F_act + 1/F_diff` |
| clamp for saddle points below a minimum | yes if `G_ts >= max(G_i, G_j)` is enforced on `F` | replaces rule 5 of section 1.4 |
| photochemical, electrochemical driving, sinks | no | stored as directed entries next to `F`; the model is then M6 |

Pseudo-first-order steps: a bimolecular step `A + B -> C` with `B` in excess enters as
`K[C][A] = k2 [B]`. Its weight uses the standard-state free energies and the factor
`[B]/c0`; the stationary distribution then contains the activity of `B`.

### 3.3 Bottlenecks with several outcomes (bifurcations, dynamic matching)

A bottleneck `b` is a dividing surface crossed with total equilibrium flux `F_b`. In TST a
crossing from side `a` ends in the one minimum on the other side. In general it ends in
minimum `j` with probability `phi_b(a -> j)`, the fraction of the trajectories crossing `b`
from `a` that are next thermalized in `j`:

```text
F[j][a] += F_b phi_b(a -> j)          K[j][a] = F[j][a] / pi[a]
```

- TST: one outcome with `phi = 1`.
- Post-transition-state bifurcation: reactant `R`, saddle point `b`, products `P1`, `P2`
  with `phi_1 + phi_2 = 1`; `k(R -> P_i) = phi_i k_TST`. The direct interconversion of
  `P1` and `P2` over the second saddle point is a separate bottleneck.
- Dynamic matching / skipping of a shallow intermediate `I`: outcomes `I` (thermalized),
  and the minima beyond `I` reached directly.

Detailed balance. If the fractions are those of the equilibrium (thermal) flux through `b`,
microscopic reversibility makes the flux from `P_i` back to `R` through `b` equal to
`F_b phi_i`, so the contribution to `F` is symmetric and the reverse rate constant is
`k(P_i -> R) = F_b phi_i / pi[P_i]`. The model then has the correct equilibrium and all of
section 3.4-3.5 applies. This is the default. If the fractions are measured for a
non-thermal ensemble (for instance trajectories started with a fixed excess energy), the
symmetric assignment is still used for the generator, the fact is recorded on the
bottleneck (`thermal = false`), and the report states that the fractions are not
equilibrium quantities; an energy-resolved treatment (3.7) is the consistent alternative.

Sources of `phi`, in order of cost: user input; a geometric model; trajectories. Unknown
fractions of a detected bifurcation default to equal shares with an uncertainty that spans
`[0, 1]`, which the priority uses (3.11).

### 3.4 Contraction of a general generator

Let `K` be any matrix with non-negative off-diagonal entries and non-positive column sums
(a generator, possibly with sinks; detailed balance is not assumed). Split the states into
`S` (to eliminate) and `T` (to keep), with `K_SS` non-singular (every state of `S` can reach
`T` or a sink). Define

```text
B = -K_TS K_SS^-1      |T| x |S|, B >= 0   B[t][j]: probability that the system, started in j,
                                            enters T first at t (columns sum to 1 without sinks)
C = -K_SS^-1 K_ST      |S| x |T|, C >= 0   C[j][t]: quasi-steady population of j per unit
                                            population of t
D = K_TT + K_TS C      Schur complement: generator of the process watched only while in T
M = I + B C            |T| x |T|, M >= 0
```

Exact identity. With `L = [I  B]` (acting on `(p_T, p_S)`), `L K = [D  0]`, because
`K_TS + B K_SS = 0`. Hence the lumped populations

```text
q = p_T + B p_S        obey        dq/dt = D p_T        exactly,
```

and `1^T q = 1^T p` when there are no sinks. Nothing has been approximated yet: `q` assigns
the population of every eliminated state to the kept states it will reach first.

Closure. The quasi-steady-state approximation `p_S = C p_T` gives `q = M p_T` and a closed
equation for the superstates:

```text
dq/dt = G q,       G = D M^-1              (closure B)
                   G = D diag(1^T M)^-1    (closure A: each superstate owns its column of C)
unfold:            p_T = M^-1 q  or  diag(1^T M)^-1 q,       p_S = C p_T
```

In projection language this is a Petrov-Galerkin reduction `G = L K R` with restriction
`L = [I  B]` and prolongation `R = [I; C] M^-1`, `L R = I`.

Properties (checked numerically, including a driven network without detailed balance):

- `1^T G = 0` and the unfolded populations sum to `1^T q` for both closures.
- `G` of closure A has non-negative off-diagonal entries. If `K` satisfies detailed balance,
  so does it, with the stationary weights `pi_t (1^T M)_t = pi_t + sum_j B[t][j] pi_j`: the
  equilibrium population of the superstate.
- `G` of closure A is the superstate rate matrix of the original RCMC: the recursion of
  refs [1, 2] (as transcribed in Appendix B of ref [4]) gives `D diag(1^T M)^-1` to
  rounding error.
- `1^T M = 1 - 1_S^T K_SS^-1 K_ST` needs only `K`, not the stationary distribution.

RCMC as a special case. The published RCMC output is `q(0) = L p(0)` unfolded without any
propagation: `p_T = V_TT L p(0)`, `p_S = C p_T`. It is the correct answer when the kept
states have not reacted (`rho(D) t << 1`) and the eliminated ones have relaxed
(`sigma(K_SS) t >> 1`).

**Proposal: contract and propagate.** Use the contraction only to remove the stiffness and
integrate the superstates exactly:

```text
p_T(t) = N exp(G t) L p(0),    p_S(t) = C p_T(t),    N = diag(1^T M)^-1 or M^-1
```

The set `S` is chosen such that every eliminated state has relaxed (`sigma(K_SS) t >=
ln(1/tol)`), and `G` is then small and non-stiff on the scale of `t` (its largest rate is
of the order of `ln(1/tol) / t`), so a dense matrix exponential is cheap and accurate. The
argument: `q` obeys `dq/dt = D p_T` exactly, so the only error is the closure, which is of
the order of the separation ratio (3.5), plus the neglected initial transient
`exp(-sigma(K_SS) t)`. To our knowledge the RCMC papers do not propagate the superstate
matrix; MultiOptPy attempts it but without the factor `N` (section 1.4). Checked
numerically: the three models of section 1.4 are reproduced to four digits, and for a
four-state model with a tunable gap the maximum error is between `0.07 eps` and `0.09 eps`
over `eps` from 0.3 to 1e-4 for both closures. When no cut satisfies the tolerance (no gap
in the spectrum), fewer
states are eliminated, down to none, and the method degrades to the plain matrix
exponential instead of to a wrong answer.

### 3.5 Validity, error estimate, diagnostics

Let `sigma = sigma(K_SS)` (slowest relaxation rate of the eliminated block) and `rho =
rho(D)` (fastest rate left). The quasi-steady manifold `p_S = C p_T` is the first
approximation of the slow invariant subspace `p_S = X p_T`, which solves the Riccati
equation `K_ST + K_SS X = X (K_TT + K_TS X)`. One fixed-point step from `X0 = C` gives
`X1 = C + K_SS^-1 C D`, so the relative error of the manifold is

```text
delta = || K_SS^-1 C D || / || C ||,   of the order of   eps = rho / sigma
```

(for a driven network without detailed balance, `eps = 5.0e-6` and `delta = 4.0e-6` were
found, with an actual error of 1e-7).

- Snapshot RCMC at the reference time `t_eigen`: by our reading of the bound of ref [4]
  (its Theorem 4.6), modes slower than `rho` contribute an error of order
  `|lambda| t + |lambda| / sigma` and modes faster than
  `sigma` an error of order `rho / |lambda| + exp(-|lambda| t)`; with a clean gap the
  error is of order `sqrt(eps)`, and a mode with `|lambda| t` near 1 is wrong by up to
  one half (it is treated as either not started or finished).
- Contract and propagate: error of order `eps + exp(-sigma t)`; no restriction on modes
  with `|lambda| t` near 1.

Diagnostics that every solution reports:

1. `eps_hat = rho_hat(D) rho_hat(K_SS^-1)` from the Gershgorin-type bounds of ref [4]
   (two triangular solves), and optionally `delta`.
2. `sigma_hat t` (initial transient) and `rho_hat t`.
3. Conservation `|1^T q - 1^T p|` and, when detailed balance is claimed, the largest
   `|ln(K[i][j] pi[j]) - ln(K[j][i] pi[i])|`.
4. With `--validate`: the difference to the direct solution (matrix exponential or stiff
   integration) in the maximum norm, compared with `max(eps_hat, 1e-10)`.

Greedy order (A6). Eliminating the state with the largest `|D[j][j]|` is a heuristic for
finding a cut with a large gap; ref [4] gives a justification through the error bound. For
contract and propagate the cut is chosen after the greedy order is known: among the
prefixes with `sigma_hat t >= ln(1/tol)`, take the one with the smallest `eps_hat`; if that
is above the tolerance, take a shorter prefix.

### 3.6 Memory kernels

Eliminating `S` exactly from `dp/dt = K p` gives a generalized master equation for `T`:

```text
dp_T/dt = K_TT p_T + integral_0^t K_TS exp(K_SS (t - s)) K_ST p_T(s) ds + K_TS exp(K_SS t) p_S(0)
Laplace:  s p_T(s) - p_T(0) = D(s) p_T(s) + K_TS (s - K_SS)^-1 p_S(0),
          D(s) = K_TT + K_TS (s - K_SS)^-1 K_ST
```

Expanding at small `s` (populations slow compared with the kernel):
`D(s) = D - s B C + O(s^2)`, so

```text
zeroth order:  dp_T/dt = D p_T                       (Schur complement = Markov limit of the kernel)
first order:   M dp_T/dt = D p_T,  i.e. dq/dt = D M^-1 q   (closure B of 3.4)
initial term:  K_TS (s - K_SS)^-1 p_S(0) -> B p_S(0)        (the restriction L)
```

The matrix `M` is thus the first-order memory correction, `M = I - dD/ds` at `s = 0`
(checked numerically). This gives a recipe that does not depend on where the kernel comes
from.

**Proposal: contraction of kernels.** For a generalized master equation
`s p(s) - p(0) = Khat(s) p(s)`, eliminate `S` in the Laplace domain,
`D(s) = Khat_TT + Khat_TS (s - Khat_SS)^-1 Khat_ST`, and use the Markovian reduction
`G = D(0) (I - D'(0))^-1`. Validity: `|D'(0)|` times the rates of interest is small; test
by comparison with a numerical inverse Laplace transform or with simulation.

Two practical consequences:

- Semi-Markov input. If a state `j` has a residence-time distribution with mean `tau_j`
  and branching probabilities `P[i][j]` (for instance from trajectories through a hot
  intermediate), the Markov limit of its kernel is `K[i][j] = P[i][j] / tau_j`.
  Non-exponential residence enters only through `D'(0)`, i.e. through the second moment.
  Trajectory data therefore enter a rate matrix as lifetimes and branching fractions.
- Markovian embedding. Memory that matters is represented by additional states instead of
  a kernel: a hot intermediate `I*` next to the thermal `I`, with a cooling rate
  `I* -> I` and its own exit rates, is a two-grain energy-grained model and keeps the model
  linear and Markovian, so every solver of this framework applies.

### 3.7 The energy-grained master equation in the framework

States are pairs (well `w`, grain `a` with energy `E_a` on a common absolute scale):

```text
collisions:   K[(w,a)][(w,b)] = omega_w P_w(E_a <- E_b),   P_w(E_a <- E_b) f_w(E_b) = P_w(E_b <- E_a) f_w(E_a)
reaction:     K[(v,a)][(w,a)] = W_vw(E_a) / (h rho_w(E_a))
sinks:        loss W_prod(E_a) / (h rho_w(E_a)) from grain (w,a) to bimolecular products
weights:      pi[(w,a)] = f_w(E_a) = rho_w(E_a) exp(-E_a / k_B T)
flux form:    F[(v,a)][(w,a)] = W_vw(E_a) exp(-E_a / k_B T) / h
```

It is a generator in flux form (3.2), so assembly, checks, contraction, and solvers are the
ones of the canonical model. The canonical model is its limit: for `omega -> infinity`
each well relaxes to `f_w / Q_w`, and lumping gives `K[v][w] = sum_a k_vw(E_a) f_w(E_a) /
Q_w`, the canonical TST rate constant.

Contraction and the chemically-significant-eigenvalue reduction. Both project the dynamics
on the slow subspace. The eigenvalue analysis uses the exact invariant subspace and needs
an eigen-decomposition of a matrix with a huge dynamic range, which loses accuracy in
double precision at low temperature [58]. The contraction uses the quasi-steady manifold,
whose error is of the order of the ratio of the chemically significant eigenvalues to the
relaxation eigenvalues (3.5), and needs no subtraction (4.3).

**Proposal: phenomenological rate constants by contraction.** Eliminate grains in the
greedy order until one representative grain per well is left (or until the time criterion
stops; then wells whose interconversion is faster than energy relaxation remain merged,
which is the species-merging step of the eigenvalue analysis). The lumped generator `G`
is the matrix of phenomenological rate constants; entries between wells that are not
directly connected are the well-skipping rate constants, because the Schur complement sums
over all paths through eliminated grains. Checked numerically for a two-well model with
exponential-down collisions (180 + 168 grains, barrier 12 `k_B T`):

| `omega` | `k / k_inf` (eigenvalue) | closure A / eigenvalue | closure B / eigenvalue | relaxation / chemical eigenvalue |
|---|---|---|---|---|
| 1e-4 | 8e-6 | 0.909 | 1.076 | 8e1 |
| 1e0 | 0.041 | 0.951 | 1.038 | 2e2 |
| 1e2 | 0.574 | 0.992 | 1.005 | 1e3 |
| 1e4 | 0.988 | 0.9999 | 1.0001 | 7e4 |
| 1e6 | 0.9999 | 1.0000 | 1.0000 | 7e6 |

The fall-off curve is reproduced, the high-pressure limit is the canonical rate constant,
and the deviation shrinks with the eigenvalue gap; the two closures bracket the eigenvalue
result in this model. The microcanonical application of RCMC in ref [2] treats one total
energy; the energy-grained use with collisions is not taken from the RCMC papers.

A minimal energy-resolved model for hot intermediates in solution (two grains per
intermediate, `I*` and `I`, section 3.6) covers the regime `H` of order one of section 2.3
at no cost: `K[I][I*] = omega_cool`, exits of `I*` from `k(E*)`, and the entrance step
feeds `I*` instead of `I`.

### 3.8 Non-linear kinetics

Mass action: `dc/dt = N v(c)`, `v_r = k_r prod_s c_s^nu_sr`, conserved totals `W^T c` with
`W^T N = 0`. The Jacobian is `J = N (dv/dc)`.

What carries over, in order of rigour:

1. Pseudo-first-order windows (exact while the reservoir is constant). If in every step at
   most one tracked species appears on each side and all other partners are reservoir
   species (excess reagent, catalyst resting state, solvent), the model is the linear
   generator `K(c_res)` and everything above applies. A changing reservoir is handled by
   windows: propagate with `K(c_res)` over a window in which no entry of `K` changes by
   more than a tolerance, update `c_res` by stoichiometry, repeat. This is the
   time-dependent linearization; its error is controlled by the window tolerance and
   checked against direct integration.
2. Direct stiff integration. The reference solver for mass action: an L-stable implicit
   method with error control (4.4). Networks from the mapper are small enough (up to 10^4
   species, sparse Jacobian) that this is affordable, and it has no validity condition.
3. Quasi-steady-state and partial-equilibrium reduction with a validity check. A species
   set `S` is eliminated by solving `f_S(c_S, c_T) = 0`; the error estimate of ref [81],
   `|dc_j| = |(dc_j/dt) / J[j][j]|` evaluated on the reduced trajectory, must stay below a
   tolerance relative to `c_j`, otherwise the species is returned to `T`.
4. Contraction of the mass-action equations themselves (3.8.1): the extension of RCMC to
   reactions of any order.

Recommendation: implement 1, 2, and 4; 2 is the reference that 4 is validated against
(`--validate`); 3 remains a diagnostic.

#### 3.8.1 Contraction of mass-action networks of any reaction order

A proposal, checked numerically (`tools/kinetics/check_nonlinear_contraction.py`; the numbers
below are its output). It applies the algebra of 3.4 to the Jacobian at the current
composition, with the eliminated species on their quasi-steady manifold instead of in a linear
steady state. For first-order networks every formula below is the one of 3.4.

**Setting.** `dc/dt = f(c) = N v(c)` with mass-action rates of any integer order (`A + B ->`,
`2 A ->`, `2 A + B ->`, ...), `J = df/dc`. The species are split into an eliminated set `S` and
a kept set `T`.

**Manifold.** The eliminated species follow the kept ones: `f_S(c_S, c_T) = 0`, written
`c_S = h(c_T)`, with `h' = -J_SS^-1 J_ST` (implicit function theorem). An eliminated species
may enter its own rate law non-linearly (`2 A <-> D` with `A` eliminated gives
`c_A = sqrt(k_d c_D / k_a)`); the manifold is then found by Newton iterations.

**Lumping.** With `B = -J_TS J_SS^-1`,

```text
d/dt (c_T + B c_S) = f_T + B f_S + (dB/dt) c_S ,
```

and in `f_T + B f_S` the terms linear in the distance from the manifold cancel
(`J_TS + B J_SS = 0`): `c_T + B c_S` does not feel the fast relaxation. This is `L K = [D 0]`
of 3.4 for a Jacobian. `B` is the stoichiometric share of the kept species in an eliminated
one: for `2 A <-> D` with `A` eliminated, `B = 1/2`, and `c_D + c_A / 2` is the conserved
amount of dimer.

**Reduced equation.** To first order in the separation of time scales the eliminated species
lag behind the manifold by `J_SS^-1 h' dc_T/dt`; inserting this into `f_T` gives

```text
(I + B h') dc_T/dt = f_T(h(c_T), c_T) ,
```

the form `M dp_T/dt = D p_T` of 3.4 (`M = I + B C`, `f_T = D p_T` when `f` is linear). Without
the factor `I + B h'` this is the textbook quasi-steady-state approximation, whose error is of
first order in the separation; with it the error is of second order (table below).

**Conserved totals.** For every `w` with `w^T N = 0`, `w_S^T = w_T^T B`, hence
`w_T^T (I + B h') dc_T/dt = w^T f = 0` on the manifold: element and moiety totals are kept
exactly by the reduced equation, although the eliminated species are no longer integrated.

**Start.** The composition is moved onto the manifold along `[I B]`:
`c_T + B c_S = c_T(0) + B c_S(0)` with `c_S = h(c_T)`. This is the starting value of the slow
solution continued back to `t = 0` (matched asymptotics), not the composition at `t = 0`: a
product of an eliminated species that has yet to build up starts slightly below zero. The
rate laws are therefore evaluated as signed polynomials during the propagation, and the
reported composition is taken at the requested time.

**Which species.** As in RCMC, fastest first: the effective loss rate of a kept species is
`-D_jj` with `D = J_TT - J_TS J_SS^-1 J_ST`, and species are eliminated while
`rate * t > theta`. The rate depends on the composition (for `A` in `A + B ->` it is
`k c_B`), which is how the molecularity enters. A second condition is new: the eliminated
block must relax as a whole within the horizon, `-max Re eig(J_SS) * t > theta`. All forms of
a catalyst are short-lived, their total is not; a set that contains every species of a
conserved total has a singular `J_SS`. (In linear RCMC the same condition keeps one state of
every closed component.)

**Changing composition.** Because the rates depend on `c`, the fast set can change along the
trajectory (a reagent in excess is used up). The set is re-examined at the ends of windows in
which no effective rate changes by more than a tolerance: species whose block no longer
satisfies the conditions return to `T` (with `c_S = h(c_T)` as their starting values), others
are eliminated (with the projection above); both conserve the totals. The check script fixes
the set at the start, which suffices for its cases.

**Where it fails, and how that shows.** `J_SS` must be stable: an autocatalytic or
oscillatory fast block has eigenvalues with non-negative real part and is refused. The error
grows as the separation shrinks (first-order network below: 5e-3 at `theta = 10`). Rate laws
of non-integer order are not polynomials below zero; such networks need the start clipped,
which costs one order of accuracy at the start.

**Check against stiff integration of the full equations** (relative error of the composition
at the final time; conserved totals to 1e-14 in every case):

| Model | Eliminated | Error |
|---|---|---|
| first-order network of 7 states | 1 state | blocks equal to RCMC's (`D`, `M`: 1e-20; `q`: 1e-13); 5e-3, 4e-5, 2e-5 at `theta` = 10, 100, 1000; total population to 1e-13 |
| `A + B <-> X -> P`, `k_diss + k_2` = 1e1 ... 1e5 | `X` | 7e-4, 7e-6, 7e-8, 7e-10, 7e-12 (second order); plain quasi-steady state: 3e-2 ... 3e-6 (first order) |
| `A + B <-> X` strong and fast, `X -> P` slow (the complex is the major species) | `B` (rate `k c_A`) | 2e-8 |
| `2 A <-> D` fast, `D -> P` slow | `A` (rate `4 k c_A`) | 8e-7 |
| `E + S <-> ES -> E + P`, substrate in excess | `E` | 6e-7 |
| catalytic cycle of four catalyst forms with a second-order side reaction `2 CI -> Q` | three of the four catalyst forms | 3e-5 |
| `2 A + B <-> T` fast, `T -> P` slow (third order) | `A` | 8e-7 |

Taking the projected composition as the answer without propagating (the analogue of a
snapshot) is not usable for these models (errors of 0.3 to 1): the slow steps proceed during
the time of interest. The propagated form is the one to implement.

**Relation to existing methods.** The manifold is the quasi-steady-state approximation with
its validity estimate [81, 82]; the factor `I + B h'` is its first-order correction from
singular perturbation theory; computational singular perturbation [87] reaches the same order
with eigenvectors of the Jacobian instead of species as the fast directions. What is taken
from RCMC is the greedy choice of species by effective loss rate against the time of
interest, the lumped variables `c_T + B c_S` with their conservation, and the reduction to
RCMC for first-order networks.

### 3.9 Open and driven systems

Sources, sinks, and non-thermal steps are directed entries next to the symmetric flux part.
The contraction of 3.4 needs no detailed balance, so superstates and lumped generators are
defined as before; the error bound of ref [4] does not apply (eigenvalues may be complex),
so the diagnostics of 3.5 and the comparison with direct integration are mandatory.

Steady state: `K pi = 0` solved by eliminating all states but one with the same
subtraction-free elimination (the Grassmann-Taksar-Heyman algorithm [79]) and unfolding
with `C`. Fluxes `J[i][j] = K[i][j] pi[j] - K[j][i] pi[i]` and cycle affinities [93] give
the turnover frequency of catalytic cycles without the energetic-span approximation; the
energetic span is reported as the dominant term.

### 3.10 Sensitivity and uncertainty

Parameters `theta`: `G_j`, `G_ts`, `ln f_c`, `ln phi`. In flux form the derivatives of `K`
are trivial (`dK[i][j]/dG_ts = -K[i][j]/RT`, and so on).

- Observable `y = w^T p(t)`: `dy/dtheta = integral_0^t lambda(s)^T (dK/dtheta) p(s) ds`
  with the adjoint `lambda(s) = exp((t - s) K^T) w`. One forward and one adjoint solution
  give the derivative with respect to every parameter; in the contracted model both are
  small matrix exponentials.
- For a handful of parameters, forward mode with the dual numbers of `math::ad` through the
  contraction and the matrix exponential is simpler; the block identity `exp([[K, dK], [0,
  K]] t)` gives the same derivative and serves as its test.
- Degree of rate control: `X_e = d ln y / d(-G_ts(e)/RT)`, and the thermodynamic degree of
  rate control `X_j = d ln y / d(-G_j/RT)` [101, 102]; for a steady-state rate the `X_e`
  sum to one.
- Uncertainty: first order, `var(ln y) = sum_theta X_theta^2 sigma_theta^2 / (RT)^2`;
  sampling, `N` draws of the energies with a user covariance (independent by default, with
  an option for a common shift per level of theory, which cancels in relative rates),
  quantiles of populations and probabilities of rankings.

### 3.11 Exploration priority

The mapper asks which minimum (or which refinement) to do next. A priority is a functional
of the kinetic solution for the current network, start distribution `p(0)`, temperature,
and time horizon `t_max`.

| Name | Score of node `i` | Origin |
|---|---|---|
| `population` | `p_i(t_max)` | MultiOptPy (`RCMCQueue`) |
| `max_population` | `max over t <= t_max of p_i(t)` | reaches short-lived intermediates that are passed early |
| `residence` | `tau_i = integral_0^t_max p_i(t) dt` | expected time spent in `i` |
| `discovery_flux` | `tau_i a_i`, with `a_i = k_guess g(n_i)` | **Proposal**, below |

Kinetics-based navigation as published restricts the search to the minima that are
kinetically accessible from the start structure at the given temperature and time [3, 6];
the exact selection rule of the original implementation could not be checked in the primary
text, so the names above are defined here and `population` reproduces MultiOptPy.

**Proposal: discovery flux.** The quantity the search wants to maximize is the expected
number of reactive events, within `t_max`, through channels that are not yet in the
network. If node `i` has an unknown exit with rate constant `a_i`, that number is
`tau_i a_i` to first order in `a_i`. The unknown `a_i` is modelled as an optimistic rate
constant `k_guess` (default: the Eyring value for the lowest barrier found so far out of
any node) times a factor `g(n_i)` that decreases with the number of trials already started
from `i` (`1 / (1 + n_i)` or `exp(-alpha n_i)`, the visit penalty of the existing `tabu`
rule). A long-lived populated minimum and a short-lived intermediate through which all
flux passes are both ranked high only if exits unknown so far could compete there.
`residence` is the special case of equal `a_i`. Test: on a toy network with one hidden
low-barrier exit, the rule finds it in fewer trials than `population`.

How non-TST information changes the priority. The principle: it changes the kinetic model
first (so every score changes consistently), and second it creates refinement tasks that
compete with exploration tasks in the same queue.

| Finding | Change of the model | Task created |
|---|---|---|
| Bifurcation detected on an edge (VRI on the IRC, or IRC end points that differ between runs) | the edge becomes a bottleneck with outcomes `P1`, `P2`; fractions unknown: equal shares, uncertainty `[0, 1]` | "branching": geometric model, then trajectories; exploration of `P1` and `P2` uses the optimistic fraction, so neither is starved |
| Step between different multiplicities | the edge is an MECP bottleneck with the Landau-Zener factor; without a coupling value a default with a wide uncertainty | "MECP search" where two spin states of a populated minimum lie within a threshold; "coupling" where the factor controls a populated path |
| Barrierless association or dissociation found (no saddle point on the path) | edge without saddle point: weight from the capture / diffusion limit (upper bound), marked as a bound | "variational" refinement if the step has a high degree of rate control |
| Low barrier after a strongly exothermic step (`H > 0.1`) | hot grain `I*` added; direct flux to the minima beyond `I` | none; nodes beyond `I` rise in priority through the model |
| Large tunnelling diagnostic (`T < T_c`) | Eckart factor applied; uncertainty of the edge widened | "tunnelling" refinement only on request |
| Edge with high degree of rate control and large uncertainty | none | "refine energy" (higher level of theory, conformer search of the saddle point) |

Score of a refinement task: `|X_theta| sigma_theta`, the first-order change of the target
observable that the refinement could cause, converted to the scale of the exploration
scores by the same observable. Pruning (not exploring nodes below a population threshold)
uses the upper quantile of the population under the current uncertainties, never the mean,
so that an uncertain branching fraction or coupling cannot hide a region.

---

## 4. Numerical formulation

Rate constants span more than 200 orders of magnitude in real networks [4, 5]. Rules:

### 4.1 Assembly in log space

- Store `ln pi_j = -G_j/RT` and `ln F_e` per bottleneck. Shift all `ln pi` by their maximum.
- `ln K[i][j] = ln F[i][j] - ln pi[j]`. Parallel bottlenecks between the same pair are
  added with log-sum-exp. Rate constants are exponentiated only when a solver needs them.
- Boltzmann sums (partition functions over conformers, energy grains, thermal averages of
  `k(E)`) use log-sum-exp; quotients of sums are differences of logarithms.
  `math::stable` gets `log_sum_exp`, `log_diff_exp`, `ln_sinh`, `ln_cosh` next to the
  existing `normalized_exp` and `exp_shifted`.
- The Eckart transmission probability is evaluated as `2 sinh(2 pi a) sinh(2 pi b) /
  (cosh(2 pi (a + b)) + cosh(2 pi d))` (equal to the textbook form `1 - (cosh(2 pi (a -
  b)) + cosh(2 pi d)) / (cosh(2 pi (a + b)) + cosh(2 pi d))`) with every term divided by
  `exp(2 pi (a + b))`; `d` may be imaginary (then `cosh` becomes `cos`). The thermal
  average uses Gauss-Laguerre quadrature above the barrier and Gauss-Legendre below.
- Entries below a floor (default `exp(-700)` relative to the largest rate) are dropped and
  counted in the diagnostics; the log-domain scalar of 4.3 removes the need for a floor.

### 4.2 Symmetrized generator

With detailed balance, `Ksym = Pi^(-1/2) K Pi^(1/2)` is symmetric:
`ln Ksym[i][j] = ln F[i][j] - (ln pi_i + ln pi_j)/2`. Its eigenvalues are real and
non-positive and its eigenvectors orthogonal, so the symmetric eigensolver of `faer`
applies. The spectral solution `p(t) = Pi^(1/2) U exp(Lambda t) U^T Pi^(-1/2) p(0)` is
used for diagnostics (spectrum, gaps, chemically significant eigenvalues, relaxation
times), not as the reference for small populations: components below about `1e-16` of the
largest are lost to cancellation [5].

### 4.3 Elimination

One elimination step (state `j`, remaining states `i`, `k`), written so that no two numbers
of the same sign are subtracted:

```text
out      = -D[j][j]                      (positive; rebuilt, never accumulated)
D[i][k] += D[i][j] D[j][k] / out         (i != k: sum of non-negative terms)
D[k][k]  = -(sum_{i != k} D[i][k]) - loss[k]     (rebuilt from the column; loss[k] >= 0 for sinks,
loss[k] += loss[j] D[j][k] / out                  updated like an off-diagonal entry)
```

and the factors needed for `B`, `C`, `1^T M` are accumulated in the same way (the
incremental triangular factor of refs [4, 5]; all its entries have fixed signs). This is
the Grassmann-Taksar-Heyman scheme [79]: every quantity is computed to high relative
accuracy regardless of the dynamic range, and the order of elimination does not affect
stability, only fill-in and, for RCMC, the meaning of the cut. Consequences:

- The algorithm needs only `+`, `*`, `/` on non-negative numbers. It is written once,
  generic over a scalar: `f64` (default), a log-domain scalar (`ln x` stored, addition by
  log-sum-exp; no underflow at any temperature), and dual numbers for sensitivities.
- Pivot: largest `out`; ties are broken by the lower state index so that results do not
  depend on storage order. A pivot with `out = 0` (isolated state) is skipped and reported.
- Sparse storage (adjacency maps) and the lazy greedy selection of ref [5] for large
  networks; dense storage below a few hundred states.
- `K_SS` is never factorized by a general LU.

Checks after every solution: column sums of `G` (zero by construction; verified to
`1e-12` relative), `|1^T q - 1^T p| <= 1e-12`, non-negativity of `B`, `C`, `1^T M`, and
the detailed-balance residual of `G` when the input has detailed balance.

### 4.4 Direct solutions for validation

- Linear, dense: `exp(tK)` by scaling and squaring with a Pade approximant [109, 110].
  For a generator the result is a stochastic matrix, the squarings are benign, and the
  absolute error is of the order of `1e-13`; this is the reference for populations above
  about `1e-10`.
- Linear, large or sparse: the action `exp(tK) p` by an L-stable implicit integrator with
  step-size control on a logarithmic time axis; a few hundred steps cover 20 decades.
- Non-linear: a Rosenbrock method of order 4 with an embedded error estimate (linearly
  implicit, one LU per step, analytic Jacobian from the rate laws) as the default, and a
  variable-step second-order L-stable method (TR-BDF2 type) as an independent cross-check
  [111, 112]. Conserved totals are monitored (`|W^T c - W^T c(0)| <= 1e-10`), and negative
  concentrations below the tolerance are clipped and reported.
- Stochastic: Gillespie's direct method with the project's random number generator, for
  small non-stiff models and for semi-Markov tests.

Stated tolerance for the contraction against the direct solution:
`max_i |q_i - (exp(tK) p)_i| <= max(eps_hat, 1e-10)`, and `<= 1e-8` whenever
`eps_hat <= 1e-9`.

---

## 5. Module design

### 5.1 Placement

The kinetics code is pure numerics (no I/O) and is used both by the mapper (workflow) and
by `run_rcmc`, so it sits with the algorithm layers. Additions to the layer map of
[architecture.md](architecture.md):

```text
 L6  cli/run_rcmc.rs            options and settings file -> RcmcConfig -> job
 L5  workflow/rcmc.rs           reads the network, runs a kinetic job, writes the outputs
     workflow/explore/priority.rs   + ExplorationPriority and the kinetic priorities
 L4  kinetics/                  (new) no I/O; uses math/, domain/, parameters/, and the
                                thermochemistry of vib/ (pure functions at this layer)
       network.rs    states, bottlenecks, annotations (pure data)
       conditions.rs temperature, pressure, standard state, concentrations, bath
       rate/         RateTheory, RateCorrection, registry: eyring, tst, tunneling, recrossing,
                     diffusion, nonadiabatic, electron_transfer, capture
       generator.rs  sparse generator in flux form; symmetrized form; checks
       model/        KineticModel: canonical, egme, mass_action, semi_markov
       contract.rs   subtraction-free elimination, B, C, 1^T M, superstates
       solve/        KineticSolver: rcmc, legacy_rcmc, propagate, expm, ode, spectral, kmc, steady
       observe.rs    populations, residence times, fluxes, committors, first-passage times, turnover
       sensitivity.rs uncertainty.rs
       rrkm.rs collision.rs   densities of states, k(E), energy-transfer models
       branching.rs  outcome fractions: user, model, trajectory counts
 L2  math/           stable.rs (+ log-sum-exp family)  expm.rs  ode.rs  quadrature.rs
                     special.rs (scaled Bessel I0/I1, Airy)  logreal.rs (log-domain scalar)
```

The rate formulas of `analysis/electron_transfer.rs` (Landau-Zener, Marcus) are pure
functions and are needed by `kinetics/rate`; they move to `kinetics/rate/electron_transfer.rs`
and `analysis` re-exports them, so that no lower layer depends on a higher one. The
thermochemistry code (`vib::thermo`) must likewise expose its partition functions and `G(T)` as pure
functions at or below this layer; report writing stays in `analysis/`.

### 5.2 Traits

Sketches in the style of `workflow/explore/priority.rs` (names registered in a registry,
`create(name)` returns a boxed trait object, user implementations can be registered from
Rust and Python).

```rust
/// One dynamical bottleneck: a saddle point, a crossing seam, or a barrierless step.
pub struct Bottleneck {
    pub kind: BottleneckKind,          // Saddle, Mecp, ElectronTransfer, Barrierless, Diffusive
    pub from: Vec<StateId>,            // reactant side (two entries: bimolecular)
    pub outcomes: Vec<(StateId, f64)>, // product side with branching fractions (sum 1)
    pub data: StationaryData,          // energies, frequencies, symmetry, ... (what exists)
    pub thermal_fractions: bool,       // fractions measured on the thermal flux (3.3)
}

/// ln of the symmetric equilibrium flux of a bottleneck and what it was made of.
pub struct LnFlux {
    pub ln_tst: f64,
    pub factors: Vec<(String, f64)>,   // (correction name, ln factor)
    pub diagnostics: Vec<Diagnostic>,  // u, T/T_c, k_act/k_D, P_LZ, H, ...
}

/// A rate theory: stationary-point data and conditions -> ln flux.
pub trait RateTheory: Send + Sync {
    fn name(&self) -> &str;
    fn ln_flux(&self, b: &Bottleneck, net: &KineticNetwork, c: &Conditions) -> MoptResult<LnFlux>;
}

/// A multiplicative correction (tunnelling, recrossing, non-adiabatic probability, ...).
pub trait RateCorrection: Send + Sync {
    fn name(&self) -> &str;
    fn ln_factor(&self, b: &Bottleneck, net: &KineticNetwork, c: &Conditions)
        -> MoptResult<Option<(f64, Vec<Diagnostic>)>>; // None: does not apply to this bottleneck
}

/// A kinetic model assembles the evolution law from a network.
pub trait KineticModel: Send + Sync {
    fn name(&self) -> &str;
    fn assemble(&self, net: &KineticNetwork, rates: &RateSet, c: &Conditions)
        -> MoptResult<Evolution>;      // Generator (linear) or MassAction (non-linear)
}

/// A solver or reducer: evolution law + initial state + times -> solution with diagnostics.
pub trait KineticSolver: Send {
    fn name(&self) -> &str;
    fn solve(&mut self, ev: &Evolution, p0: &[f64], times: &[f64], rng: &mut Rng)
        -> MoptResult<KineticSolution>; // populations(t), superstates, diagnostics
}

/// Choice of the next task of a network search.
pub trait ExplorationPriority: Send {
    fn name(&self) -> &str;
    /// Score per candidate (higher first); `None` prunes the candidate.
    fn rank(&mut self, net: &NetworkView, candidates: &[Candidate], rng: &mut Rng)
        -> MoptResult<Vec<Option<f64>>>;
}
```

Any existing `SamplingPriority` (Boltzmann, tabu, water filling) is usable as an
`ExplorationPriority` through an adapter that passes energies and visit counts; the kinetic
priorities of 3.11 are `ExplorationPriority` implementations that own a rate theory, a
model, and a solver. `Candidate` is `Explore { node }` or `Refine { bottleneck, kind }`.

Built-in names:

| Registry | Names |
|---|---|
| rate theory | `eyring` (MultiOptPy: `k_B T / h` and one energy per stationary point), `tst` (partition functions, standard state), `rrkm` |
| corrections | `wigner`, `eckart`, `skodje_truhlar`, `kappa` (given value), `kramers`, `grote_hynes`, `collins_kimball`, `landau_zener`, `weak_coupling`, `marcus`, `symmetry`, `variational`; built besides: `bell`, `zhu_nakamura`, `small_curvature`, `zero_curvature`, `reactive_flux` |
| models | `canonical`, `egme`, `mass_action`, `semi_markov`; built besides: `two_grain` (the hot intermediates of 3.6 and 3.7) |
| solvers | `rcmc` (published Type A), `rcmc_b`, `legacy_rcmc` (MultiOptPy as it is), `propagate` (contract and propagate), `expm`, `ode`, `spectral`, `kmc`, `steady`; built besides: `rosenbrock` |
| priorities | `boltzmann`, `tabu`, `water_filling` (existing), `population`, `max_population`, `residence`, `discovery_flux` |

### 5.3 Data needed from earlier stages

| Consumer | Needs | Provided by |
|---|---|---|
| `eyring` | electronic or Gibbs energy of minima and saddle points | network file (as in MultiOptPy) |
| `tst`, quasi-RRHO, symmetry | frequencies, rotational constants, mass, multiplicity, point group | vibrational analysis and thermochemistry (`vib`, `freqmain`), point groups (`vib::symmetry`) |
| `wigner`, `eckart`, `skodje_truhlar` | imaginary frequency, forward and reverse barriers | vibrational analysis at the saddle point |
| `variational`, small-curvature tunnelling | energies, gradients, projected frequencies along the path | IRC (`irc`, `ircmain`), with Hessians at selected points |
| bifurcation flag, outcomes | sign change of the lowest projected Hessian eigenvalue along the IRC; second saddle point | IRC (the VRI test lives next to it), mapper |
| trajectory fractions, lifetimes, friction kernels, reactive flux | trajectories started at a saddle point; constrained MD | molecular dynamics with thermostats (`md`, `mdmain`) |
| `landau_zener`, `weak_coupling` | MECP geometry, energies, both gradients, seam Hessian, coupling | `pes/combine` (MECP / MECI optimization); coupling from the engine or the user |
| `marcus` | reorganization energy, driving force, coupling | EDEEL analysis (`analysis/electron_transfer.rs`) |
| `rrkm`, `egme` | frequencies and rotational constants, collision model | vibrational analysis; parameters from the user |
| conformer lumping | conformer ensembles of minima and saddle points | conformation search, saddle-point conformer search of the mapper |
| Markov / semi-Markov input | transition counts, lifetimes | trajectory-based reaction discovery (the nanoreactor of the mapper, `workflow::mapper::nanoreactor`) |

Every item degrades gracefully: a missing datum disables the correction for that bottleneck
and is listed in the report; nothing is guessed silently.

### 5.4 Command line and settings of `run_rcmc`

MultiOptPy's options keep their names, defaults, and meaning: positional `json_file`,
`-T/--temperature` (300), `-t/--time` (1.0), `--start-node` (0), `--output-dir`,
`--energy-type electronic|free`, `--log-level`. The MultiOptPy network file
(`nodes[].node_id / energy_hartree / free_energy_hartree`, `edges[].node_id_1 / node_id_2 /
ts_energy_hartree / ts_free_energy_hartree`) is read unchanged. Additions:

| Option | Meaning |
|---|---|
| `--method NAME` | solver (5.2); `legacy_rcmc` reproduces MultiOptPy's numbers |
| `--time-rule eigen\|gershgorin\|diag` | reference time of a contraction step (default `gershgorin`, as MultiOptPy) |
| `--times T1 T2 ...`, `--time-grid TMIN TMAX N` | several output times (logarithmic grid) |
| `--initial ID:FRACTION ...` | initial population over several nodes |
| `--rate-theory NAME`, `--corrections NAME ...` | rate theory and corrections |
| `--kappa VALUE` | transmission coefficient for all steps |
| `--clamp consistent\|legacy\|none` | treatment of saddle points below a minimum (1.4, item 5) |
| `--annotations FILE` | per-node and per-bottleneck data: outcomes and fractions, kinds, couplings, factors, uncertainties, concentrations |
| `--model NAME` | `canonical` (default), `egme`, `mass_action` |
| `--pressure P`, `--bath SPEC`, `--concentration SPECIES=VALUE ...` | conditions |
| `--lump-conformers` | report species-level results for nodes that share a bond graph |
| `--sensitivity [OBSERVABLE]` | degrees of rate control |
| `--uncertainty SIGMA`, `--samples N`, `--seed S` | sampling of energy errors |
| `--validate [TOL]` | compare with the direct solution and report the error |
| `--priority NAME` | print the exploration ranking under a priority rule |
| `--config FILE`, `--dump-config` | settings file (defaults < file < command line), as for every command |

Settings file: one `kinetics` object with the same keys (`method`, `time_rule`, `times`,
`initial`, `rate_theory`, `corrections`, `model`, `conditions`, `sensitivity`,
`uncertainty`, `validate`), usable verbatim inside the mapper settings, where
`use_rcmc`, `rcmc_temperature_K`, `rcmc_reaction_time_s`, `rcmc_start_node_id` keep
working and `--use_rcmc` selects the priority `population`.

Open decision: the default of `--method`. `propagate` is recommended (it equals the
published snapshot when the time scales are separated and is closer to the exact solution
otherwise); `rcmc` is the published method; `legacy_rcmc` is only for comparison.

### 5.5 Outputs

Written to the run directory by `workflow/rcmc.rs`, and shown on screen:

- `rcmc_K_contracted.csv`: same layout as MultiOptPy (matrix and population table), with
  the header naming which matrix it is; `rcmc_superstates.json`: members of each
  superstate with their weights (`B`), internal composition (`C`), lumped generator `G`,
  lifetimes;
- `populations.csv` (node by time), `rates.csv` (per bottleneck: `ln k` both ways, each
  factor, diagnostics), `diagnostics.json` (3.5), `sensitivity.csv`, `uncertainty.csv`;
- `populations.svg` (population against log time) through the built-in plotting;
- `mopt.log`, `settings.json` like every command; restart is trivial (the job is
  deterministic and fast) except for trajectory-based refinements, which use the journal.

Python: `mopt_rs.kinetics` exposes the network, rate theories, corrections, generator,
`contract`, the solvers, and the priorities as standalone objects (for instance
`contract(K, eliminate=...)` on a NumPy matrix).

### 5.6 What has to be written in the project

Only `faer`, `rayon`, `libm`, `pyo3` are available.

| Piece | Uses | Notes |
|---|---|---|
| log-sum-exp family, log-domain scalar | `libm` | `math/stable.rs`, `math/logreal.rs` |
| sparse generator, subtraction-free elimination | - | `kinetics/contract.rs`; lazy greedy selection for large networks |
| matrix exponential (scaling and squaring, Pade 13) | `faer` LU and products | `math/expm.rs` |
| stiff integrators (Rosenbrock 4(3), TR-BDF2), explicit RK for tests | `faer` LU | `math/ode.rs` |
| symmetric eigen-decomposition of the symmetrized generator | `faer` | exists (`math/linalg.rs`) |
| Gauss-Legendre and Gauss-Laguerre nodes | `faer` (Golub-Welsch) or Newton | `math/quadrature.rs` |
| scaled modified Bessel functions I0, I1 (hindered rotor), Airy Ai (weak coupling) | `libm` | `math/special.rs`; `erf`, `erfc`, `lgamma` come from `libm` |
| direct count of states (Beyer-Swinehart, Stein-Rabinovitch) | - | `kinetics/rrkm.rs` |
| Gillespie sampler | `core_infra::rng` | `kinetics/solve/kmc.rs` |
| parallel sampling for uncertainty | `rayon` | `kinetics/uncertainty.rs` |

---

## 6. Verification

Every piece has a test with a known answer. Tolerances are absolute unless stated.

| # | Piece | Test | Known answer | Tolerance |
|---|---|---|---|---|
| V1 | generator assembly | two minima, one saddle point | `k = (k_B T/h) exp(-dE/RT)`; `K pi = 0`; detailed-balance residual | 1e-13 relative |
| V2 | clamp | saddle point below the higher minimum | `consistent`: `k_uv/k_vu = exp(-(E_v - E_u)/RT)`; `legacy`: MultiOptPy's values | 1e-13 relative |
| V3 | `expm`, `ode`, `spectral` | A <-> B | `p_A(t) = pi_A + (p_A(0) - pi_A) exp(-(k_f + k_r) t)` | 1e-12 / 1e-8 / 1e-10 |
| V4 | same | A -> B -> C; A <-> B -> C | Bateman solution; eigenvalues of the 2x2 block; `k_eff = k_2 K / (1 + K)` in the pre-equilibrium limit | 1e-10 |
| V5 | elimination | energies spanning 500 `RT`, random sparse network | stationary distribution from full elimination equals `exp(-G/RT)/Z` | 1e-12 relative per state |
| V6 | contraction identities | random generators with and without detailed balance | `L K = [D 0]`; `1^T G = 0`; conservation after unfolding | 1e-12 |
| V7 | `rcmc` | random detailed-balanced networks | `G` of closure A equals the superstate matrix of the original recursion; `q = V p` of ref [4] | 1e-12 relative |
| V8 | `rcmc`, `propagate` | the three models of section 1.4 | values of the table | 1e-3 |
| V9 | `propagate` against `expm` | four-state model with tunable gap; random networks | error `<= max(eps_hat, 1e-10)`; `<= 1e-8` when `eps_hat <= 1e-9` | as stated |
| V10 | `legacy_rcmc` | networks in `tests/golden` generated with MultiOptPy | MultiOptPy's populations and `rcmc_K_contracted.csv` | 1e-9 relative |
| V11 | `wigner`, `eckart`, `skodje_truhlar` | Eckart barrier | `1 + u^2/24` for small `u`; parabolic limit `(u/2)/sin(u/2)` for a high barrier; transmission probability against a Numerov solution of the one-dimensional Schroedinger equation; symmetry of the factor referred to the saddle point | 1e-6 / 1e-4 |
| V12 | `kramers`, `collins_kimball` | limits | `kappa -> 1` and `omega_b/gamma`; `k -> k_act` and `k_D` | 1e-12 |
| V13 | `tst`, quasi-RRHO, symmetry | harmonic oscillator, rigid rotor, known symmetry numbers | closed-form partition functions; `G` against the thermochemistry module | 1e-10 |
| V14 | conformer lumping | `n` degenerate conformers in fast exchange | rate constant times `n` (or divided), Curtin-Hammett ratio `exp(-ddG/RT)` | 1e-8 |
| V15 | `rrkm` | few harmonic oscillators | direct count against explicit enumeration; Boltzmann average of `k(E)` equals the canonical rate constant | exact / 1e-6 relative |
| V16 | `egme` fall-off | one well with a sink, strong collisions | `integral of omega k(E) f(E)/(omega + k(E)) dE`; two-level model: Lindemann form `k_inf/(1 + k_inf/(k_0 [M]))`; `omega -> infinity`: canonical; `omega -> 0`: proportional to `omega` | 1e-8 relative |
| V17 | `egme` two wells | model of section 3.7 | eigenvalue rate constants; table of 3.7; deviation decreasing with the gap | 1e-3 at gap 1e4 |
| V18 | bottleneck outcomes | split weight | detailed balance of the generator; `k(R->P1)/k(R->P2) = phi_1/phi_2` | 1e-13 |
| V19 | bifurcation | model surface `V = f(x) + k(x) y^2/2 + c y^4/4` with `k` changing sign | VRI at the root of `k(x)`; symmetric surface: fractions 0.5 within three binomial standard errors; asymmetric: converged trajectory reference | exact location 1e-6; statistical |
| V20 | `landau_zener`, `weak_coupling`, `marcus` | limits | `p_sh -> 1` for strong coupling; thermal weak-coupling closed form of 2.5; Marcus normal and inverted region against the existing EDEEL tests | 1e-8 relative |
| V21 | `mass_action`, `ode` | A + B -> C; Robertson problem; Michaelis-Menten | closed-form second-order solution; published reference values; quasi-steady-state criterion of ref [82]; conserved totals | 1e-8 / 1e-6 relative / 1e-10 |
| V22 | pseudo-first-order windows | A + B -> C with B in excess and with B = A(0) | exponential decay; convergence to the second-order solution with the window tolerance | first order in the tolerance |
| V23 | `kmc` | two- and three-state models | master-equation populations within three standard errors | statistical |
| V24 | `steady` | unicyclic driven network | closed-form steady-state flux; sum of degrees of rate control equal to 1 | 1e-10 |
| V25 | sensitivities | random networks | finite differences; block-exponential identity | 1e-6 relative |
| V26 | memory | generator with a fast block | `D(s) = D - s (M - I) + O(s^2)`; semi-Markov chain with gamma-distributed residence times: long-time populations from `K = P / tau` | 1e-5 relative; statistical |
| V27 | priorities | golden network; toy network with a hidden exit | `population` ranking equals MultiOptPy's when the methods agree; `discovery_flux` finds the exit in fewer trials; invariance under relabelling of nodes | exact |
| V28 | command line | MultiOptPy's command lines | same options accepted; same files written | - |
| V29 | contraction of mass action (3.8.1) | first-order network; `A + B <-> X -> P` over five decades of separation; strong association; `2 A <-> D -> P`; enzyme; catalytic cycle with a second-order side reaction; `2 A + B <-> T -> P` | blocks equal to RCMC's for first-order networks; second-order convergence in the separation; agreement with the stiff integrator; conserved totals | 1e-12 (blocks, totals); the errors of the table in 3.8.1 |

---

## 7. Staged plan

The state of each stage as of 2026-11-20 is given after it; what is built is described in
[kinetics.md](kinetics.md). The names that are not built are refused by name with what
they would need (`kinetics::rate`, `kinetics::model`, `kinetics::solve`).

Stage 1: the first working `run_rcmc`.

1. Network input (MultiOptPy's file), `Conditions`, `eyring` with electronic or Gibbs
   energies, generator in flux form with log-space assembly and the consistent clamp.
2. Subtraction-free elimination; `rcmc` (published Type A, three time rules);
   `legacy_rcmc` reproducing MultiOptPy for the golden comparison; `propagate`; `expm` as
   reference; diagnostics; `--validate`.
3. `workflow/rcmc.rs`, `cli/run_rcmc.rs` with MultiOptPy's options and outputs,
   `--method`, `--times`, `--initial`; Python access.
4. `ExplorationPriority` with the adapter for the existing rules and the priorities
   `population`, `max_population`, `residence`; `--use_rcmc` of the mapper.
5. Tests V1-V10, V27, V28; the differences of section 1.4 recorded in `migration_workflows.md`.

State: built.

Stage 2: established corrections that need no new electronic-structure data.

1. `tst` with quasi-RRHO, symmetry numbers, standard states; `wigner`, `bell` (the infinite
   and the truncated parabola), `eckart`, `skodje_truhlar`; `kappa`; `collins_kimball`;
   pseudo-first-order concentrations.
2. Conformer lumping and species-level reports; saddle-point conformers of the mapper.
3. Sensitivities (degree of rate control), uncertainty sampling, `spectral` diagnostics,
   `steady` with turnover frequency and the energetic-span report.
4. `discovery_flux`; refinement tasks ranked by sensitivity times uncertainty.
5. Tests V11-V14, V24, V25.

State: built, the refinement tasks ranked by sensitivity times uncertainty included
(`run_kinetics --refine`, `kinetics::refine`; in the mapper's queue they are not yet, which
needs the task kinds of the mapper); `collins_kimball` is both the encounter model of the
barrierless steps and the correction of the bimolecular steps (`--viscosity`).

Stage 3: beyond TST where another module supplies the data.

1. Bottlenecks with several outcomes; VRI detection next to the IRC; fractions from the
   user and from trajectories started at the saddle point (needs MD); the priority rules of
   3.11 for bifurcations.
2. Non-adiabatic steps: `landau_zener` at MECPs from `pes/combine`, `marcus` from the EDEEL
   quantities; MECP-search tasks in the mapper.
3. Hot intermediates as additional states (`H` diagnostic, two-grain model); `rrkm`.
4. `mass_action` with the Rosenbrock integrator and pseudo-first-order windows; its
   contraction for reactions of any order (3.8.1) as the solver `contract`, validated
   against the integrator; `kmc`.
5. `variational` (canonical VTST along the IRC).
6. Tests V15, V18-V23, V29.

State: built: bottlenecks with several outcomes (TS–TS links, higher-order saddles, the
equal split; `mapper_design.md`; measured `outcomes` in the annotations), fractions and the
transmission coefficient from trajectories started at the saddle point
(`workflow::kinetics_dynamics`, `kinetics::branching`; V19 on a model surface),
`landau_zener` and `zhu_nakamura` at crossing points (with the crossing searches of the
mapper), the Marcus and Zhu-Nakamura rates of electron transfer (`etmain`,
[electron_transfer.md](electron_transfer.md)), hot intermediates with the `H` diagnostic and
the two-grain model (`two_grain`), `rrkm` (V15), `mass_action` with the Rosenbrock integrator,
`kmc` (V23), `variational`, and the contraction of mass action of any reaction order, the
solver `contract` (3.8.1, V29: the blocks of RCMC for a first-order network, second-order
convergence in the separation, dimerization, enzyme, third order), the pseudo-first-order
windows of `mass_action` (the solver `windows`, V22), and the automatic judgment of the order of
every step (`--model auto`: pseudo-first order where the partner changes by less than a
tolerance, `canonical`; otherwise `mass_action`). Planned: the re-examination of the
eliminated set of `contract` along the trajectory; the VRI test is that of the IRC and the
mapper.

Stage 4: research-grade, optional.

1. Full `egme` with collision models, phenomenological rate constants by contraction and
   by eigenvalues (V16, V17).
2. `semi_markov` and kernel contraction; rate matrices from MD (Markov state models)
   (V26).
3. `grote_hynes` and reactive-flux transmission coefficients from MD; small-curvature
   tunnelling; `weak_coupling`.
4. A geometric model of branching fractions, validated against trajectories on model
   surfaces before any default use.

State: built: `egme` with exponential-down and strong collisions, Lennard-Jones collision
frequencies, and the phenomenological rate constants by contraction (closures A and B) and by
eigenpairs (V16 with the strong-collision secular equation, V17 with the bracketing and the
convergence with the gap); `semi_markov` with gamma-distributed residence times, its Markov
limit, the first-order memory correction, the kernel contraction (V26), and its residence
times and Markov state model from a trajectory (`--md-trajectory`); `kramers`, `grote_hynes`
(exponential and tabulated kernels; the kernel from dynamics with the reaction coordinate
held, `workflow::kinetics_dynamics::friction_kernel`), `reactive_flux`, `small_curvature`
(with the path of `freqmain -path`), `weak_coupling` (V20). Planned: the geometric model of
branching fractions (Lee and Goodman, J. Am. Chem. Soc. 2020, 142, 9210; Org. Biomol. Chem.
2021, 19, 3940), to be validated against the trajectories of model surfaces before any use;
until then the fractions come from the user or from trajectories.

Planned after these stages (in the order of their use): the re-examination of the eliminated
set of `contract` when the composition changes; `rcmc_b`; the geometric branching model; the
refinement tasks in the mapper's queue (3.11); a command-line route for the trajectories from a
saddle point. They are listed in [kinetics.md](kinetics.md#planned).

Recommended against:

- Variable-reaction-coordinate TST, instanton and ring-polymer rate theory, semiclassical
  TST with anharmonic constants: each needs its own sampling or path-optimization
  machinery and electronic-structure data far beyond what the mapper produces. Barrierless
  steps get a capture or diffusion bound and a flag instead.
- Surface-hopping and conical-intersection dynamics inside the kinetics module: the
  network records MECI / MECP bottlenecks and accepts user fractions; the dynamics belongs
  to a dedicated program.
- A statistical or trajectory treatment of roaming: no diagnostic that the mapper could
  compute reliably, and a global surface is required.
- Two-dimensional (energy and angular momentum) master equations and automated fitting of
  energy-transfer parameters.
- Closure B (`M^-1`) as the default: it is the better projection but its entries are not
  sign-definite, so the subtraction-free guarantee is lost; it stays available.
- Multi-precision arithmetic: the elimination does not need it, and the log-domain scalar
  covers underflow.
- Kinetic Monte Carlo as a general solver: useless for stiff networks; kept for
  validation and semi-Markov tests.
- Correcting rate constants by fitted or learned transmission factors without a
  diagnostic that says when they apply.

---

## 8. References

RCMC and kinetics-based navigation

1. Sumiya, Nagahata, Komatsuzaki, Taketsugu, Maeda, J. Phys. Chem. A 2015, 119, 11641. doi:10.1021/acs.jpca.5b09447
2. Sumiya, Taketsugu, Maeda, J. Comput. Chem. 2017, 38, 101. doi:10.1002/jcc.24526
3. Sumiya, Maeda, Chem. Lett. 2020, 49, 553. doi:10.1246/cl.200092
4. Iwata, Oki, Sakaue, SIAM J. Sci. Comput. 2026, 48, A261. doi:10.1137/24M1668639 (arXiv:2312.05470, doi:10.48550/arXiv.2312.05470)
5. Hemmi, Iwata, Oki, arXiv:2409.13128; ICIAM2023 Springer Series 2026, 9. doi:10.1007/978-981-95-1446-5_2
6. Maeda, Harabuchi, WIREs Comput. Mol. Sci. 2021, 11, e1538. doi:10.1002/wcms.1538
7. Sumiya, Harabuchi, Nagata, Maeda, JACS Au 2022, 2, 1181. doi:10.1021/jacsau.2c00157

Transition state theory and its refinements

8. Eyring, J. Chem. Phys. 1935, 3, 107. doi:10.1063/1.1749604; Wigner, Trans. Faraday Soc. 1938, 34, 29. doi:10.1039/TF9383400029
9. Truhlar, Garrett, Klippenstein, J. Phys. Chem. 1996, 100, 12771. doi:10.1021/jp953748q; Fernandez-Ramos, Miller, Klippenstein, Truhlar, Chem. Rev. 2006, 106, 4518. doi:10.1021/cr050205w
10. Bao, Truhlar, Chem. Soc. Rev. 2017, 46, 7548. doi:10.1039/C7CS00602K; Truhlar, Garrett, Acc. Chem. Res. 1980, 13, 440. doi:10.1021/ar50156a002
11. Georgievskii, Klippenstein, J. Chem. Phys. 2003, 118, 5442. doi:10.1063/1.1539035
12. Wigner, Z. Phys. Chem. B 1932, 19, 203. doi:10.1515/zpch-1932-1920
13. Eckart, Phys. Rev. 1930, 35, 1303. doi:10.1103/PhysRev.35.1303; Johnston, Heicklen, J. Phys. Chem. 1962, 66, 532. doi:10.1021/j100809a040
14. Skodje, Truhlar, J. Phys. Chem. 1981, 85, 624. doi:10.1021/j150606a003
15. Liu, Lynch, Truong, Lu, Truhlar, Garrett, J. Am. Chem. Soc. 1993, 115, 2408. doi:10.1021/ja00059a041
16. Grimme, Chem. Eur. J. 2012, 18, 9955. doi:10.1002/chem.201200497
17. Li, Gomes, Sharada, Bell, Head-Gordon, J. Phys. Chem. C 2015, 119, 1840. doi:10.1021/jp509921r
18. Pitzer, Gwinn, J. Chem. Phys. 1942, 10, 428. doi:10.1063/1.1723744
19. Zheng, Yu, Papajak, Alecu, Mielke, Truhlar, Phys. Chem. Chem. Phys. 2011, 13, 10885. doi:10.1039/C0CP02644A
20. Yu, Zheng, Truhlar, Chem. Sci. 2011, 2, 2199. doi:10.1039/C1SC00225B
21. Seeman, Chem. Rev. 1983, 83, 83. doi:10.1021/cr00054a001
22. Fernandez-Ramos, Ellingson, Meana-Paneda, Marques, Truhlar, Theor. Chem. Acc. 2007, 118, 813. doi:10.1007/s00214-007-0328-0
23. Kozuch, Shaik, Acc. Chem. Res. 2011, 44, 101. doi:10.1021/ar1000956

Recrossing, friction, diffusion

24. Chandler, J. Chem. Phys. 1978, 68, 2959. doi:10.1063/1.436049
25. Bennett, ACS Symp. Ser. 1977, 46, 63. doi:10.1021/bk-1977-0046.ch004
26. Kramers, Physica 1940, 7, 284. doi:10.1016/S0031-8914(40)90098-2
27. Grote, Hynes, J. Chem. Phys. 1980, 73, 2715. doi:10.1063/1.440485
28. Hanggi, Talkner, Borkovec, Rev. Mod. Phys. 1990, 62, 251. doi:10.1103/RevModPhys.62.251
29. Smoluchowski, Z. Phys. Chem. 1917, 92, 129. doi:10.1515/zpch-1918-9209
30. Collins, Kimball, J. Colloid Sci. 1949, 4, 425. doi:10.1016/0095-8522(49)90023-9

Non-statistical behaviour

31. Valtazanos, Ruedenberg, Theor. Chim. Acta 1986, 69, 281. doi:10.1007/BF00527705; Quapp, J. Mol. Struct. 2004, 695-696, 95. doi:10.1016/j.molstruc.2003.10.034
32. Ess, Wheeler, Iafe, Xu, Celebi-Olcum, Houk, Angew. Chem. Int. Ed. 2008, 47, 7592. doi:10.1002/anie.200800918; Hare, Tantillo, Pure Appl. Chem. 2017, 89, 679. doi:10.1515/pac-2017-0104
33. Maeda, Harabuchi, Ono, Taketsugu, Morokuma, Int. J. Quantum Chem. 2015, 115, 258. doi:10.1002/qua.24757
34. Yang, Dong, Yu, Yu, Li, Jamieson, Houk, J. Am. Chem. Soc. 2018, 140, 3061. doi:10.1021/jacs.7b13562
35. Lee, Goodman, J. Am. Chem. Soc. 2020, 142, 9210. doi:10.1021/jacs.9b13449
36. Lee, Goodman, Org. Biomol. Chem. 2021, 19, 3940. doi:10.1039/D1OB00234A
37. Carpenter, J. Am. Chem. Soc. 1995, 117, 6336. doi:10.1021/ja00128a024; Zheng, Papajak, Truhlar, J. Am. Chem. Soc. 2009, 131, 15754. doi:10.1021/ja904405v
38. Carpenter, Annu. Rev. Phys. Chem. 2005, 56, 57. doi:10.1146/annurev.physchem.56.092503.141240; Rehbein, Carpenter, Phys. Chem. Chem. Phys. 2011, 13, 20906. doi:10.1039/C1CP22565K; Carpenter, Chem. Rev. 2013, 113, 7265. doi:10.1021/cr300511u
39. Glowacki, Liang, Marsden, Harvey, Pilling, J. Am. Chem. Soc. 2010, 132, 13621. doi:10.1021/ja105100f
40. Oyola, Singleton, J. Am. Chem. Soc. 2009, 131, 3130. doi:10.1021/ja807666d; Singleton et al., J. Am. Chem. Soc. 2003, 125, 1319. doi:10.1021/ja027225p
41. Lourderaj, Hase, J. Phys. Chem. A 2009, 113, 2236. doi:10.1021/jp806659f; Sun, Song, Hase, Science 2002, 296, 875. doi:10.1126/science.1068053
42. Goldman, Glowacki, Carpenter, J. Am. Chem. Soc. 2011, 133, 5312. doi:10.1021/ja1095717; Kurouchi, Singleton, Nat. Chem. 2018, 10, 237. doi:10.1038/nchem.2907
43. Peslherbe, Wang, Hase, Adv. Chem. Phys. 1999, 105, 171. doi:10.1002/9780470141649.ch6
44. Lourderaj, Park, Hase, Int. Rev. Phys. Chem. 2008, 27, 361. doi:10.1080/01442350802045446
45. Pratihar, Ma, Homayoon, Barnes, Hase, J. Am. Chem. Soc. 2017, 139, 3570. doi:10.1021/jacs.6b12017
46. Townsend et al., Science 2004, 306, 1158. doi:10.1126/science.1104386; Bowman, Shepler, Annu. Rev. Phys. Chem. 2011, 62, 531. doi:10.1146/annurev-physchem-032210-103518
47. Klippenstein, Georgievskii, Harding, J. Phys. Chem. A 2011, 115, 14370. doi:10.1021/jp208347j

RRKM theory and master equations

48. Marcus, J. Chem. Phys. 1952, 20, 359. doi:10.1063/1.1700424
49. Beyer, Swinehart, Commun. ACM 1973, 16, 379. doi:10.1145/362248.362275; Stein, Rabinovitch, J. Chem. Phys. 1973, 58, 2438. doi:10.1063/1.1679522
50. Miller, J. Am. Chem. Soc. 1979, 101, 6810. doi:10.1021/ja00517a004
51. Bartis, Widom, J. Chem. Phys. 1974, 60, 3474. doi:10.1063/1.1681562
52. Klippenstein, Miller, J. Phys. Chem. A 2002, 106, 9267. doi:10.1021/jp021175t; Miller, Klippenstein, J. Phys. Chem. A 2006, 110, 10528. doi:10.1021/jp062693x
53. Georgievskii, Miller, Burke, Klippenstein, J. Phys. Chem. A 2013, 117, 12146. doi:10.1021/jp4060704; Glowacki, Liang, Morley, Pilling, Robertson, J. Phys. Chem. A 2012, 116, 9545. doi:10.1021/jp3051033
54. Troe, J. Chem. Phys. 1977, 66, 4745. doi:10.1063/1.433837; Pilling, Robertson, Annu. Rev. Phys. Chem. 2003, 54, 245. doi:10.1146/annurev.physchem.54.011002.103822; Allen, Goldsmith, Green, Phys. Chem. Chem. Phys. 2012, 14, 1131. doi:10.1039/C1CP22765C
55. Miller, Hernandez, Handy, Jayatilaka, Willetts, Chem. Phys. Lett. 1990, 172, 62. doi:10.1016/0009-2614(90)87217-F
56. Richardson, Int. Rev. Phys. Chem. 2018, 37, 171. doi:10.1080/0144235X.2018.1472353
57. Craig, Manolopoulos, J. Chem. Phys. 2005, 122, 084106. doi:10.1063/1.1850093
58. Frankcombe, Smith, Theor. Chem. Acc. 2009, 124, 303. doi:10.1007/s00214-009-0623-z

Non-adiabatic steps

59. Harvey, Phys. Chem. Chem. Phys. 2007, 9, 331. doi:10.1039/B614390C
60. Harvey, Aschi, Phys. Chem. Chem. Phys. 1999, 1, 5555. doi:10.1039/A907723E
61. Lykhin, Kaliakin, dePolo, Kuzubov, Varganov, Int. J. Quantum Chem. 2016, 116, 750. doi:10.1002/qua.25124
62. Zener, Proc. R. Soc. A 1932, 137, 696. doi:10.1098/rspa.1932.0165
63. Marcus, J. Chem. Phys. 1956, 24, 966. doi:10.1063/1.1742723
64. Marcus, Sutin, Biochim. Biophys. Acta 1985, 811, 265. doi:10.1016/0304-4173(85)90014-X
65. Jortner, J. Chem. Phys. 1976, 64, 4860. doi:10.1063/1.432142
66. Mutsuji, Saita, Maeda, RSC Adv. 2023, 13, 32097. doi:10.1039/D3RA05784D
67. Tully, J. Chem. Phys. 1990, 93, 1061. doi:10.1063/1.459170

Memory, Markov models, elimination of states

68. Zwanzig, Phys. Rev. 1961, 124, 983. doi:10.1103/PhysRev.124.983
69. Mori, Prog. Theor. Phys. 1965, 33, 423. doi:10.1143/PTP.33.423
70. Prinz et al., J. Chem. Phys. 2011, 134, 174105. doi:10.1063/1.3565032
71. Husic, Pande, J. Am. Chem. Soc. 2018, 140, 2386. doi:10.1021/jacs.7b12191
72. Cao, Montoya-Castillo, Wang, Markland, Huang, J. Chem. Phys. 2020, 153, 014105. doi:10.1063/5.0010787
73. Hummer, Szabo, J. Phys. Chem. B 2015, 119, 9029. doi:10.1021/jp508375q
74. Trygubenko, Wales, J. Chem. Phys. 2006, 124, 234110. doi:10.1063/1.2198806
75. Wales, J. Chem. Phys. 2009, 130, 204111. doi:10.1063/1.3133782; Mol. Phys. 2002, 100, 3285. doi:10.1080/00268970210162691
76. Kannan, Sharpe, Swinburne, Wales, J. Chem. Phys. 2020, 153, 244108. doi:10.1063/5.0025174; Swinburne, Wales, J. Chem. Theory Comput. 2020, 16, 2661. doi:10.1021/acs.jctc.9b01211
77. Meyer, SIAM Rev. 1989, 31, 240. doi:10.1137/1031050
78. Simon, Ando, Econometrica 1961, 29, 111. doi:10.2307/1909285
79. Grassmann, Taksar, Heyman, Oper. Res. 1985, 33, 1107. doi:10.1287/opre.33.5.1107
80. Nagahata, Kobayashi, Toda, Maeda, Taketsugu, Komatsuzaki, Proc. Natl. Acad. Sci. 2024, 121, e2317781121. doi:10.1073/pnas.2317781121

Non-linear kinetics, stochastic kinetics, open systems, exploration

81. Turanyi, Tomlin, Pilling, J. Phys. Chem. 1993, 97, 163. doi:10.1021/j100103a028
82. Segel, Slemrod, SIAM Rev. 1989, 31, 446. doi:10.1137/1031091
83. Turanyi, Tomlin, Analysis of Kinetic Reaction Mechanisms, Springer 2014. doi:10.1007/978-3-662-44562-4
84. Motagamwala, Dumesic, Chem. Rev. 2021, 121, 1049. doi:10.1021/acs.chemrev.0c00394
85. Besora, Maseras, WIREs Comput. Mol. Sci. 2018, 8, e1372. doi:10.1002/wcms.1372
86. Klippenstein, Pande, Truhlar, J. Am. Chem. Soc. 2014, 136, 528. doi:10.1021/ja408723a
87. Lam, Goussis, Int. J. Chem. Kinet. 1994, 26, 461. doi:10.1002/kin.550260408
88. Lam, Goussis, Symp. (Int.) Combust. 1989, 22, 931. doi:10.1016/S0082-0784(89)80102-X
89. Maas, Pope, Combust. Flame 1992, 88, 239. doi:10.1016/0010-2180(92)90034-M
90. Gillespie, J. Comput. Phys. 1976, 22, 403. doi:10.1016/0021-9991(76)90041-3
91. Gillespie, J. Phys. Chem. 1977, 81, 2340. doi:10.1021/j100540a008
92. Bortz, Kalos, Lebowitz, J. Comput. Phys. 1975, 17, 10. doi:10.1016/0021-9991(75)90060-1
93. Schnakenberg, Rev. Mod. Phys. 1976, 48, 571. doi:10.1103/RevModPhys.48.571
94. Horn, Jackson, Arch. Ration. Mech. Anal. 1972, 47, 81. doi:10.1007/BF00251225
95. Norskov et al., J. Phys. Chem. B 2004, 108, 17886. doi:10.1021/jp047349j
96. Chidsey, Science 1991, 251, 919. doi:10.1126/science.251.4996.919
97. Susnow, Dean, Green, Peczak, Broadbelt, J. Phys. Chem. A 1997, 101, 3731. doi:10.1021/jp9637690
98. Gao, Allen, Green, West, Comput. Phys. Commun. 2016, 203, 212. doi:10.1016/j.cpc.2016.02.013
99. Bensberg, Reiher, Isr. J. Chem. 2023, 63, e202200123. doi:10.1002/ijch.202200123
100. Bensberg, Reiher, J. Phys. Chem. A 2024, 128, 4532. doi:10.1021/acs.jpca.3c08386

Sensitivity and uncertainty

101. Campbell, ACS Catal. 2017, 7, 2770. doi:10.1021/acscatal.7b00115
102. Stegelmann, Andreasen, Campbell, J. Am. Chem. Soc. 2009, 131, 8077. doi:10.1021/ja9000097
103. Harabuchi, Yokoyama, Matsuoka, Oki, Iwata, Maeda, J. Phys. Chem. A 2024, 128, 2883. doi:10.1021/acs.jpca.4c00204
104. Proppe, Husch, Simm, Reiher, Faraday Discuss. 2016, 195, 497. doi:10.1039/C6FD00144K
105. Proppe, Reiher, J. Chem. Theory Comput. 2019, 15, 357. doi:10.1021/acs.jctc.8b00310
106. Simm, Reiher, J. Chem. Theory Comput. 2018, 14, 5238. doi:10.1021/acs.jctc.8b00504
107. Sutton, Guo, Katsoulakis, Vlachos, Nat. Chem. 2016, 8, 331. doi:10.1038/nchem.2454
108. Medford et al., Science 2014, 345, 197. doi:10.1126/science.1253486

Numerical methods

109. Higham, SIAM J. Matrix Anal. Appl. 2005, 26, 1179. doi:10.1137/04061101X; Moler, Van Loan, SIAM Rev. 2003, 45, 3. doi:10.1137/S00361445024180
110. Al-Mohy, Higham, SIAM J. Sci. Comput. 2011, 33, 488. doi:10.1137/100788860
111. Hairer, Wanner, Solving Ordinary Differential Equations II, Springer 1996. doi:10.1007/978-3-642-05221-7
112. Shampine, Reichelt, SIAM J. Sci. Comput. 1997, 18, 1. doi:10.1137/S1064827594276424
