# Workflows from Python: AutoTS step by step

`run_autots` runs four steps: an artificial-force push, a chain relaxation, an optimization
to a saddle point, and an IRC. It decides on its own what goes from one step to the next.
This page builds the same workflow from the Python building blocks: `Optimizer` for
`optmain`, `relax_band` for `nebmain`, `vibrational_analysis` for `freqmain`, and
`irc_path` for `ircmain`. Each of those decisions is then a few lines of your own script
that you can change.

The script is `examples/python_autots/autots_by_hand.py`, with its input `SN2.xyz` next to
it (fluoromethane and hydroxide, charge -1):

```bash
python examples/python_autots/autots_by_hand.py run_by_hand
```

It writes the push, the relaxed chain, the transition state, and the optimized ends of the
IRC as xyz files into `run_by_hand/`, and the numbers into `results.json`. If a step does
not reach its criteria, or gives something the next step cannot use, the script stops with
a message that names the step and exits with status 1. Every code block on this page is
part of the script; `tests/test_python_workflows.py` checks this, runs the script, and
compares its numbers with those of `run_autots` on the same input.

**When to use the command instead.** `run_autots INPUT -cfg config.json --run_step4` (from
Python: `mopt_rs.cli.run_counted("run_autots", [...])`) runs the same steps. It also writes
the report page `autots_report.html`, the summary, journals for `--restart`, and the
`workflow` block. It runs on every surface the commands support (ONIOM with
microiterations, periodic cells, two-state model functions), see "Other surfaces" below.
Use the command when its decisions suit you. Write the steps yourself when you want to
change one of them: which candidates to try, what to do with a candidate that fails, or
what counts as the reaction.

## The engine and the reaction

The example uses the built-in GFN1-xTB (`NativeCalculator("gfn1xtb")`), so it needs no
optional software. The project prefers GFN1-xTB for new work on organic reactions. Here
it also converged at every step and every candidate of both runs. The reaction is the
SN2 substitution HO⁻ + CH3F → CH3OH + F⁻. It is a push-type reaction, so a straight AFIR
push between C (atom 1) and O (atom 6) drives it.

## The surface and its counts

```python
class CountedEngine:
    """A built-in calculator that counts what it is asked for."""

    def __init__(self, name: str, symbols: list[str], charge: int, multiplicity: int) -> None:
        self.calculator = NativeCalculator(name)
        self.symbols, self.charge, self.multiplicity = symbols, charge, multiplicity
        self.gradients = 0
        self.hessians = 0
```

All steps use one calculator object. Its two counters are the counterpart of the line
`Evaluations (the whole run): ... gradient calculation(s), ... exact Hessian
calculation(s)` at the end of `autots.log`. The optimizers call `evaluate(task)`; the
chain, the vibrational analysis, and the IRC call `at(x, need_hessian)`.

## Step 1: the push (`optmain -ma`)

```python
    push = Optimizer(
        engine.symbols,
        x0.ravel().tolist(),
        charge=engine.charge,
        multiplicity=engine.multiplicity,
        methods=["rsirfo_block_fsb"],
        model_hessian="fischerd3",
    )
    push.add_afir(gamma, a, b)
    _, _, frames = converged_point("step 1 (AFIR push)", push, engine)
    return frames
```

- **Handed on:** the structures the optimization evaluated, the first one being the input.
  This is the approximate path, the counterpart of `<stem>_step1_traj.xyz` of the command.
  The last structure is the pushed product. The script keeps it as the reference for the
  product in step 6.
- **What `run_autots` decides:** step 1 needs a bias potential that drives the reaction,
  and the command refuses to run without one. If the push stops at its iteration limit,
  the command still takes its trajectory as the path. The script is stricter: in
  `converged_point`, an optimization that did not converge ends the workflow.
- **To change it:** `add_afir(gamma, a, b, gamma_final=...)` ramps the force. `add_bias`
  takes any bias option of `optmain` (`"-kp"`, ...) with the arguments of the command line.

## Step 2: the chain (`nebmain`)

```python
    relaxed = relax_band(
        surface,
        [f.tolist() for f in frames],
        symbols=engine.symbols,
        spacing=spacing_angstrom * _core.ANGSTROM_TO_BOHR,
        max_iterations=iterations,
        tolerance=3e-4,
    )
```

- **Handed on:** the frames of the push become a chain of images 0.5 Angstrom apart along
  the path (`spacing`, the distance `nebmain` uses by default; `-nd`). The chain is relaxed
  on the surface alone: `surface` calls the engine, and the artificial force of step 1 is
  not part of it.
- **The candidates:**

```python
    inner = range(1, len(energies) - 1)
    maxima = [i for i in inner if energies[i] > energies[i - 1] and energies[i] > energies[i + 1]]
    if not maxima:
        maxima = [max(inner, key=lambda i: energies[i])]
    return sorted(maxima, key=lambda i: energies[i], reverse=True)[:top_n]
```

  This is the rule of `run_autots`: the inner images whose energy is above that of both
  neighbours. If there is none, the highest inner image is used. They are sorted highest
  first and cut to `top_n_candidates` (3).
- **What `run_autots` decides:** a chain that ends at its iteration limit still gives its
  maxima as candidates. The command takes them, and so does the script, which prints a
  line about it. The real test comes in steps 3 and 4.
- **The same as `nebmain`:** `relax_band` reads its settings with the parser of `nebmain`
  and sets the chain up as `nebmain` does, so without options it relaxes the chain of
  `nebmain` without options: quasi-Newton steps with model Hessians (`lindh2007d3`), the
  images put towards the predicted maxima every second iteration, the structures aligned.
  `options` takes any option of `nebmain` (`["-qsm"]`, `["-lup"]`, `["-ewbneb"]`,
  `["-modelhess", "fischerd3"]`, ...). The keyword arguments used here are the defaults of
  `nebmain` spelled out (`-nd 0.5`, `-ns 30`, `-neb_opt tolerance=3e-4`).

## Step 3: the saddle point (`optmain -order 1`)

```python
    ts = Optimizer(
        engine.symbols,
        guess.ravel().tolist(),
        charge=engine.charge,
        multiplicity=engine.multiplicity,
        methods=["rsirfo_block_bofill"],
        saddle_order=1,
        exact_hessian_interval=5,
        criteria="tight",
        max_trust_radius=0.2,
    )
```

- **Handed on:** the structure of a candidate image. The Hessian for the first saddle
  step is computed exactly at the first iteration (`exact_hessian_interval=5`, the `-fc 5`
  of the command's step 3). It sets the mode the optimizer climbs. `Optimizer` can also
  start from a Hessian that is there already (`initial_hessian=`: an array, a file as
  `optmain -ih` reads it, or a model Hessian name); the script computes it instead, as the
  command does.
- **What `run_autots` decides:** every candidate is tried. A candidate that fails or does
  not converge is reported, and the next one is tried. A transition state found twice
  (energy within 1e-6 Hartree and the same structure) goes on once. The script does the
  same. It uses `compare_structures` for "the same structure": same bonds and an RMSD
  below 0.01 Bohr after matching. The command uses no interatomic distance changing by
  0.01 Bohr or more.

## Step 4: the vibrational analysis (`freqmain`, `-freq`)

```python
    found = engine.at(x, need_hessian=True)
    analysis = _core.vibrational_analysis(
        engine.symbols,
        x.ravel().tolist(),
        np.ravel(found.hessian).tolist(),
        found.energy,
        engine.charge,
        engine.multiplicity,
    )
    modes = analysis["modes"]
    return [
        w
        for w, k in zip(modes["wavenumbers"], modes["curvatures"], strict=True)
        if k < -_core.IMAGINARY_CURVATURE
    ]
```

```python
    imaginary = imaginary_wavenumbers(engine, x)
    if len(imaginary) != 1:
        listed = ", ".join(f"{w:.1f}" for w in imaginary)
        raise WorkflowError(
            f"{name}: {len(imaginary)} imaginary modes ({listed} cm^-1); "
            "a transition state has exactly one"
        )
    return imaginary[0]
```

- **The rule:** the exact Hessian at the converged structure is analysed with translations
  and rotations projected out. A mode is imaginary when its eigenvalue of the mass-weighted
  Hessian is below `-IMAGINARY_CURVATURE` (1e-8 Hartree / (Bohr^2 amu), about 0.5 cm^-1).
  This is the library's one criterion, the same as for the mode kill, the mapper, and the
  start of an IRC. It only excludes the numerically flat projected motions (their
  wavenumbers print as -0.0). A structure without exactly one imaginary mode is refused,
  and the message lists the modes.
- **What `run_autots` decides:** the command counts the imaginary modes of `optmain -freq`
  by the same criterion (`vib::imaginary_wavenumbers`), so the command and the script accept
  the same transition states. The criterion is not loosened: a mode is imaginary by it or it
  is not.

## Step 5: the IRC and its ends (`ircmain`, `optmain -irc`)

```python
    path = irc_path(
        energy_and_gradient,
        x_ts.ravel(),
        symbols=engine.symbols,
        step_size=0.5,
        max_steps=200,
        method="lqa",
        exact_hessian_interval=10,
        hessian=hessian,
    )
```

- **Handed on:** the transition state. `irc_path` takes the transition vector (the
  imaginary mode) from the Hessian it asks for at the start; no direction needs to be
  given. The step is in mass-weighted coordinates (amu^1/2 Bohr), as in `-irc 0.5 200 lqa`.
  The script checks that the IRC started from a transition state (`kind`). The last point
  of each side then goes to a minimum:

```python
    end = Optimizer(
        engine.symbols,
        x.ravel().tolist(),
        charge=engine.charge,
        multiplicity=engine.multiplicity,
        methods=["rsirfo_block_fsb"],
        exact_hessian_interval=10,
        criteria="tight",
        mode_kill=True,
    )
    x_min, energy, _ = converged_point(name, end, engine)
    imaginary = imaginary_wavenumbers(engine, x_min)
```

  These are the settings of the ends in step 4 of the command (`step4b_opt_method`,
  `-fc 10`, tight criteria), plus two additions: the mode kill (`mode_kill=True`, `optmain
  -modekill`), and the vibrational analysis of step 4, which must find no imaginary mode.
- **Why the additions.** The transition state found here has Cs symmetry to 1e-4 Bohr.
  Its IRC keeps the symmetry, and without the mode kill the reactant end converged at the
  Cs stationary point of the reactant complex. That point has one imaginary mode (-64.5
  cm^-1, the rotation of the hydroxide) and lies 0.053 kcal/mol above the minimum. The
  transition state of `run_autots` is the same saddle point (RMSD 3e-5 Bohr) but breaks
  the symmetry slightly (`point_group` finds C1 at a tolerance of 1e-4 Bohr and Cs at
  1e-3), and its IRC reached the minimum. With the mode kill, the script reaches the same
  minimum.
- **What `run_autots` decides:** both ends are optimized, however the sides ended. The
  command takes their energies even when the optimization stopped at its iteration limit
  or at a stationary point with an imaginary mode; the vibrational analysis of the ends is
  only run when `step4_settings` asks for it (`frequency_analysis`). The script stops
  instead.

## Step 6: the reaction and the report

```python
def bonds_alike(symbols: list[str], a: np.ndarray, b: np.ndarray) -> bool:
    """Whether two structures have the same bonds (their bond graphs agree up to a
    permutation of equivalent atoms)."""
    found = compare_structures(symbols, np.ravel(a).tolist(), np.ravel(b).tolist())
    return bool(found["same_bonds"])
```

```python
        reactant = [e for x, e in ends if bonds_alike(symbols, x, reactant_ref)]
        product = [e for x, e in ends if bonds_alike(symbols, x, product_ref)]
```

- **The rule:** a transition state is the reaction when one optimized end has the bonds of
  the input and the other has those of the pushed product. The comparison is made with
  `compare_structures` of `mopt_rs.orient`, which matches bond graphs up to equivalent
  atoms. `run_autots` reports the ends of every transition state and does not decide which
  one is the reaction.
- **The report:**

```python
        kcal = _core.HARTREE_TO_KCAL_MOL
        barrier = self.transition_state - self.reactant
        reaction = self.product - self.reactant
```

  The script reports the energies in Hartree, the barrier and the reaction energy in
  Hartree and in kcal/mol (with the CODATA 2018 factor of the module), the imaginary
  wavenumber, and the gradient and Hessian evaluations of the whole workflow.

## The numbers

The script and `run_autots` with `autots_config.json` (release build, GFN1-xTB):

| | the script | `run_autots --run_step4` |
|---|---|---|
| reactant complex (Hartree) | -14.271926817389 | -14.271926817390 |
| transition state (Hartree) | -14.269376098875 | -14.269376099061 |
| product complex (Hartree) | -14.308485555289 | -14.308485555289 |
| barrier from the reactant complex | 1.6006 kcal/mol | 1.6006 kcal/mol |
| reaction energy | -22.941 kcal/mol | -22.941 kcal/mol |
| imaginary wavenumber of the transition state | -404.4 cm^-1 | |
| chain | 11 images, 30 iterations (not converged) | 11 images, 30 iterations |
| candidates, transition states found | 1, 1 | 2, 2 |
| gradients, exact Hessians | 566, 30 | 819, 88 |
| time | 2 s | 6 s |

The energies agree to 2e-10 Hartree. The test allows 1e-6 Hartree; its comment gives the
reason. The chain of the script has one maximum, the image that leads to the transition
state of the reaction. The command finds a second transition state, the rotation of the
hydroxide in the reactant complex (-14.271842 Hartree), which joins the reactant complex
with itself.

## Constraints and bias potentials through the steps

- **Constraints.** `Optimizer(..., constraints=["bond", "2,3"])` holds a constraint that
  has no value at its value in the optimizer's own starting structure. If each step gets
  the same tokens, each step holds a different value: the push holds the value of the
  input, and the saddle search holds the value of its candidate. To keep the value of the
  input through all steps, measure it once at the input and give it explicitly:
  `constraints=["manual", "bond", "2,3", "1.0950"]` (Angstrom and degrees, as `-pc manual`
  takes them). `relax_band` and `irc_path` take the same constraints with the same
  spelling: `irc_path` resolves them once at its start and holds them at every point,
  `relax_band` holds in every image its own value of the first chain (explicit values in
  all images), as `nebmain` does; both to 1e-10, frozen atoms exactly. In `run_autots`,
  each step reads the constraints of its own settings block and resolves those without a
  value at its own input.
- **Bias potentials.** A static bias that belongs to the surface of every step goes into
  every step: `optimizer.add_bias("-kp", [...])` for the optimizers. For `relax_band` and
  `irc_path`, add `mopt_rs.evaluate_bias(option, args, symbols, x)` (energy, gradient, and
  Hessian when asked for) to what the callbacks return. A bias that only drives the push,
  like the AFIR of step 1, is dropped afterwards. In the script it is added only to the
  optimizer of step 1. The chain, the saddle search, and the IRC call the engine alone.
  `run_autots` does the same: `-ma` and `manual_AFIR` belong to `step1_settings` only.

## Other surfaces

- **ONIOM with microiterations.** `mopt_rs.OniomCalculator` is the ONIOM calculator of the
  commands (layers, link atoms, embedding, the settings of `-oniom_opt`).
  `Optimizer(..., calculator=oniom)` with `run(optimizer, oniom)` moves the core and
  relaxes the environment at every evaluation, as `optmain -oniom` does; `relax_band`
  takes it as its calculator, as `nebmain -oniom` does. The script calls one engine and does
  not use it.
- **Periodic cells.** `Optimizer`, `relax_band`, and `irc_path` take `cell_bohr` and `pbc`.
  `relax_band` puts every image in the cell of the first structure and lets every atom
  follow its nearest periodic image, as `nebmain` does; in a cell `irc_path` moves no
  structure to its centre of mass.

## Structure files

The script reads its input and writes its results with `mopt_rs.read_xyz` and
`mopt_rs.write_xyz`, the reader and writer of the commands (12 decimals of Angstrom for the
coordinates, 15 for a lattice):

```python
    frame = read_xyz(path)
    charge, multiplicity = frame.state or (0, 1)
    return frame.symbols, frame.positions_bohr, charge, multiplicity
```

## The same steps on the command line

`examples/python_autots/autots_config.json` has the settings of the command that match
the script:

```bash
run_autots examples/python_autots/SN2.xyz -cfg examples/python_autots/autots_config.json --run_step4
```

The test runs it with `mopt_rs.cli.run_counted("run_autots", [...])`, which also returns
the evaluations. Step 2 uses the defaults of `nebmain` there (30 iterations).
