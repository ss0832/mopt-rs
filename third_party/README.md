# Third-party material and provenance

mopt-rs (the successor implementation of MultiOptPy) is distributed under GPL-3.0-or-later,
Copyright (C) 2023-2026 ss0832. This directory records where its scientific parameters and
algorithms come from and the licenses of its dependencies.

- [DISPERSION_AND_PARAMETERS.md](DISPERSION_AND_PARAMETERS.md): sources of every compiled-in
  parameter table (force fields, dispersion models, model Hessians).
- [GEODESIC_INTERPOLATE.md](GEODESIC_INTERPOLATE.md): the program whose algorithm the geodesic
  interpolation follows, and its licence (MIT).
- [PYDMF.md](PYDMF.md): the program whose definitions the flat-bottom elastic network
  model follows, and its licence (MIT).
- [TWO_STRUCTURE_SEARCHES.md](TWO_STRUCTURE_SEARCHES.md): the publications behind the
  searches of `ieipmain` (iEIP, dimer, spring pair, 2PSHS, ADDF, growing Newton trajectory).
- [MOLECULAR_DYNAMICS.md](MOLECULAR_DYNAMICS.md): the publications behind the integrators,
  thermostats, barostats, and constraints of the molecular dynamics.
- `ALGORITHMS_PROVENANCE.md`: original publications of the implemented methods (added as the
  methods are ported).
- `RUST_CRATES_LICENSES.md`: licenses of the Rust dependencies `faer`, `rayon`, `libm`, `pyo3`.

Reference implementations used only for testing (MultiOptPy 1.21.2 and ASE) are run from a
separate Python environment by the scripts in `tools/golden/`; no code from them is included in
mopt-rs. The inputs of the quick start in `examples/quickstart/` are the example files of
`test/` of MultiOptPy 1.21.2 (GPL-3.0-or-later, by the author of mopt-rs), unchanged; see the
README there.
