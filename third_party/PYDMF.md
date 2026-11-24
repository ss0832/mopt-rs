# PyDMF (direct MaxFlux method, flat-bottom elastic network model)

`crates/mopt-rs/src/path/dmf/` implements the direct MaxFlux method,

> S.-i. Koda, S. Saito, "Locating Transition States by Variational Reaction Path
> Optimization with an Energy-Derivative-Free Objective Function", J. Chem. Theory Comput.
> 2024, 20, 2798-2811 (DOI: 10.1021/acs.jctc.3c01246).
>
> S.-i. Koda, S. Saito, "PyDMF: A Python package for variational double-ended reaction-path
> and transition-state optimization", J. Open Source Softw. 2026, 11, 10379
> (DOI: 10.21105/joss.10379).

with the definitions of PyDMF (`src/dmf/dmf.py`: the B-spline of degree 3 in 4 segments
fitted to the first path, the objective `ln(I) / beta` with `beta` = 10 per eV, the
trapezoidal rule over the points, the energies taken from the lower end, the movement of
the points towards the top of the energy profile with its gates and constants, the
tolerance and the first curvature of the quasi-Newton model as PyDMF hands them to
IPOPT). The Rust code is written anew and does not use IPOPT; what differs (the length of
the path integrated along the spline and the points placed by their shares of it instead
of bounds on the pace, the control points moved across the path only, the points between
the ends and those near the top) is in the module documentation and in
`docs/source/migration_paths.md`, "The direct MaxFlux method". The text of the 2024 paper was
not at hand; the definitions are those of the program and of the JOSS paper.

`crates/mopt-rs/src/path/interp/fbenm.rs` implements the flat-bottom elastic network model
and its correlated form,

> S.-i. Koda, S. Saito, "Flat-bottom Elastic Network Model for Generating Improved Plausible
> Reaction Paths", J. Chem. Theory Comput. 2024, 20, 7176-7187 (DOI: 10.1021/acs.jctc.4c00792).
>
> S.-i. Koda, S. Saito, "Correlated Flat-bottom Elastic Network Model for Improved Bond
> Rearrangement in Reaction Paths", J. Chem. Theory Comput. 2025, 21, 3513-3522
> (DOI: 10.1021/acs.jctc.4c01549).

with the definitions of the authors' program PyDMF (https://github.com/shin1koda/dmf,
`src/dmf/fbenm.py`): which pairs of atoms keep their distances, the limits of the other
pairs, the widths of the walls, the groups of atoms in a plane, the correlated term, and the
values of the parameters. The Rust code is written anew; what differs is in the module
documentation (among it: the path is found with the chain relaxation of mopt-rs, not with
the direct MaxFlux method of the program).

PyDMF is distributed under the MIT License:

```text
MIT License

Copyright (c) 2025 Shin-ichi Koda

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

The copyright line is the one of the `LICENSE` file of the repository as read on
2026-11-15.
