# geodesic-interpolate

`crates/mopt-rs/src/path/interp/geodesic.rs` follows the algorithm of the program
geodesic-interpolate (https://github.com/virtualzx-nad/geodesic-interpolate), the
implementation of

> X. Zhu, K. C. Thompson, T. J. Martinez, "Geodesic interpolation for reaction pathways",
> J. Chem. Phys. 2019, 150, 164103.

What was taken from it: the coordinates (scaled interatomic distances with the parameters
`alpha` and `beta`), the rule for which pairs of atoms are coordinates (a cutoff, atoms
within two bonds, a least number of neighbours), the search for a new image between two
images, the function that is minimized for the whole path, and the values of the
parameters. It was read in the copy that MultiOptPy 1.21.2 has
(`multioptpy/Interpolation/geodesic_interpolation.py`). The Rust code is written anew and
solves the least-squares problems without the dense Jacobian of the program (see the module
documentation for the differences).

geodesic-interpolate is distributed under the MIT License:

```text
MIT License

Copyright (c) 2024 Xiaolei Zhu

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
2026-11-14.
