# mopt-rs container image: the `mopt` binary and the Python package (distribution `mopt-py`,
# imported as `mopt_rs`, with the commands `mopt-py`, `optmain`, ...)
# Copyright (C) 2023-2026 ss0832
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Build (from the root of the repository):
#     docker build -t mopt-rs .
# Run a command on the files of the current directory:
#     docker run --rm -v "$PWD":/work mopt-rs optmain input.xyz -os lj
#     docker run --rm -v "$PWD":/work mopt-rs mopt optmain input.xyz -os lj
#     docker run --rm -it -v "$PWD":/work mopt-rs python
#
# The image holds the engines built into mopt-rs and numpy. The optional engines (PySCF,
# tblite, dxtb, ASE calculators, OpenMM, ...) are extras of the Python package and are not
# installed; add what you need in an image of your own that starts `FROM` this one, for example
#     RUN pip install --no-cache-dir pyscf

ARG PYTHON_IMAGE=python:3.12-slim-bookworm

# --- build stage: the wheel and the stand-alone binary --------------------------------------
FROM ${PYTHON_IMAGE} AS build

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# The Rust toolchain (stable, minimal profile).
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --profile minimal --default-toolchain stable
ENV PATH="/root/.cargo/bin:${PATH}"

RUN pip install --no-cache-dir "maturin>=1.7,<2.0"

WORKDIR /src
COPY . .

# The versions of Cargo.lock are used as they are.
RUN maturin build --release --locked --out /wheels
RUN cargo build --release --locked -p mopt-rs --bin mopt

# --- runtime stage ---------------------------------------------------------------------------
FROM ${PYTHON_IMAGE}

COPY --from=build /wheels /tmp/wheels
RUN pip install --no-cache-dir /tmp/wheels/*.whl && rm -rf /tmp/wheels
COPY --from=build /src/target/release/mopt /usr/local/bin/mopt

WORKDIR /work
