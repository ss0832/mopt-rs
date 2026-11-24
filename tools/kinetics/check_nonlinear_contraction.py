"""Numerical check of the contraction of mass-action networks of any reaction order.

RCMC contracts a linear rate equation ``dp/dt = K p``. This script checks its extension to
``dc/dt = f(c) = N v(c)`` with mass-action rates of any order (design:
``docs/source/kinetics_beyond_tst.md``, "Contraction of mass-action networks"):

- eliminated species ``S`` sit on the quasi-steady manifold ``f_S(c_S, c_T) = 0``, i.e.
  ``c_S = h(c_T)``;
- the kept species follow ``M(c_T) dc_T/dt = f_T(h(c_T), c_T)`` with ``M = I + B h'``,
  ``B = -J_TS J_SS^-1`` and ``h' = -J_SS^-1 J_ST`` (``J`` the Jacobian of ``f``);
- species are eliminated in the order of their effective loss rate, the diagonal of the Schur
  complement ``D = J_TT - J_TS J_SS^-1 J_ST``, while ``rate * t > theta``;
- the starting composition is moved onto the manifold along ``[I B]``, which keeps every
  conserved total.

For first-order networks ``J = K`` and all of this is RCMC (``M dp_T/dt = D p_T``). The
script compares the contracted model with a stiff integration of the full equations for
association, dimerization, enzyme, and catalytic-cycle models and prints the errors against the
separation of time scales.

Usage (needs numpy and scipy; nothing else of the project):
    python tools/kinetics/check_nonlinear_contraction.py

MultiOptPy: no counterpart (new in mopt-rs).

This file is part of mopt-rs.
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class Network:
    """Mass-action network: reactions ``sum alpha_i X_i -> sum beta_i X_i`` with constants ``k``."""

    species: list[str]
    alpha: np.ndarray  # (n, R) reactant stoichiometry
    stoich: np.ndarray  # (n, R) net stoichiometry N
    k: np.ndarray  # (R,)

    @staticmethod
    def build(species: list[str], reactions: list[tuple[str, str, float]]) -> Network:
        index = {s: i for i, s in enumerate(species)}
        n, r = len(species), len(reactions)
        alpha, beta, k = np.zeros((n, r)), np.zeros((n, r)), np.zeros(r)

        def side(text: str, into: np.ndarray, col: int) -> None:
            for term in filter(None, (t.strip() for t in text.split("+"))):
                parts = term.split()
                count, name = (float(parts[0]), parts[1]) if len(parts) == 2 else (1.0, parts[0])
                into[index[name], col] += count

        for col, (left, right, const) in enumerate(reactions):
            side(left, alpha, col)
            side(right, beta, col)
            k[col] = const
        return Network(species, alpha, beta - alpha, k)

    def rates(self, c: np.ndarray) -> np.ndarray:
        # Signed: the continued slow solution may start slightly below zero (see `project`), and
        # the rate laws stay polynomials there (the orders are integers).
        with np.errstate(divide="ignore", invalid="ignore"):
            powers = np.where(self.alpha > 0, c[:, None] ** self.alpha, 1.0)
        return self.k * powers.prod(axis=0)

    def f(self, c: np.ndarray) -> np.ndarray:
        return self.stoich @ self.rates(c)

    def jacobian(self, c: np.ndarray) -> np.ndarray:
        n, r = self.alpha.shape
        dv = np.zeros((r, n))
        cc = c
        for col in range(r):
            for i in np.nonzero(self.alpha[:, col])[0]:
                a = self.alpha[:, col].copy()
                a[i] -= 1.0
                term = self.k[col] * self.alpha[i, col]
                for j in np.nonzero(a)[0]:
                    term *= cc[j] ** a[j]
                dv[col, i] = term
        return self.stoich @ dv

    def conserved(self) -> np.ndarray:
        """Rows spanning the conserved totals (left null space of ``N``)."""
        u, s, _ = np.linalg.svd(self.stoich)
        rank = int((s > 1e-12 * max(s.max(), 1.0)).sum())
        return u[:, rank:].T


@dataclass
class Contraction:
    """A network with the species ``eliminated`` on their quasi-steady manifold."""

    net: Network
    eliminated: list[int]

    @property
    def kept(self) -> list[int]:
        return [i for i in range(len(self.net.species)) if i not in self.eliminated]

    def blocks(self, c: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``B``, ``h'``, and the Schur complement ``D`` at the full composition ``c``."""
        j = self.net.jacobian(c)
        s, t = self.eliminated, self.kept
        if not s:
            return np.zeros((len(t), 0)), np.zeros((0, len(t))), j
        jss_inv = np.linalg.inv(j[np.ix_(s, s)])
        b = -j[np.ix_(t, s)] @ jss_inv
        hp = -jss_inv @ j[np.ix_(s, t)]
        d = j[np.ix_(t, t)] + j[np.ix_(t, s)] @ hp
        return b, hp, d

    def manifold(self, c_t: np.ndarray, guess: np.ndarray) -> np.ndarray:
        """Full composition with ``f_S = 0`` for the kept concentrations ``c_t`` (Newton)."""
        s, t = self.eliminated, self.kept
        c = np.zeros(len(self.net.species))
        c[t] = c_t
        if not s:
            return c
        floor = 1e-12 * max(float(np.abs(c_t).max()), 1e-300)
        c[s] = np.maximum(guess[s], floor)
        for _ in range(200):
            f_s = self.net.f(c)[s]
            j_ss = self.net.jacobian(c)[np.ix_(s, s)]
            step = np.linalg.solve(j_ss, -f_s)
            # Keep the concentrations non-negative: shorten a step that overshoots, and let a
            # solution that is zero be zero.
            lam = 1.0
            if np.all(c[s] + step >= -floor):
                c[s] = np.maximum(c[s] + step, 0.0)
            else:
                while np.any(c[s] + lam * step <= 0.0) and lam > 1e-12:
                    lam *= 0.5
                c[s] = c[s] + lam * step
            if np.abs(lam * step).max() <= 1e-14 * max(float(np.abs(c[s]).max()), floor):
                break
        return c

    def project(self, c_from: np.ndarray) -> np.ndarray:
        """``c_from`` moved onto the manifold along ``[I B]`` (conserved totals are kept)."""
        s, t = self.eliminated, self.kept
        if not s:
            return c_from.copy()
        guess = c_from.copy()
        c_t = c_from[t].copy()
        scale = max(float(np.abs(c_from).max()), 1e-300)

        def residual(x: np.ndarray) -> np.ndarray:
            c = self.manifold(x, guess)
            b, _, _ = self.blocks(c)
            return c[t] - c_from[t] + b @ (c[s] - c_from[s])

        for _ in range(100):
            r = residual(c_t)
            if np.abs(r).max() <= 1e-13 * scale:
                break
            jac = np.empty((len(t), len(t)))
            for col in range(len(t)):
                h = 1e-6 * scale
                shifted = c_t.copy()
                shifted[col] += h
                jac[:, col] = (residual(shifted) - r) / h
            # No positivity here: the start of the slow solution may lie slightly below zero for
            # a product of an eliminated species that has yet to build up (matched asymptotics:
            # the slow solution continued back to t = 0).
            c_t = c_t + np.linalg.solve(jac, -r)
            guess = self.manifold(c_t, guess)
        return self.manifold(c_t, guess)

    def propagate(self, c_start: np.ndarray, t_end: float) -> np.ndarray:
        """Integrate ``M dc_T/dt = f_T`` on the manifold from the projected start."""
        s, t = self.eliminated, self.kept
        state = {"c": self.project(c_start)}

        def rhs(_time: float, c_t: np.ndarray) -> np.ndarray:
            c = self.manifold(c_t, state["c"])
            state["c"] = c
            b, hp, _ = self.blocks(c)
            m = np.eye(len(t)) + b @ hp
            return np.linalg.solve(m, self.net.f(c)[t])

        if not s:
            return integrate(self.net, c_start, t_end)
        sol = solve_ivp(rhs, (0.0, t_end), state["c"][t], method="Radau", rtol=1e-10, atol=1e-16)
        return self.manifold(sol.y[:, -1], state["c"])


def integrate(net: Network, c0: np.ndarray, t_end: float) -> np.ndarray:
    """Reference: stiff integration of the full rate equations."""
    sol = solve_ivp(
        lambda _t, c: net.f(c),
        (0.0, t_end),
        c0,
        method="Radau",
        jac=lambda _t, c: net.jacobian(c),
        rtol=1e-11,
        atol=1e-18,
    )
    return sol.y[:, -1]


def contract(net: Network, c0: np.ndarray, t: float, theta: float) -> tuple[Contraction, list]:
    """Eliminate species, fastest first, while ``effective loss rate * t > theta``."""
    con = Contraction(net, [])
    c = c0.copy()
    log = []
    while len(con.eliminated) < len(net.species) - 1:
        _, _, d = con.blocks(c)
        diag = -np.diag(d)
        pick = int(np.argmax(diag))
        rate = diag[pick]
        if rate * t <= theta:
            break
        trial = Contraction(net, [*con.eliminated, con.kept[pick]])
        j = net.jacobian(c)
        jss = j[np.ix_(trial.eliminated, trial.eliminated)]
        # The eliminated block has to relax as a whole within the horizon: its slowest mode
        # decides, not the loss rates of its members (all forms of a catalyst are short-lived,
        # their total is not).
        if -np.linalg.eigvals(jss).real.max() * t <= theta:
            break
        try:
            c = trial.project(c)
        except np.linalg.LinAlgError:
            break  # no quasi-steady state for this set at this composition
        log.append((net.species[con.kept[pick]], rate))
        con = trial
    return con, log


def relative_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).max() / np.abs(b).max())


def report(title: str, net: Network, c0: np.ndarray, t: float, theta: float = 10.0) -> None:
    exact = integrate(net, c0, t)
    con, log = contract(net, c0, t, theta)
    snapshot = con.project(c0)
    reduced = con.propagate(c0, t)
    w = net.conserved()
    drift = float(np.abs(w @ (reduced - c0)).max() / np.abs(c0).max()) if len(w) else 0.0
    names = ", ".join(f"{s} ({r:.1e}/s)" for s, r in log) or "none"
    print(f"{title}\n  eliminated: {names}")
    print(
        f"  error at t = {t:g}: contracted and propagated {relative_error(reduced, exact):.2e}, "
        f"snapshot (no propagation) {relative_error(snapshot, exact):.2e}, "
        f"conserved totals {drift:.1e}"
    )


def linear_case() -> None:
    """First-order network: the blocks are those of RCMC."""
    rng = np.random.default_rng(3)
    n = 7
    g = rng.uniform(0.0, 6.0, n)
    k = np.zeros((n, n))
    reactions = []
    for i in range(n):
        for j in range(i):
            if j == i - 1 or rng.random() < 0.4:
                barrier = max(g[i], g[j]) + rng.uniform(1.0, 12.0)
                k[i, j], k[j, i] = np.exp(-(barrier - g[j])), np.exp(-(barrier - g[i]))
                reactions += [(f"X{j}", f"X{i}", k[i, j]), (f"X{i}", f"X{j}", k[j, i])]
    net = Network.build([f"X{i}" for i in range(n)], reactions)
    gen = k - np.diag(k.sum(axis=0))
    c0 = np.zeros(n)
    c0[int(np.argmax(g))] = 1.0
    t_end = 30.0
    con, _ = contract(net, c0, t_end, 10.0)
    s, t = con.eliminated, con.kept
    c = con.project(c0)
    b, hp, d = con.blocks(c)
    kss_inv = np.linalg.inv(gen[np.ix_(s, s)])
    d_rcmc = gen[np.ix_(t, t)] - gen[np.ix_(t, s)] @ kss_inv @ gen[np.ix_(s, t)]
    m_rcmc = np.eye(len(t)) + gen[np.ix_(t, s)] @ kss_inv @ kss_inv @ gen[np.ix_(s, t)]
    q_rcmc = c0[t] - gen[np.ix_(t, s)] @ kss_inv @ c0[s]
    print(f"First-order network ({n} states): the contraction is RCMC")
    print(
        f"  eliminated {len(s)} states; |D - D_RCMC| = {np.abs(d - d_rcmc).max():.1e}, "
        f"|M - M_RCMC| = {np.abs(np.eye(len(t)) + b @ hp - m_rcmc).max():.1e}, "
        f"|q - q_RCMC| = {np.abs(c[t] + b @ c[s] - q_rcmc).max():.1e}"
    )
    for theta in [10.0, 100.0, 1000.0]:
        t_long = theta * 3.0
        con, _ = contract(net, c0, t_long, theta)
        reduced, exact = con.propagate(c0, t_long), integrate(net, c0, t_long)
        print(
            f"  t = {t_long:g}, states with loss rate * t > {theta:g} eliminated "
            f"({len(con.eliminated)}): error {relative_error(reduced, exact):.2e}, "
            f"total population {reduced.sum():.14f}"
        )


def separation_scan() -> None:
    """A + B <-> X -> P: the error against the separation of time scales."""
    print("A + B <-> X -> P with c_A = 1, c_B = 0.6, k_assoc = 1: error against k_diss + k_2")
    for fast in [1e1, 1e2, 1e3, 1e4, 1e5]:
        net = Network.build(
            ["A", "B", "X", "P"],
            [("A + B", "X", 1.0), ("X", "A + B", 0.7 * fast), ("X", "P", 0.3 * fast)],
        )
        c0 = np.array([1.0, 0.6, 0.0, 0.0])
        exact = integrate(net, c0, 3.0)
        con = Contraction(net, [2])
        zeroth = solve_ivp(
            lambda _t, y, con=con, net=net: net.f(con.manifold(y, np.zeros(4)))[con.kept],
            (0.0, 3.0),
            c0[con.kept],
            method="Radau",
            rtol=1e-10,
            atol=1e-16,
        ).y[:, -1]
        corrected = con.propagate(c0, 3.0)
        print(
            f"  k_diss + k_2 = {fast:.0e}: with M and the projected start "
            f"{relative_error(corrected[con.kept], exact[con.kept]):.2e}, "
            f"plain QSSA {relative_error(zeroth, exact[con.kept]):.2e}"
        )


def main() -> None:
    np.set_printoptions(precision=6, suppress=True)
    linear_case()
    separation_scan()
    report(
        "Association with the complex as the major species: A + B <-> X (fast, strong), X -> P (slow)",
        Network.build(
            ["A", "B", "X", "P"],
            [("A + B", "X", 1e6), ("X", "A + B", 1e2), ("X", "P", 1e-2)],
        ),
        np.array([1.0, 0.6, 0.0, 0.0]),
        t=30.0,
    )
    report(
        "Dimerization: 2 A <-> D (fast), D -> P (slow)",
        Network.build(["A", "D", "P"], [("2 A", "D", 1e5), ("D", "2 A", 1e3), ("D", "P", 1e-1)]),
        np.array([1.0, 0.0, 0.0]),
        t=10.0,
    )
    report(
        "Enzyme: E + S <-> ES -> E + P (substrate in excess)",
        Network.build(
            ["E", "S", "ES", "P"],
            [("E + S", "ES", 1e4), ("ES", "E + S", 1e3), ("ES", "E + P", 1e2)],
        ),
        np.array([1e-3, 1.0, 0.0, 0.0]),
        t=20.0,
    )
    report(
        "Catalytic cycle: C + S <-> CS -> CI -> CP <-> C + P, with a second-order side reaction 2 CI -> Q",
        Network.build(
            ["C", "S", "CS", "CI", "CP", "P", "Q"],
            [
                ("C + S", "CS", 5e3),
                ("CS", "C + S", 2e3),
                ("CS", "CI", 4e2),
                ("CI", "CP", 9e2),
                ("CP", "C + P", 3e3),
                ("C + P", "CP", 1e2),
                ("2 CI", "Q", 5e3),
            ],
        ),
        np.array([1e-2, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        t=5.0,
    )
    report(
        "Third order: 2 A + B <-> T (fast), T -> P (slow)",
        Network.build(
            ["A", "B", "T", "P"],
            [("2 A + B", "T", 1e5), ("T", "2 A + B", 1e3), ("T", "P", 5e-2)],
        ),
        np.array([1.0, 0.8, 0.0, 0.0]),
        t=10.0,
    )


if __name__ == "__main__":
    main()
