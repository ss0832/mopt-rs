//! User-defined bias potentials written as expressions (`-ubias`).
//!
//! The expression gives the energy in Hartree from geometric variables in atomic units (Bohr,
//! radians). Exact gradients and Hessians come from automatic differentiation, and the
//! geometry uses minimum images in periodic cells, like the built-in biases. Grammar:
//!
//! ```text
//! expr    = term { ("+" | "-") term }
//! term    = unary { ("*" | "/") unary }
//! unary   = ("-" | "+") unary | power
//! power   = primary [ ("^" | "**") unary ]
//! primary = number | name | name "(" args ")" | "(" expr ")"
//! ```
//!
//! - Geometry (arguments are atoms, `3`, or atom groups, `[1-3,5]`, entering by their
//!   centroids; numbers are 1-based): `dist(a, b)`, `angle(a, b, c)` (at `b`),
//!   `dihedral(a, b, c, d)` (IUPAC sign), `oop(c, a, b, d)` (angle of `a` above the plane of
//!   `b` and `d` around the center `c`), `x(a)`, `y(a)`, `z(a)` (Cartesian coordinates; in
//!   periodic cells those of the image nearest to the origin).
//! - The cell (no arguments; the lattice vectors `a`, `b`, `c` of a periodic structure, in
//!   Bohr and radians): `cell_a()`, `cell_b()`, `cell_c()` (lengths), `cell_alpha()` (the
//!   angle between `b` and `c`), `cell_beta()` (`a`, `c`), `cell_gamma()` (`a`, `b`),
//!   `volume()` (Bohr^3, also `cell_volume()`), and the Cartesian components `cell_ax()`,
//!   `cell_ay()`, ..., `cell_cz()`. In a variable-cell optimization they change with the
//!   cell and are differentiated exactly along it ([`CellExpression`]); a structure without
//!   a cell has none of them.
//! - Functions: `exp`, `ln` (`log`), `sqrt`, `sin`, `cos`, `tan`, `tanh`, `acos`, `abs`,
//!   `atan2(y, x)`, `min(a, b)`, `max(a, b)`.
//! - Constants: `pi`; unit factors `ang` (Bohr per Angstrom), `deg` (radian per degree),
//!   `kjmol`, `kcalmol`, `ev` (Hartree per unit).
//! - Variables: `it` (optimizer iteration) and `t` (simulated time in atomic units in molecular
//!   dynamics, otherwise the iteration), for time-dependent biases.
//! - In objectives of plain variables ([`VariableExpression`], `-os expr:...`): `q1`, `q2`, ...
//!   (1-based) instead of the geometry functions.
//!
//! Example: `0.5 * 0.3 * (dist(1, 2) - 1.5 * ang)^2 + 10 * kjmol * exp(-dist([3-5], 6) / ang)`.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::f64::consts::PI;
use std::sync::Arc;

use faer::Mat;

use super::engine::{local_coordinates, norm, scatter_local, AdBias, Clock, Geom, GeometricBias};
use super::restraints::{dihedral, out_of_plane};
use crate::core_infra::arg_parser::parse_atom_indices_one_based;
use crate::domain::{AtomId, EvalOutput, MolecularSystem, Quantity, StrainHessian, Units};
use crate::error::{MoptError, MoptResult};
use crate::math::ad::{
    angle_between, eval_value_and_gradient, eval_value_grad_hessian, Dual, HyperDual, Scalar,
};
use crate::pes::term::{check_term_output, Frame, History, PesTerm, TermContext};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Op {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Func {
    Exp,
    Ln,
    Sqrt,
    Sin,
    Cos,
    Tan,
    Tanh,
    Acos,
    Abs,
    Atan2,
    Min,
    Max,
}

impl Func {
    fn lookup(name: &str) -> Option<(Self, usize)> {
        Some(match name {
            "exp" => (Self::Exp, 1),
            "ln" | "log" => (Self::Ln, 1),
            "sqrt" => (Self::Sqrt, 1),
            "sin" => (Self::Sin, 1),
            "cos" => (Self::Cos, 1),
            "tan" => (Self::Tan, 1),
            "tanh" => (Self::Tanh, 1),
            "acos" => (Self::Acos, 1),
            "abs" => (Self::Abs, 1),
            "atan2" => (Self::Atan2, 2),
            "min" => (Self::Min, 2),
            "max" => (Self::Max, 2),
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GeoFn {
    Dist,
    Angle,
    Dihedral,
    OutOfPlane,
    Coord(usize),
}

impl GeoFn {
    fn lookup(name: &str) -> Option<(Self, usize)> {
        Some(match name {
            "dist" => (Self::Dist, 2),
            "angle" => (Self::Angle, 3),
            "dihedral" => (Self::Dihedral, 4),
            "oop" => (Self::OutOfPlane, 4),
            "x" => (Self::Coord(0), 1),
            "y" => (Self::Coord(1), 1),
            "z" => (Self::Coord(2), 1),
            _ => return None,
        })
    }
}

/// A function of the lattice vectors (the rows `a`, `b`, `c` of the cell).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CellFn {
    /// Length of a lattice vector.
    Length(usize),
    /// Angle between the other two lattice vectors (alpha: `b`, `c`; beta: `a`, `c`; gamma:
    /// `a`, `b`).
    Angle(usize),
    /// Volume of the cell.
    Volume,
    /// Cartesian component `c` of the lattice vector `k`.
    Component(usize, usize),
}

impl CellFn {
    fn lookup(name: &str) -> Option<Self> {
        Some(match name {
            "cell_a" => Self::Length(0),
            "cell_b" => Self::Length(1),
            "cell_c" => Self::Length(2),
            "cell_alpha" => Self::Angle(0),
            "cell_beta" => Self::Angle(1),
            "cell_gamma" => Self::Angle(2),
            "volume" | "cell_volume" => Self::Volume,
            other => {
                let tag = other.strip_prefix("cell_")?.as_bytes();
                let vector = match tag {
                    [b'a', _] => 0,
                    [b'b', _] => 1,
                    [b'c', _] => 2,
                    _ => return None,
                };
                let axis = match tag[1] {
                    b'x' => 0,
                    b'y' => 1,
                    b'z' => 2,
                    _ => return None,
                };
                Self::Component(vector, axis)
            }
        })
    }

    fn name(self) -> String {
        let vectors = ["a", "b", "c"];
        match self {
            Self::Length(k) => format!("cell_{}", vectors[k]),
            Self::Angle(k) => format!("cell_{}", ["alpha", "beta", "gamma"][k]),
            Self::Volume => "volume".to_string(),
            Self::Component(k, c) => format!("cell_{}{}", vectors[k], ["x", "y", "z"][c]),
        }
    }

    /// The value at the lattice vectors `l` (rows).
    fn value<T: Scalar>(self, l: &[[T; 3]; 3]) -> T {
        match self {
            Self::Length(k) => T::norm3(l[k]),
            Self::Angle(k) => {
                let (u, v) = match k {
                    0 => (1, 2),
                    1 => (0, 2),
                    _ => (0, 1),
                };
                // atan2(|u x v|, u . v): accurate up to 0 and 180 degrees.
                angle_between(l[u], l[v])
            }
            Self::Volume => {
                let v = T::dot3(l[0], T::cross3(l[1], l[2]));
                pick(v.val() < 0.0, -v, v)
            }
            Self::Component(k, c) => l[k][c],
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Node {
    Num(f64),
    Iteration,
    Time,
    Neg(Box<Node>),
    Bin(Op, Box<Node>, Box<Node>),
    Pow(Box<Node>, Box<Node>),
    Call(Func, Vec<Node>),
    Geo(GeoFn, Vec<Vec<AtomId>>),
    /// A function of the lattice vectors.
    Cell(CellFn),
    /// Plain variable `q{k+1}` (objectives only).
    Var(usize),
}

/// What an expression is evaluated on: the geometry of atoms or a vector of variables.
trait Env<T: Scalar> {
    fn geo(&self, f: GeoFn, groups: &[Vec<AtomId>]) -> T;
    fn var(&self, k: usize) -> T;
    /// The lattice vectors (rows), where the evaluation sees them.
    fn lattice(&self) -> Option<[[T; 3]; 3]> {
        None
    }
}

/// The geometry of the atoms together with the lattice vectors of their cell, both
/// variables of the derivatives ([`CellExpression`]).
struct CellGeom<'a, T: Scalar> {
    geom: Geom<'a, T>,
    lattice: [[T; 3]; 3],
}

impl<T: Scalar> Env<T> for CellGeom<'_, T> {
    fn geo(&self, f: GeoFn, groups: &[Vec<AtomId>]) -> T {
        Env::<T>::geo(&self.geom, f, groups)
    }

    fn var(&self, _k: usize) -> T {
        T::zero()
    }

    fn lattice(&self) -> Option<[[T; 3]; 3]> {
        Some(self.lattice)
    }
}

impl<T: Scalar> Env<T> for Geom<'_, T> {
    fn geo(&self, f: GeoFn, gr: &[Vec<AtomId>]) -> T {
        let g = self;
        match f {
            GeoFn::Dist => norm(g.d_centroids(&gr[0], &gr[1])),
            GeoFn::Angle => {
                angle_between(g.d_centroids(&gr[0], &gr[1]), g.d_centroids(&gr[2], &gr[1]))
            }
            GeoFn::Dihedral => {
                let b1 = g.d_centroids(&gr[1], &gr[0]);
                let b2 = g.d_centroids(&gr[2], &gr[1]);
                let b3 = g.d_centroids(&gr[3], &gr[2]);
                dihedral(b1, b2, b3).0
            }
            GeoFn::OutOfPlane => {
                let a1 = g.d_centroids(&gr[1], &gr[0]);
                let a2 = g.d_centroids(&gr[2], &gr[0]);
                let a3 = g.d_centroids(&gr[3], &gr[0]);
                out_of_plane(a1, a2, a3).unwrap_or(T::zero())
            }
            GeoFn::Coord(k) => g.mic(g.centroid(&gr[0]))[k],
        }
    }

    // Variables are rejected when a bias is parsed.
    fn var(&self, _k: usize) -> T {
        T::zero()
    }
}

/// Plain variables.
struct Vars<'a, T>(&'a [T]);

impl<T: Scalar> Env<T> for Vars<'_, T> {
    // Geometry is rejected when an objective is parsed.
    fn geo(&self, _f: GeoFn, _groups: &[Vec<AtomId>]) -> T {
        T::zero()
    }

    fn var(&self, k: usize) -> T {
        self.0.get(k).copied().unwrap_or(T::zero())
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Num(f64),
    Name(String),
    Group(String),
    LParen,
    RParen,
    Comma,
    Plus,
    Minus,
    Star,
    Slash,
    Caret,
}

fn lex(src: &str) -> MoptResult<Vec<Tok>> {
    let err = |m: String| MoptError::parse(format!("bias expression: {m} in '{src}'"));
    let chars: Vec<char> = src.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        match c {
            ' ' | '\t' | '\n' | '\r' => i += 1,
            '(' => {
                out.push(Tok::LParen);
                i += 1;
            }
            ')' => {
                out.push(Tok::RParen);
                i += 1;
            }
            ',' => {
                out.push(Tok::Comma);
                i += 1;
            }
            '+' => {
                out.push(Tok::Plus);
                i += 1;
            }
            '-' => {
                out.push(Tok::Minus);
                i += 1;
            }
            '/' => {
                out.push(Tok::Slash);
                i += 1;
            }
            '^' => {
                out.push(Tok::Caret);
                i += 1;
            }
            '*' => {
                if chars.get(i + 1) == Some(&'*') {
                    out.push(Tok::Caret);
                    i += 2;
                } else {
                    out.push(Tok::Star);
                    i += 1;
                }
            }
            '[' => {
                let end = chars[i..]
                    .iter()
                    .position(|&d| d == ']')
                    .ok_or_else(|| err("unclosed '['".into()))?;
                out.push(Tok::Group(chars[i + 1..i + end].iter().collect()));
                i += end + 1;
            }
            d if d.is_ascii_digit() || d == '.' => {
                let start = i;
                while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                    i += 1;
                }
                // Exponent: 1e-3, 2.5E+4.
                if i < chars.len() && (chars[i] == 'e' || chars[i] == 'E') {
                    let mut j = i + 1;
                    if j < chars.len() && (chars[j] == '+' || chars[j] == '-') {
                        j += 1;
                    }
                    if j < chars.len() && chars[j].is_ascii_digit() {
                        i = j;
                        while i < chars.len() && chars[i].is_ascii_digit() {
                            i += 1;
                        }
                    }
                }
                let text: String = chars[start..i].iter().collect();
                out.push(Tok::Num(
                    text.parse()
                        .map_err(|_| err(format!("invalid number '{text}'")))?,
                ));
            }
            a if a.is_ascii_alphabetic() || a == '_' => {
                let start = i;
                while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                    i += 1;
                }
                out.push(Tok::Name(chars[start..i].iter().collect()));
            }
            other => return Err(err(format!("unexpected character '{other}'"))),
        }
    }
    Ok(out)
}

struct Parser<'a> {
    toks: Vec<Tok>,
    pos: usize,
    src: &'a str,
    /// Plain variables `q1`, `q2`, ... instead of geometry (objectives).
    variables: bool,
}

impl Parser<'_> {
    fn err(&self, m: &str) -> MoptError {
        MoptError::parse(format!("bias expression: {m} in '{}'", self.src))
    }

    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos)
    }

    fn next(&mut self) -> Option<Tok> {
        let t = self.toks.get(self.pos).cloned();
        self.pos += 1;
        t
    }

    fn expect(&mut self, t: &Tok) -> MoptResult<()> {
        if self.next().as_ref() == Some(t) {
            Ok(())
        } else {
            Err(self.err(&format!("expected {t:?}")))
        }
    }

    fn expr(&mut self) -> MoptResult<Node> {
        let mut lhs = self.term()?;
        while let Some(op) = match self.peek() {
            Some(Tok::Plus) => Some(Op::Add),
            Some(Tok::Minus) => Some(Op::Sub),
            _ => None,
        } {
            self.pos += 1;
            lhs = Node::Bin(op, Box::new(lhs), Box::new(self.term()?));
        }
        Ok(lhs)
    }

    fn term(&mut self) -> MoptResult<Node> {
        let mut lhs = self.unary()?;
        while let Some(op) = match self.peek() {
            Some(Tok::Star) => Some(Op::Mul),
            Some(Tok::Slash) => Some(Op::Div),
            _ => None,
        } {
            self.pos += 1;
            lhs = Node::Bin(op, Box::new(lhs), Box::new(self.unary()?));
        }
        Ok(lhs)
    }

    fn unary(&mut self) -> MoptResult<Node> {
        match self.peek() {
            Some(Tok::Minus) => {
                self.pos += 1;
                Ok(Node::Neg(Box::new(self.unary()?)))
            }
            Some(Tok::Plus) => {
                self.pos += 1;
                self.unary()
            }
            _ => self.power(),
        }
    }

    fn power(&mut self) -> MoptResult<Node> {
        let base = self.primary()?;
        if self.peek() == Some(&Tok::Caret) {
            self.pos += 1;
            return Ok(Node::Pow(Box::new(base), Box::new(self.unary()?)));
        }
        Ok(base)
    }

    fn group(&mut self) -> MoptResult<Vec<AtomId>> {
        let text = match self.next() {
            Some(Tok::Num(v)) if v >= 1.0 && v.fract() == 0.0 => format!("{v}"),
            Some(Tok::Group(g)) => g,
            _ => return Err(self.err("expected an atom number or an atom group [..]")),
        };
        let atoms = parse_atom_indices_one_based(&text)?
            .into_iter()
            .map(AtomId::from_one_based)
            .collect::<MoptResult<Vec<_>>>()?;
        if atoms.is_empty() {
            return Err(self.err("empty atom group"));
        }
        Ok(atoms)
    }

    fn args<T>(
        &mut self,
        n: usize,
        name: &str,
        mut one: impl FnMut(&mut Self) -> MoptResult<T>,
    ) -> MoptResult<Vec<T>> {
        self.expect(&Tok::LParen)?;
        let mut v = vec![one(self)?];
        while self.peek() == Some(&Tok::Comma) {
            self.pos += 1;
            v.push(one(self)?);
        }
        self.expect(&Tok::RParen)?;
        if v.len() == n {
            Ok(v)
        } else {
            Err(self.err(&format!("{name} takes {n} arguments, got {}", v.len())))
        }
    }

    fn primary(&mut self) -> MoptResult<Node> {
        match self.next() {
            Some(Tok::Num(v)) => Ok(Node::Num(v)),
            Some(Tok::LParen) => {
                let e = self.expr()?;
                self.expect(&Tok::RParen)?;
                Ok(e)
            }
            Some(Tok::Name(name)) => {
                if self.peek() == Some(&Tok::LParen) {
                    if let Some(f) = CellFn::lookup(&name) {
                        if self.variables {
                            return Err(self.err(&format!(
                                "'{name}()' needs a cell; an objective uses the variables q1, \
                                 q2, ..."
                            )));
                        }
                        self.expect(&Tok::LParen)?;
                        if self.next() != Some(Tok::RParen) {
                            return Err(self.err(&format!("{name}() takes no arguments")));
                        }
                        return Ok(Node::Cell(f));
                    }
                    if let Some((g, n)) = GeoFn::lookup(&name) {
                        if self.variables {
                            return Err(self.err(&format!(
                                "'{name}' needs atoms; an objective uses the variables q1, q2, ..."
                            )));
                        }
                        return Ok(Node::Geo(g, self.args(n, &name, Self::group)?));
                    }
                    if let Some((f, n)) = Func::lookup(&name) {
                        return Ok(Node::Call(f, self.args(n, &name, Self::expr)?));
                    }
                    return Err(self.err(&format!("unknown function '{name}'")));
                }
                Ok(match name.as_str() {
                    "pi" => Node::Num(PI),
                    "ang" => Node::Num(Units::ANGSTROM_TO_BOHR),
                    "deg" => Node::Num(Units::DEG_TO_RAD),
                    "kjmol" => Node::Num(Units::KJ_MOL_TO_HARTREE),
                    "kcalmol" => Node::Num(Units::KCAL_MOL_TO_HARTREE),
                    "ev" => Node::Num(Units::EV_TO_HARTREE),
                    "it" => Node::Iteration,
                    "t" => Node::Time,
                    v if self.variables
                        && v.len() > 1
                        && v.starts_with('q')
                        && v[1..].chars().all(|c| c.is_ascii_digit()) =>
                    {
                        let k: usize = v[1..]
                            .parse()
                            .map_err(|_| self.err(&format!("invalid variable '{v}'")))?;
                        if k == 0 {
                            return Err(self.err("variables are numbered from q1"));
                        }
                        Node::Var(k - 1)
                    }
                    _ => return Err(self.err(&format!("unknown name '{name}'"))),
                })
            }
            _ => Err(self.err("unexpected end or token")),
        }
    }
}

fn pick<T: Scalar>(cond: bool, a: T, b: T) -> T {
    if cond {
        a
    } else {
        b
    }
}

/// `base^p` for an exponent `p` that is itself a variable: `exp(p ln base)` for a positive
/// base; for a negative base and an exponent whose value is an integer `n`,
/// `(-1)^n exp(p ln |base|)` (the real branch); `0^p = 0` for `p > 0`. Anything else (a
/// negative base to a non-integer power, `0^p` for `p <= 0`) has no real value and gives NaN,
/// which the bias reports with [`Node::undefined`].
fn variable_power<T: Scalar>(base: T, p: T) -> T {
    let (b, e) = (base.val(), p.val());
    if b > 0.0 {
        (p * base.ln()).exp()
    } else if b == 0.0 && e > 0.0 {
        T::zero()
    } else if b < 0.0 && e.is_finite() && e.fract() == 0.0 {
        let magnitude = (p * (-base).ln()).exp();
        if (0.5 * e).fract() == 0.0 {
            magnitude
        } else {
            -magnitude
        }
    } else {
        T::from_f64(f64::NAN)
    }
}

/// Name of a function in messages.
fn func_name(f: Func) -> &'static str {
    match f {
        Func::Exp => "exp",
        Func::Ln => "ln",
        Func::Sqrt => "sqrt",
        Func::Sin => "sin",
        Func::Cos => "cos",
        Func::Tan => "tan",
        Func::Tanh => "tanh",
        Func::Acos => "acos",
        Func::Abs => "abs",
        Func::Atan2 => "atan2",
        Func::Min => "min",
        Func::Max => "max",
    }
}

impl Node {
    fn constant(&self) -> Option<f64> {
        Some(match self {
            Self::Num(v) => *v,
            Self::Iteration | Self::Time | Self::Geo(..) | Self::Cell(_) | Self::Var(_) => {
                return None
            }
            Self::Neg(a) => -a.constant()?,
            Self::Bin(op, a, b) => {
                let (a, b) = (a.constant()?, b.constant()?);
                match op {
                    Op::Add => a + b,
                    Op::Sub => a - b,
                    Op::Mul => a * b,
                    Op::Div => a / b,
                }
            }
            Self::Pow(a, b) => a.constant()?.powf(b.constant()?),
            Self::Call(f, args) => {
                let x = args[0].constant()?;
                let y = || args.get(1).and_then(Node::constant);
                match f {
                    Func::Exp => x.exp(),
                    Func::Ln => x.ln(),
                    Func::Sqrt => x.sqrt(),
                    Func::Sin => x.sin(),
                    Func::Cos => x.cos(),
                    Func::Tan => x.tan(),
                    Func::Tanh => x.tanh(),
                    Func::Acos => x.acos(),
                    Func::Abs => x.abs(),
                    Func::Atan2 => x.atan2(y()?),
                    Func::Min => x.min(y()?),
                    Func::Max => x.max(y()?),
                }
            }
        })
    }

    fn uses_cartesian(&self) -> bool {
        match self {
            Self::Num(_) | Self::Iteration | Self::Time | Self::Var(_) | Self::Cell(_) => false,
            Self::Neg(a) => a.uses_cartesian(),
            Self::Bin(_, a, b) | Self::Pow(a, b) => a.uses_cartesian() || b.uses_cartesian(),
            Self::Call(_, args) => args.iter().any(Node::uses_cartesian),
            Self::Geo(f, _) => matches!(f, GeoFn::Coord(_)),
        }
    }

    /// Whether the expression contains a leaf for which `leaf` holds.
    fn any_leaf(&self, leaf: &impl Fn(&Self) -> bool) -> bool {
        match self {
            Self::Neg(a) => a.any_leaf(leaf),
            Self::Bin(_, a, b) | Self::Pow(a, b) => a.any_leaf(leaf) || b.any_leaf(leaf),
            Self::Call(_, args) => args.iter().any(|a| a.any_leaf(leaf)),
            other => leaf(other),
        }
    }

    /// The first operation (innermost first) whose value is not finite although its arguments
    /// are, described for an error message; `None` if every value is finite or a geometric
    /// variable is already not finite.
    fn undefined<E: Env<f64>>(&self, g: &E, clock: Clock) -> Option<String> {
        let children: Vec<&Node> = match self {
            Self::Neg(a) => vec![a],
            Self::Bin(_, a, b) | Self::Pow(a, b) => vec![a, b],
            Self::Call(_, args) => args.iter().collect(),
            _ => Vec::new(),
        };
        for c in &children {
            if let Some(why) = c.undefined(g, clock) {
                return Some(why);
            }
        }
        let value: f64 = self.eval(g, clock);
        if value.is_finite() {
            return None;
        }
        let args: Vec<f64> = children.iter().map(|c| c.eval(g, clock)).collect();
        if args.iter().any(|v| !v.is_finite()) {
            return None;
        }
        Some(match self {
            Self::Cell(f) => format!(
                "{}() is a function of the lattice: it needs a periodic cell and an evaluation \
                 that sees the lattice (ExpressionBias::into_term)",
                f.name()
            ),
            Self::Pow(..) => format!(
                "{}^{} has no real value (a negative number to a non-integer power, or zero to \
                 a power that is not positive)",
                args[0], args[1]
            ),
            Self::Bin(Op::Div, ..) => format!("{} / {} is not finite", args[0], args[1]),
            Self::Bin(..) => format!("an operation on {} and {} is not finite", args[0], args[1]),
            Self::Call(f, _) => format!(
                "{}({}) is not defined",
                func_name(*f),
                args.iter()
                    .map(f64::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            _ => "a value is not finite".to_string(),
        })
    }

    fn atoms(&self, out: &mut Vec<AtomId>) {
        match self {
            Self::Num(_) | Self::Iteration | Self::Time | Self::Var(_) | Self::Cell(_) => {}
            Self::Neg(a) => a.atoms(out),
            Self::Bin(_, a, b) | Self::Pow(a, b) => {
                a.atoms(out);
                b.atoms(out);
            }
            Self::Call(_, args) => args.iter().for_each(|a| a.atoms(out)),
            Self::Geo(_, groups) => groups.iter().for_each(|g| out.extend(g)),
        }
    }

    fn max_var(&self) -> Option<usize> {
        match self {
            Self::Var(k) => Some(*k),
            Self::Num(_) | Self::Iteration | Self::Time | Self::Geo(..) | Self::Cell(_) => None,
            Self::Neg(a) => a.max_var(),
            Self::Bin(_, a, b) | Self::Pow(a, b) => a.max_var().max(b.max_var()),
            Self::Call(_, args) => args.iter().filter_map(Node::max_var).max(),
        }
    }

    fn eval<T: Scalar, E: Env<T>>(&self, g: &E, clock: Clock) -> T {
        match self {
            Self::Num(v) => T::from_f64(*v),
            Self::Iteration => T::from_f64(clock.iteration as f64),
            Self::Time => T::from_f64(clock.time_au.unwrap_or(clock.iteration as f64)),
            Self::Neg(a) => -a.eval(g, clock),
            Self::Bin(op, a, b) => {
                let (a, b) = (a.eval(g, clock), b.eval(g, clock));
                match op {
                    Op::Add => a + b,
                    Op::Sub => a - b,
                    Op::Mul => a * b,
                    Op::Div => a / b,
                }
            }
            Self::Pow(a, b) => {
                let base = a.eval(g, clock);
                match **b {
                    Self::Num(p) if p.fract() == 0.0 && p.abs() <= 64.0 => base.powi(p as i32),
                    Self::Num(p) => base.powf(p),
                    _ => variable_power(base, b.eval(g, clock)),
                }
            }
            Self::Call(f, args) => {
                let x = args[0].eval(g, clock);
                match f {
                    Func::Exp => x.exp(),
                    Func::Ln => x.ln(),
                    Func::Sqrt => x.sqrt(),
                    Func::Sin => x.sin(),
                    Func::Cos => x.cos(),
                    Func::Tan => x.sin() / x.cos(),
                    // The library tanh saturates at +-1 instead of forming
                    // (e^2x - 1) / (e^2x + 1), which is inf / inf beyond x ~ 354.
                    Func::Tanh => x.tanh(),
                    Func::Acos => x.acos(),
                    Func::Abs => pick(x.val() < 0.0, -x, x),
                    Func::Atan2 => x.atan2(args[1].eval(g, clock)),
                    Func::Min => {
                        let y = args[1].eval(g, clock);
                        pick(x.val() <= y.val(), x, y)
                    }
                    Func::Max => {
                        let y = args[1].eval(g, clock);
                        pick(x.val() >= y.val(), x, y)
                    }
                }
            }
            Self::Geo(f, gr) => g.geo(*f, gr),
            // Without the lattice (an evaluation of the atoms alone) the value is undefined,
            // which `undefined` reports.
            Self::Cell(f) => match g.lattice() {
                Some(l) => f.value(&l),
                None => T::from_f64(f64::NAN),
            },
            Self::Var(k) => g.var(*k),
        }
    }
}

/// Bias potential given by an expression.
#[derive(Debug, Clone, PartialEq)]
pub struct ExpressionBias {
    source: String,
    label: String,
    root: Node,
    atoms: Vec<AtomId>,
}

impl ExpressionBias {
    /// Parse `source`.
    pub fn parse(source: &str) -> MoptResult<Self> {
        let mut p = Parser {
            toks: lex(source)?,
            pos: 0,
            src: source,
            variables: false,
        };
        let root = p.expr()?;
        if p.pos != p.toks.len() {
            return Err(p.err("unexpected trailing input"));
        }
        let mut atoms = Vec::new();
        root.atoms(&mut atoms);
        atoms.sort_unstable();
        atoms.dedup();
        Ok(Self {
            source: source.to_string(),
            label: "user_bias_expression".to_string(),
            root,
            atoms,
        })
    }

    /// The same expression reported under another name (e.g. `user_constraint`).
    pub fn named(mut self, label: impl Into<String>) -> Self {
        self.label = label.into();
        self
    }

    /// The expression as given.
    pub fn source(&self) -> &str {
        &self.source
    }

    /// The value when the expression has no geometry, iteration, or time (`1.5 * ang`).
    pub fn constant_value(&self) -> Option<f64> {
        self.root.constant()
    }

    /// Whether the expression uses Cartesian coordinates (`x`, `y`, `z`), so that it depends on
    /// the position and orientation in space.
    pub fn uses_cartesian(&self) -> bool {
        self.root.uses_cartesian()
    }

    /// Whether the expression uses a function of the cell (`cell_a()`, `volume()`, ...).
    pub fn uses_cell(&self) -> bool {
        self.root.any_leaf(&|n| matches!(n, Node::Cell(_)))
    }

    /// Whether the expression uses a Cartesian component of a lattice vector (`cell_ax()`,
    /// ...), so that it depends on the orientation of the cell in space.
    pub fn uses_lattice_components(&self) -> bool {
        self.root
            .any_leaf(&|n| matches!(n, Node::Cell(CellFn::Component(..))))
    }

    /// Whether the value depends on the atoms (a geometric function of them).
    pub fn uses_atoms(&self) -> bool {
        !self.atoms.is_empty()
    }

    /// The expression as a surface term: [`AdBias`] for an expression of the atoms alone,
    /// [`CellExpression`] for one that uses the cell, which differentiates along the lattice
    /// as well.
    pub fn into_term(self) -> Arc<dyn PesTerm> {
        if self.uses_cell() {
            Arc::new(CellExpression::new(self))
        } else {
            Arc::new(AdBias::new(self))
        }
    }
}

/// An expression of the atoms and the cell (`-uc "cell_a() = 10.5*ang"`, `-uc "volume()"`, `-uc
/// "dist(1,2) / cell_c()"`) as a surface term, differentiated exactly by automatic
/// differentiation in the coordinates of its atoms and in the nine components of the strain
/// `eps` of the cell (`r -> (1 + eps) r` for every position and lattice vector, the minimum-
/// image shifts included; [`Geom::strained`]):
///
/// - the Cartesian gradient at a fixed lattice;
/// - the stress `sigma_ab = (1/V) dF/d eps_ab`, not made symmetric: a Cartesian component of a
///   lattice vector depends on the orientation of the cell, and its stress has an
///   antisymmetric part (the gradient with respect to the lattice vectors is
///   [`crate::opt::cell::lattice_gradient`] of it);
/// - with a Hessian asked for, the Cartesian Hessian and the second derivatives along the
///   strain ([`StrainHessian`]: of the strain, and of the Cartesian gradient), from one pass
///   over the coordinates and the strain together.
///
/// A structure without a periodic cell has no lattice: an error naming the expression.
///
/// MultiOptPy: no counterpart (new in mopt-rs).
#[derive(Debug, Clone)]
pub struct CellExpression {
    expr: ExpressionBias,
}

impl CellExpression {
    /// The term of `expr`.
    pub fn new(expr: ExpressionBias) -> Self {
        Self { expr }
    }

    /// The value at the coordinates `v[..m]` of the atoms and the strain `v[m..m + 9]`
    /// (`eps_ab` at `m + 3 a + b`), with `values` the coordinates of the atoms for the
    /// minimum-image shifts and `lattice` the lattice vectors (rows) at no strain.
    fn value_at<T: Scalar>(
        &self,
        atoms: &[AtomId],
        values: &[f64],
        cell: &crate::domain::PeriodicCell,
        lattice: &[[f64; 3]; 3],
        v: &[T],
        clock: Clock,
    ) -> T {
        let m = values.len();
        let e: [[T; 3]; 3] = std::array::from_fn(|a| std::array::from_fn(|b| v[m + 3 * a + b]));
        let strain = |p: [T; 3]| -> [T; 3] {
            std::array::from_fn(|a| {
                let mut out = p[a];
                for (b, pb) in p.iter().enumerate() {
                    out += e[a][b] * *pb;
                }
                out
            })
        };
        let xs: Vec<T> = (0..m / 3)
            .flat_map(|k| strain([v[3 * k], v[3 * k + 1], v[3 * k + 2]]))
            .collect();
        let strained: [[T; 3]; 3] = std::array::from_fn(|k| strain(lattice[k].map(T::from_f64)));
        let env = CellGeom {
            geom: Geom::strained(atoms, &xs, values, Some(cell), e),
            lattice: strained,
        };
        self.expr.root.eval(&env, clock)
    }
}

impl PesTerm for CellExpression {
    fn name(&self) -> &str {
        &self.expr.label
    }

    fn atoms(&self) -> Option<Vec<AtomId>> {
        Some(self.expr.atoms.clone())
    }

    fn evaluate(
        &self,
        sys: &MolecularSystem,
        ctx: &TermContext<'_>,
        need_hessian: bool,
    ) -> MoptResult<EvalOutput> {
        let cell = sys.active_cell().ok_or_else(|| {
            MoptError::config(format!(
                "'{}': a function of the cell needs a structure in a periodic cell",
                self.expr.label
            ))
        })?;
        let (atoms, x) = local_coordinates(self.expr.atoms.clone(), sys)?;
        let lattice: [[f64; 3]; 3] = std::array::from_fn(|k| cell.lattice_vector(k).value());
        let clock = Clock::of(ctx);
        let m = x.len();
        // The coordinates of the atoms and the strain (zero) as the variables.
        let z: Vec<f64> = x.iter().copied().chain([0.0; 9]).collect();
        let (value, g, h) = if need_hessian {
            let (value, g, h) = eval_value_grad_hessian(&z, None, |v: &[HyperDual]| {
                self.value_at(&atoms, &x, cell, &lattice, v, clock)
            });
            (value, g, Some(h))
        } else {
            let (value, g) = eval_value_and_gradient(&z, |v: &[Dual]| {
                self.value_at(&atoms, &x, cell, &lattice, v, clock)
            });
            (value, g, None)
        };
        let local: Vec<f64> = (0..m).map(|i| g[i]).collect();
        let local_hessian = h.as_ref().map(|h| Mat::from_fn(m, m, |i, j| h[(i, j)]));
        let (gradient_bohr, hessian_bohr) =
            scatter_local(sys, &atoms, &local, local_hessian.as_ref());
        let volume = cell.volume_bohr3();
        let stress: [[f64; 3]; 3] =
            std::array::from_fn(|a| std::array::from_fn(|b| g[m + 3 * a + b] / volume));
        // d/d eps_cd of the Cartesian gradient: the mixed second derivative in the variables
        // less the part that comes from the strain of the coordinates themselves
        // (d/d eps_cd of (1 + eps)_ab' gives delta_ac delta_b'd dF/dr_ia).
        let strain_hessian = h.as_ref().map(|h| {
            let mut out = StrainHessian::zeros(sys.dof());
            for p in 0..9 {
                for q in 0..9 {
                    out.strain[p][q] = h[(m + p, m + q)];
                }
            }
            for (slot, atom) in atoms.iter().enumerate() {
                for b in 0..3 {
                    for q in 0..9 {
                        let (c, d) = (q / 3, q % 3);
                        let own = if b == d { g[3 * slot + c] } else { 0.0 };
                        out.mixed[(atom.dof(b), q)] = h[(3 * slot + b, m + q)] - own;
                    }
                }
            }
            out
        });
        let out = EvalOutput {
            energy: Quantity::new(value),
            gradient_bohr,
            hessian_bohr,
            orbitals: None,
            stress: Some(stress),
            strain_hessian,
        };
        if let Err(e) = check_term_output(&self.expr.label, &out) {
            let fixed: [[f64; 3]; 3] = lattice;
            let env = CellGeom {
                geom: Geom::new(&atoms, &x, &x, Some(cell)),
                lattice: fixed,
            };
            return Err(match self.expr.root.undefined(&env, clock) {
                Some(why) => MoptError::calc(format!("'{}': {why}", self.expr.label)),
                None => e,
            });
        }
        Ok(out)
    }

    fn frame(&self) -> Frame {
        self.expr.frame()
    }

    fn history(&self) -> History {
        self.expr.history()
    }

    fn analytic_hessian(&self) -> bool {
        true
    }

    fn analytic_stress(&self) -> bool {
        true
    }
}

/// An objective of plain variables `q1`, `q2`, ... written as an expression (same grammar and
/// functions as the biases, without geometry); exact derivatives by automatic differentiation.
#[derive(Debug, Clone, PartialEq)]
pub struct VariableExpression {
    source: String,
    root: Node,
    used: usize,
}

impl VariableExpression {
    /// Parse `source`.
    pub fn parse(source: &str) -> MoptResult<Self> {
        let mut p = Parser {
            toks: lex(source)?,
            pos: 0,
            src: source,
            variables: true,
        };
        let root = p.expr()?;
        if p.pos != p.toks.len() {
            return Err(p.err("unexpected trailing input"));
        }
        let used = root.max_var().map_or(0, |k| k + 1);
        Ok(Self {
            source: source.to_string(),
            root,
            used,
        })
    }

    /// The expression as given.
    pub fn source(&self) -> &str {
        &self.source
    }

    /// The highest variable used (`q3` -> 3; 0 when none).
    pub fn variables_used(&self) -> usize {
        self.used
    }

    /// Value at `q` (any [`Scalar`], for derivatives); `iteration` sets `it` and `t`.
    pub fn value<T: Scalar>(&self, q: &[T], iteration: usize) -> T {
        self.root.eval(&Vars(q), Clock::at_iteration(iteration))
    }
}

impl GeometricBias for ExpressionBias {
    fn name(&self) -> &str {
        &self.label
    }
    fn atoms(&self) -> Vec<AtomId> {
        self.atoms.clone()
    }
    fn energy<T: Scalar>(&self, g: &Geom<'_, T>, clock: Clock) -> T {
        self.root.eval(g, clock)
    }
    /// Cartesian coordinates (`x`, `y`, `z`) fix the expression in space; the Cartesian
    /// components of the lattice vectors fix its orientation; distances, angles, dihedrals,
    /// out-of-plane angles, and the lengths, angles, and volume of the cell do not.
    fn frame(&self) -> Frame {
        if self.uses_cartesian() {
            Frame::Absolute
        } else if self.uses_lattice_components() {
            Frame::TranslationOnly
        } else {
            Frame::Invariant
        }
    }
    /// `t` makes it time-dependent, `it` iteration-dependent.
    fn history(&self) -> History {
        if self.root.any_leaf(&|n| matches!(n, Node::Time)) {
            History::TimeDependent
        } else if self.root.any_leaf(&|n| matches!(n, Node::Iteration)) {
            History::IterationRamp
        } else {
            History::Static
        }
    }
    fn diagnose(&self, g: &Geom<'_, f64>, clock: Clock) -> Option<String> {
        self.root.undefined(g, clock)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn value(src: &str) -> f64 {
        let b = ExpressionBias::parse(src).unwrap();
        let atoms: Vec<AtomId> = Vec::new();
        b.energy(
            &Geom::<f64>::new(&atoms, &[], &[], None),
            Clock::at_iteration(3),
        )
    }

    #[test]
    fn arithmetic_precedence_and_constants() {
        assert!((value("1 + 2 * 3 ^ 2") - 19.0).abs() < 1e-14);
        assert!((value("-2^2") + 4.0).abs() < 1e-14);
        assert!((value("2 ** 3 ** 2") - 512.0).abs() < 1e-12);
        assert!((value("(1 + 2) * 3 / 4") - 2.25).abs() < 1e-14);
        assert!((value("1.5e-3 * 2E2") - 0.3).abs() < 1e-14);
        assert!((value("cos(pi) + tanh(0) + max(1, 2) + abs(-3)") - 4.0).abs() < 1e-14);
        assert!((value("ang") - Units::ANGSTROM_TO_BOHR).abs() < 1e-14);
        assert!((value("it * 2 + t") - 9.0).abs() < 1e-14);
    }

    /// Value and derivative of a one-variable objective.
    fn with_derivative(src: &str, q: f64) -> (f64, f64) {
        let e = VariableExpression::parse(src).unwrap();
        let d = e.value(&[crate::math::ad::Dual::variable(q)], 0);
        (d.re, d.eps)
    }

    #[test]
    fn tanh_saturates_instead_of_overflowing() {
        for x in [400.0_f64, 800.0, -800.0, 0.3] {
            let (v, d) = with_derivative("tanh(q1)", x);
            assert!((v - x.tanh()).abs() < 1e-15, "{x}: {v}");
            assert!((d - (1.0 - x.tanh().powi(2))).abs() < 1e-15, "{x}: {d}");
        }
    }

    #[test]
    fn powers_with_a_variable_exponent_are_real_or_reported() {
        // Positive base: exp(p ln b); d/dq of 2^q = 2^q ln 2.
        let (v, d) = with_derivative("2^q1", 3.0);
        assert!((v - 8.0).abs() < 1e-14 && (d - 8.0 * 2f64.ln()).abs() < 1e-13);
        // A negative base to an exponent that is an integer: the real branch.
        let (v, _) = with_derivative("(-2)^(q1 + 1)", 2.0);
        assert!((v + 8.0).abs() < 1e-13, "{v}");
        let (v, _) = with_derivative("(-2)^(q1 + 1)", 1.0);
        assert!((v - 4.0).abs() < 1e-13, "{v}");
        let (v, _) = with_derivative("0^(q1)", 2.0);
        assert_eq!(v, 0.0);
        // No real value: NaN, which the bias reports with the operation.
        assert!(with_derivative("(-2)^(q1)", 0.5).0.is_nan());
        let b = ExpressionBias::parse("(x(1) - 5)^(0.1 * dist(1, 2))").unwrap();
        let atoms = [AtomId::new(0), AtomId::new(1)];
        let x = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let g = Geom::<f64>::new(&atoms, &x, &x, None);
        assert!(b.energy(&g, Clock::at_iteration(0)).is_nan());
        let why = b.diagnose(&g, Clock::at_iteration(0)).unwrap();
        assert!(
            why.contains("-5^0.1") && why.contains("non-integer"),
            "{why}"
        );
    }

    #[test]
    fn the_angle_function_is_accurate_near_180_degrees() {
        let b = ExpressionBias::parse("angle(1, 2, 3)").unwrap();
        let atoms = [AtomId::new(0), AtomId::new(1), AtomId::new(2)];
        let y = 3e-9;
        let x = [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, y, 0.0];
        let theta = b.energy(
            &Geom::<f64>::new(&atoms, &x, &x, None),
            Clock::at_iteration(0),
        );
        assert!((theta - (PI - y.atan())).abs() < 1e-15, "{theta}");
    }

    /// Two argon atoms in a skewed cell.
    fn in_a_cell() -> MolecularSystem {
        use crate::domain::{positions, Angstrom, Element, PeriodicCell, SpinCharge, Vector3};
        let cell = PeriodicCell::from_lattice_vectors(
            Vector3::<Angstrom>::new([5.1, 0.2, -0.1]),
            Vector3::<Angstrom>::new([0.4, 5.3, 0.3]),
            Vector3::<Angstrom>::new([-0.3, 0.2, 5.6]),
            [true; 3],
        )
        .unwrap();
        MolecularSystem::new(
            vec![Element::parse_token("Ar").unwrap(); 2],
            &positions::<Angstrom>(&[[0.3, 0.2, 0.1], [4.6, 0.5, 0.4]]),
            SpinCharge::default(),
            Some(cell),
        )
        .unwrap()
    }

    #[test]
    fn functions_of_the_cell_have_their_values_and_exact_derivatives() {
        use crate::pes::stress::{numerical_strain_hessian, strained};
        let sys = in_a_cell();
        let cell = sys.active_cell().unwrap();
        let l: [[f64; 3]; 3] = std::array::from_fn(|k| cell.lattice_vector(k).value());
        let ctx = TermContext::at_iteration(0, &sys);
        let value = |src: &str, s: &MolecularSystem| {
            ExpressionBias::parse(src)
                .unwrap()
                .into_term()
                .evaluate(s, &TermContext::at_iteration(0, s), false)
                .unwrap()
                .energy
                .value()
        };
        let len = |v: [f64; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        assert!((value("cell_b()", &sys) - len(l[1])).abs() < 1e-12);
        assert!((value("volume()", &sys) - cell.volume_bohr3()).abs() < 1e-9);
        assert!((value("cell_cz()", &sys) - l[2][2]).abs() < 1e-15);
        let dot: f64 = (0..3).map(|c| l[0][c] * l[1][c]).sum();
        let gamma = (dot / (len(l[0]) * len(l[1]))).acos();
        assert!((value("cell_gamma()", &sys) - gamma).abs() < 1e-12);
        // A function of the atoms and of the cell, its orientation included.
        let src = "dist(1,2) * cell_a() + volume() / 1000 + cell_cz() * cell_gamma() + cell_by()";
        let term = ExpressionBias::parse(src).unwrap().into_term();
        assert_eq!(term.frame(), Frame::TranslationOnly);
        let out = term.evaluate(&sys, &ctx, true).unwrap();
        let f = |s: &MolecularSystem| value(src, s);
        let h = 1e-5;
        let x = sys.coords_flat_bohr();
        for i in 0..6 {
            let at = |d: f64| {
                let mut y = x.clone();
                y[i] += d;
                f(&sys.with_flat_coords_bohr(&y).unwrap())
            };
            let numeric = (at(h) - at(-h)) / (2.0 * h);
            assert!((numeric - out.gradient_bohr[i]).abs() < 1e-7, "x {i}");
        }
        // The stress along every (not only the symmetric) strain component.
        let stress = out.stress.unwrap();
        let volume = cell.volume_bohr3();
        for p in 0..9 {
            let at = |d: f64| {
                let mut e = [[0.0; 3]; 3];
                e[p / 3][p % 3] = d;
                f(&strained(&sys, &e).unwrap())
            };
            let numeric = (at(h) - at(-h)) / (2.0 * h) / volume;
            let analytic = stress[p / 3][p % 3];
            assert!(
                (numeric - analytic).abs() < 1e-7 * (1.0 + analytic.abs()),
                "strain {p}: {numeric} vs {analytic}"
            );
        }
        assert!((stress[0][1] - stress[1][0]).abs() > 1e-6, "{stress:?}");
        // The second derivatives along the strain and of the gradient.
        let sh = out.strain_hessian.unwrap();
        let numeric = numerical_strain_hessian(&sys, 1e-4, &|s| {
            let o = term
                .evaluate(s, &TermContext::at_iteration(0, s), false)
                .unwrap();
            Ok((o.energy.value(), o.gradient_bohr))
        })
        .unwrap();
        for p in 0..9 {
            for q in 0..9 {
                let (a, b) = (sh.strain[p][q], numeric.strain[p][q]);
                assert!((a - b).abs() < 1e-4 * (1.0 + a.abs()), "{p} {q}: {a} {b}");
            }
            for i in 0..6 {
                let (a, b) = (sh.mixed[(i, p)], numeric.mixed[(i, p)]);
                assert!((a - b).abs() < 1e-6 * (1.0 + a.abs()), "{i} {p}: {a} {b}");
            }
        }
        // Without a cell, an error naming the expression; as a bias of atoms alone
        // (AdBias), the reason.
        let free = MolecularSystem::new(
            sys.elements.clone(),
            &crate::domain::positions::<crate::domain::Bohr>(&[[0.0; 3], [1.0, 0.0, 0.0]]),
            crate::domain::SpinCharge::default(),
            None,
        )
        .unwrap();
        let err = ExpressionBias::parse("volume()")
            .unwrap()
            .into_term()
            .evaluate(&free, &TermContext::at_iteration(0, &free), false)
            .unwrap_err()
            .to_string();
        assert!(err.contains("periodic cell"), "{err}");
        let err = AdBias::new(ExpressionBias::parse("cell_a()").unwrap())
            .evaluate(&sys, &ctx, false)
            .unwrap_err()
            .to_string();
        assert!(err.contains("cell_a()") && err.contains("lattice"), "{err}");
        // Cell functions take no arguments and are not variables of objectives.
        assert!(ExpressionBias::parse("cell_a(1)").is_err());
        assert!(VariableExpression::parse("volume()").is_err());
        assert!(ExpressionBias::parse("cell_d()").is_err());
        assert!(!ExpressionBias::parse("cell_a() + dist(1,2)")
            .unwrap()
            .uses_lattice_components());
    }

    #[test]
    fn errors_are_reported() {
        for bad in [
            "1 +",
            "foo(1)",
            "dist(1)",
            "dist(1, [2-)",
            "2 $ 3",
            "dist(0, 1)",
            "q",
        ] {
            assert!(ExpressionBias::parse(bad).is_err(), "{bad}");
        }
        let b = ExpressionBias::parse("dist([1-3], 5) + angle(1, 2, 4)").unwrap();
        assert_eq!(b.atoms().len(), 5);
    }
}
