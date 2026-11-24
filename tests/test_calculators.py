"""The calculator layer: electronic states, units, finite differences, ASE mapping, registry,
and the Rust components exposed to Python. Engines of optional programs are tested when the
program is installed.

This file is part of mopt-py (the Python interface of mopt-rs).
"""

from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np
import pytest

from mopt_rs import _core
from mopt_rs.calculators import (
    ASE_UNITS,
    AseEngine,
    CalculatorError,
    CalculatorOptions,
    Capabilities,
    ElectronicState,
    Engine,
    EngineCalculator,
    EngineResult,
    SpinConvention,
    StateMapping,
    Structure,
    available,
    create,
    create_engine,
)
from mopt_rs.calculators.base import BOHR_TO_ANGSTROM, HARTREE_TO_EV
from mopt_rs.optimize import NativeCalculator, Optimizer, run

WATER = np.array([[0.0, 0.0, 0.2217], [0.0, 1.4309, -0.8867], [0.0, -1.4309, -0.8867]])


def has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def test_electronic_state_conventions() -> None:
    triplet = ElectronicState(0, 3)
    assert triplet.spin_as(SpinConvention.MULTIPLICITY) == 3
    assert triplet.spin_as(SpinConvention.UNPAIRED) == 2
    assert triplet.spin_as(SpinConvention.SPIN) == 1.0
    for conv in SpinConvention:
        back = ElectronicState.from_spin(0, triplet.spin_as(conv), conv)
        assert back == triplet
    with pytest.raises(ValueError):
        ElectronicState(0, 0)
    with pytest.raises(ValueError):
        ElectronicState(0, 2).check_electrons([8, 1, 1])
    ElectronicState(1, 2).check_electrons([8, 1, 1])


class HarmonicEv(Engine):
    """E = k/2 |x - x0|^2 in eV and Angstrom, without its own Hessian."""

    name = "harmonic"
    units = ASE_UNITS
    capabilities = Capabilities(rigid_body_invariant=False)

    def __init__(self, x0_angstrom: np.ndarray, k: float = 2.0) -> None:
        self.x0 = x0_angstrom
        self.k = k
        self.calls = 0

    def compute(self, structure: Structure, hessian: bool) -> EngineResult:
        self.calls += 1
        d = structure.positions_angstrom - self.x0
        return EngineResult(0.5 * self.k * float(np.sum(d * d)), forces=-self.k * d)


class EnergyAlone(HarmonicEv):
    """The harmonic well of an engine that can leave the gradient out."""

    def __init__(self, x0_angstrom: np.ndarray) -> None:
        super().__init__(x0_angstrom)
        self.without_gradient = 0

    def compute(  # type: ignore[override]
        self, structure: Structure, hessian: bool, gradient: bool = True
    ) -> EngineResult:
        if not gradient:
            self.calls += 1
            self.without_gradient += 1
            d = structure.positions_angstrom - self.x0
            return EngineResult(0.5 * self.k * float(np.sum(d * d)))
        return super().compute(structure, hessian)


def test_backend_recentering_preserves_the_neb_image_and_numeric_hessian() -> None:
    class CenteredWell(HarmonicEv):
        def compute(self, structure: Structure, hessian: bool) -> EngineResult:
            structure.positions_bohr[:] -= structure.positions_bohr.mean(axis=0)
            return super().compute(structure, hessian)

    origin = WATER - WATER.mean(axis=0)
    engine = CenteredWell(origin * BOHR_TO_ANGSTROM)
    calc = EngineCalculator(engine, check_electrons=False)
    coords = WATER + [34.0, -18.0, 25.0]
    image = Structure.build(["O", "H", "H"], coords.copy())
    result = calc.evaluate_structure(image, need_hessian=True)
    np.testing.assert_array_equal(image.positions_bohr, coords)
    assert result.hessian is not None
    hessian = np.asarray(result.hessian).reshape(9, 9)
    translation = np.tile([1.0, 0.0, 0.0], 3)
    np.testing.assert_allclose(hessian @ translation, 0.0, atol=1e-8)
    assert np.linalg.norm(hessian) > 0.0


@pytest.mark.parametrize("translate_field", [True, False])
def test_embedded_backend_recentering_keeps_model_and_field_together(translate_field: bool) -> None:
    class CenteredField(HarmonicEv):
        capabilities = Capabilities(point_charges=True, rigid_body_invariant=False)

        def compute_embedded(self, structure, positions_bohr, charges, elements, hessian):
            center = structure.positions_bohr.mean(axis=0)
            structure.positions_bohr[:] -= center
            if translate_field:
                positions_bohr[:] -= center
            return self.compute(structure, hessian), np.zeros_like(positions_bohr)

    calc = EngineCalculator(CenteredField(np.zeros_like(WATER)), check_electrons=False)
    coords = WATER + [34.0, -18.0, 25.0]
    image = Structure.build(["O", "H", "H"], coords.copy())
    field = coords[:1].copy() + [5.0, 0.0, 0.0]
    original_field = field.copy()
    if translate_field:
        calc.evaluate_embedded(image, field, [1.0], [None])
    else:
        with pytest.raises(CalculatorError, match="changed frames separately"):
            calc.evaluate_embedded(image, field, [1.0], [None])
    np.testing.assert_array_equal(image.positions_bohr, coords)
    np.testing.assert_array_equal(field, original_field)


def test_an_engine_gives_the_energy_alone_when_asked() -> None:
    """Surface hopping asks for the energies of the other states only: an engine that
    declares ``gradient`` is called without it, and a later request for the gradient at the
    same structure computes it."""
    engine = EnergyAlone(WATER * BOHR_TO_ANGSTROM)
    calc = EngineCalculator(engine, check_electrons=False)
    s = Structure.build(["O", "H", "H"], WATER + 0.1)
    alone = calc.evaluate_structure(s, need_gradient=False)
    assert engine.without_gradient == 1
    assert np.all(np.asarray(alone.gradient) == 0.0)
    full = calc.evaluate_structure(s)
    assert engine.calls == 2 and engine.without_gradient == 1
    assert full.energy == pytest.approx(alone.energy)
    assert np.abs(np.asarray(full.gradient)).max() > 0.0
    # The full result answers a later request for the energy alone.
    calc.evaluate_structure(s, need_gradient=False)
    assert engine.calls == 2
    # An engine without the parameter is asked for everything.
    plain = HarmonicEv(WATER * BOHR_TO_ANGSTROM)
    EngineCalculator(plain, check_electrons=False).evaluate_structure(s, need_gradient=False)
    assert plain.calls == 1


def test_units_finite_difference_hessian_and_cache() -> None:
    x0 = WATER * BOHR_TO_ANGSTROM
    engine = HarmonicEv(x0)
    calc = EngineCalculator(engine, check_electrons=False)
    s = Structure.build(["O", "H", "H"], WATER + 0.1)
    e = calc.evaluate_structure(s)
    k_au = 2.0 / HARTREE_TO_EV * BOHR_TO_ANGSTROM**2
    assert np.isclose(e.energy, 0.5 * k_au * 9 * 0.01, rtol=1e-12)
    assert np.allclose(e.gradient, k_au * 0.1, rtol=1e-12)
    calls = engine.calls
    h = calc.evaluate_structure(s, need_hessian=True).hessian
    assert h is not None and np.allclose(h, k_au * np.eye(9), atol=1e-9)
    assert engine.calls == calls + 18, "the gradient is reused; only displacements run"
    calc.evaluate_structure(s, need_hessian=True)
    assert engine.calls == calls + 18, "cached"


def test_errors_carry_the_engine_name() -> None:
    class Broken(HarmonicEv):
        def compute(self, structure: Structure, hessian: bool) -> EngineResult:
            raise RuntimeError("SCF exploded")

    calc = EngineCalculator(Broken(WATER), check_electrons=False)
    with pytest.raises(CalculatorError, match="harmonic failed"):
        calc.evaluate_structure(Structure.build(["O", "H", "H"], WATER))
    calc = EngineCalculator(HarmonicEv(WATER))
    with pytest.raises(CalculatorError, match="impossible"):
        calc.evaluate_structure(Structure.build(["O", "H", "H"], WATER, multiplicity=2))


class FakeAseCalc:
    """Records how it was built; behaves like an ASE calculator for Atoms.get_*."""

    built: list[dict[str, Any]] = []

    def __init__(self, charge: int = 0, uhf: int = 0, method: str = "x") -> None:
        FakeAseCalc.built.append({"charge": charge, "uhf": uhf, "method": method})


@pytest.mark.skipif(not has("ase"), reason="ASE not installed")
def test_ase_state_mapping_is_checked() -> None:
    uhf = StateMapping("charge", "uhf", SpinConvention.UNPAIRED)
    eng = AseEngine(FakeAseCalc, kwargs={"method": "pm6"}, state=uhf)
    eng.calculator(ElectronicState(-1, 3))
    assert FakeAseCalc.built[-1] == {"charge": -1, "uhf": 2, "method": "pm6"}
    with pytest.raises(TypeError, match="multiplicity"):
        AseEngine(FakeAseCalc, state=StateMapping("charge", "multiplicity"))
    info = AseEngine(FakeAseCalc, state=StateMapping("charge", "spin", via="info"))
    atoms = info.atoms(Structure.build(["O", "H"], WATER[:2], multiplicity=2))
    assert atoms.info == {"charge": 0, "spin": 2}


class _Mult:
    def __init__(self, charge: int = 0, multiplicity: int = 1) -> None: ...


class _AseStyle:
    default_parameters = {"charge": 0.0, "multiplicity": 1}

    def __init__(self, **kwargs: Any) -> None: ...


class _Documented:
    """Parameters: chrg (total charge), uhf (number of unpaired electrons)."""

    def __init__(self, **kwargs: Any) -> None: ...


class _Nothing:
    def __init__(self, model: str = "x") -> None: ...


def test_state_keywords_are_detected() -> None:
    from mopt_rs.calculators.ase import detect_state

    assert detect_state(_Mult) == StateMapping("charge", "multiplicity")
    assert detect_state(_AseStyle) == StateMapping("charge", "multiplicity")
    assert detect_state(_Documented) == StateMapping("chrg", "uhf", SpinConvention.UNPAIRED)
    assert detect_state(FakeAseCalc) == StateMapping("charge", "uhf", SpinConvention.UNPAIRED)
    assert detect_state(_Nothing).via == "info"


# Rust ASE engines exercised without any mopt-rs code of their own (``ase:module:Class``).
_ASE_ENGINES = [
    "pm6_rs.ase:PM6",
    "pm3_rs.ase:PM3",
    "pm7_rs.ase:PM7",
    "am1_rs.ase:AM1",
    "gfn0_rs:GFN0RSCalculator",
    "gfn1_rs:GFN1RSCalculator",
    "gfn2_rs:GFN2RSCalculator",
    "dftb_rs.calculator:DFTB",
]


@pytest.mark.parametrize("path", _ASE_ENGINES)
def test_any_ase_engine_works_without_code(path: str) -> None:
    if not has(path.split(":")[0].split(".")[0]):
        pytest.skip(f"{path} not installed")
    calc = create(f"ase:{path}")
    ch2 = [[0.0, 0.0, 0.0], [2.05, 0.0, 0.0], [-0.65, 1.94, 0.0]]

    def state(mult: int) -> Structure:
        return Structure.build(["C", "H", "H"], ch2, multiplicity=mult)

    singlet = calc.evaluate_structure(state(1), need_hessian=True)
    assert singlet.hessian is not None and np.allclose(singlet.hessian, singlet.hessian.T)
    triplet = calc.evaluate_structure(state(3), need_hessian=True)
    assert abs(triplet.energy - singlet.energy) > 1e-3, "the multiplicity reaches the engine"
    # A calculator instance (built for a singlet) is switched to the requested state.
    from mopt_rs.calculators.ase import resolve_object

    inst = create(resolve_object(path)())
    e = inst.evaluate_structure(state(3))
    assert np.isclose(e.energy, triplet.energy, atol=1e-6)


@pytest.mark.skipif(not has("ase"), reason="ASE not installed")
def test_ase_engine_with_a_real_ase_calculator() -> None:
    from ase.calculators.lj import LennardJones

    eng = AseEngine(
        LennardJones,
        kwargs={"sigma": 3.4, "epsilon": 0.0104},
        state=StateMapping(None, None),
        hessian=None,
        electrons=False,
    )
    calc = EngineCalculator(eng)
    s = Structure.build(["Ar", "Ar"], [[0.0, 0.0, 0.0], [7.5, 0.0, 0.0]])
    ev = calc.evaluate_structure(s, need_hessian=True)
    xp = s.displaced(3, 1e-4)
    xm = s.displaced(3, -1e-4)
    fd = (calc.evaluate_structure(xp).energy - calc.evaluate_structure(xm).energy) / 2e-4
    assert np.isclose(ev.gradient[1, 0], fd, rtol=1e-6)
    assert ev.hessian is not None and np.allclose(ev.hessian, ev.hessian.T)


def test_registry_and_native_engines() -> None:
    names = available()
    for name in ["pyscf", "psi4", "tblite", "dxtb", "lj", "pm6_rs", "gfn2_rs", "ase"]:
        assert name in names
    with pytest.raises(KeyError):
        create("no_such_program")
    lj = create("lj")
    pos = [[0.0, 0.0, 0.0], [7.0, 0.2, 0.0], [3.5, 6.0, 0.1]]
    native = NativeCalculator("lj")
    opt_a = Optimizer(["Ar"] * 3, np.ravel(pos).tolist(), methods=["rsirfo_bfgs"])
    opt_b = Optimizer(["Ar"] * 3, np.ravel(pos).tolist(), methods=["rsirfo_bfgs"])
    ra, rb = run(opt_a, lj), run(opt_b, native)
    assert ra.converged and ra.iterations == rb.iterations
    assert np.allclose(ra.positions_bohr, rb.positions_bohr, atol=1e-12)
    o = CalculatorOptions(memory="2GB", sub_basis={"I": "lanl2dz"}, basis="6-31g*")
    assert o.memory_mb == 2048.0
    assert o.basis_spec() == {"default": "6-31g*", "I": "lanl2dz"}


def test_rust_components_from_python() -> None:
    sym = ["O", "H", "H"]
    pos = WATER.ravel().tolist()
    assert _core.atomic_numbers(sym) == [8, 1, 1]
    assert len(_core.model_hessian_names()) > 10
    h = np.asarray(_core.model_hessian("lindh2007d3", sym, pos)).reshape(9, 9)
    assert np.allclose(h, h.T, atol=1e-12)
    translations = np.tile(np.eye(3), (3, 1))
    assert np.allclose(h @ translations, 0.0, atol=1e-8), "translation invariant"
    e, g, hd = _core.dispersion("d3", sym, pos, need_hessian=True)
    gp = _core.dispersion("d3", sym, (WATER.ravel() + 1e-5 * np.eye(9)[4]).tolist())[0]
    gm = _core.dispersion("d3", sym, (WATER.ravel() - 1e-5 * np.eye(9)[4]).tolist())[0]
    assert np.isclose(g[4], (gp - gm) / 2e-5, rtol=1e-5, atol=1e-10)
    assert hd is not None and len(hd) == 81
    props = _core.dispersion_properties(sym, pos)
    assert len(props["d3_coordination_numbers"]) == 3 and abs(sum(props["d4_charges"])) < 1e-10
    c = _core.constraint_values(["dist(1,2) = dist(1,3)", "angle(2,1,3) = 104.5*deg"], sym, pos)
    assert abs(c["residuals"][0]) < 1e-12 and len(c["jacobian"]) == 18


@pytest.mark.skipif(not has("pyscf"), reason="PySCF not installed")
def test_pyscf_engine_states_and_derivatives() -> None:
    calc = create("pyscf", CalculatorOptions(method="hf", basis="sto-3g"))
    s = Structure.build(["O", "H", "H"], WATER)
    ev = calc.evaluate_structure(s, need_hessian=True)
    xp, xm = s.displaced(5, 1e-4), s.displaced(5, -1e-4)
    fd = (calc.evaluate_structure(xp).energy - calc.evaluate_structure(xm).energy) / 2e-4
    assert np.isclose(ev.gradient.ravel()[5], fd, atol=1e-6)
    assert ev.hessian is not None and np.allclose(ev.hessian, ev.hessian.T, atol=1e-8)
    # Multiplicity 3 is mol.spin = 2 in PySCF: triplet O2 lies below the singlet.
    o2 = [[0.0, 0.0, 0.0], [0.0, 0.0, 2.28]]
    t = calc.evaluate_structure(Structure.build(["O", "O"], o2, multiplicity=3)).energy
    sgl = calc.evaluate_structure(Structure.build(["O", "O"], o2, multiplicity=1)).energy
    assert t < sgl


@pytest.mark.skipif(not has("tblite"), reason="tblite not installed")
def test_tblite_engine_restarts_and_converts_spin() -> None:
    calc = create("GFN2-xTB")
    s = Structure.build(["O", "H", "H"], WATER)
    ev = calc.evaluate_structure(s, need_hessian=True)
    assert ev.hessian is not None and np.allclose(ev.hessian, ev.hessian.T, atol=1e-6)
    ch3 = [[0.0, 0.0, 0.0], [2.04, 0.0, 0.0], [-1.02, 1.77, 0.0], [-1.02, -1.77, 0.0]]
    calc.evaluate_structure(Structure.build(["C", "H", "H", "H"], ch3, multiplicity=2))


@pytest.mark.skipif(not has("pm6_rs"), reason="pm6-rs not installed")
def test_rust_engine_profile_open_shell() -> None:
    calc = create("pm6_rs")
    ch3 = [[0.0, 0.0, 0.0], [2.04, 0.0, 0.0], [-1.02, 1.77, 0.0], [-1.02, -1.77, 0.0]]
    s = Structure.build(["C", "H", "H", "H"], ch3, multiplicity=2)
    ev = calc.evaluate_structure(s, need_hessian=True)
    assert ev.hessian is not None and ev.hessian.shape == (12, 12)
    quartet = calc.evaluate_structure(Structure.build(["C", "H", "H", "H"], ch3, multiplicity=4))
    assert quartet.energy > ev.energy, "the multiplicity reaches the engine"


def _water_xyz(path: Any) -> str:
    f = path / "water.xyz"
    rows = "\n".join(
        f"{s} {x * BOHR_TO_ANGSTROM:.6f} {y * BOHR_TO_ANGSTROM:.6f} {z * BOHR_TO_ANGSTROM:.6f}"
        for s, (x, y, z) in zip(["O", "H", "H"], WATER + 0.05, strict=True)
    )
    f.write_text(f"3\n\n{rows}\n")
    return str(f)


def _optimized(directory: Any) -> list[Any]:
    return sorted(directory.rglob("*_optimized.xyz"))


def test_commands_use_python_engines(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from mopt_rs.calculators import register

    x0 = WATER * BOHR_TO_ANGSTROM
    register(
        "harmonic_test",
        lambda arg, o: HarmonicEv(x0, k=float(o.extra.get("k", 2.0))),
        replace_existing=True,
    )
    monkeypatch.chdir(tmp_path)
    xyz = _water_xyz(tmp_path)
    code = _core.run_cli("optmain", [xyz, "-os", "harmonic_test", "-calc_opt", "k=3.0", "-tcc"])
    assert code == 0
    out = _optimized(tmp_path)
    assert out, "run directory with the optimized structure"
    rows = [line.split() for line in out[0].read_text().splitlines()[2:]]
    pos = np.array([[float(v) for v in r[1:4]] for r in rows])
    assert np.allclose(pos, x0, atol=1e-3)
    log = (out[0].parent / "mopt.log").read_text()
    assert any(line.startswith("Calculator: harmonic") for line in log.splitlines()), log
    assert "software_path.conf: not found" in log
    assert _core.run_cli("optmain", [xyz, "-os", "no_such_program"]) != 0


class _DummyAse:
    """Stands for a third-party ASE calculator class."""

    def __init__(self, charge: int = 0, multiplicity: int = 1) -> None:
        self.parameters = {"charge": charge, "multiplicity": multiplicity}


def test_engines_describe_where_they_come_from() -> None:
    from mopt_rs import __version__
    from mopt_rs.calculators import AseEngine, create
    from mopt_rs.calculators._bridge import make
    from mopt_rs.calculators.provenance import module_origin

    assert create("lj").describe() == f"lj: built into mopt-rs {__version__} (Rust)"
    _, info = make("lj", {})
    assert info["description"].startswith("lj: built into mopt-rs")
    numpy_origin = module_origin("numpy")
    assert numpy_origin.startswith(f"numpy {np.__version__} (") and "numpy" in numpy_origin
    assert module_origin("no_such_package_xyz") == "no_such_package_xyz (not installed)"
    engine = AseEngine(_DummyAse, name="dummy")
    text = engine.describe()
    assert text.startswith("dummy: ") and "_DummyAse from " in text, text
    engine.notes = ("model=/data/m.pt from software_path.conf",)
    assert engine.describe().endswith(
        "; model=/data/m.pt from software_path.conf; Hessian: finite differences of the forces"
    )


@pytest.mark.skipif(not has("pyscf"), reason="PySCF not installed")
def test_optmain_defaults_to_pyscf(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    xyz = _water_xyz(tmp_path)
    assert _core.run_cli("optmain", [xyz, "-func", "hf", "-bs", "sto-3g"]) == 0
    assert _optimized(tmp_path)


@pytest.mark.parametrize("method", ["am1", "pm3", "pm6", "pm7"])
@pytest.mark.parametrize("use_profile", [False, True], ids=["ase-class", "profile"])
def test_optmain_with_any_ase_engine(
    method: str, use_profile: bool, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = f"{method}_rs"
    if not has(package):
        pytest.skip(f"{package} not installed")
    monkeypatch.chdir(tmp_path)
    xyz = _water_xyz(tmp_path)
    calculator = package if use_profile else f"ase:{package}.ase:{method.upper()}"
    assert _core.run_cli("optmain", [xyz, "-os", calculator, "-fc", "5"]) == 0
    assert _optimized(tmp_path)


def test_settings_files(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    monkeypatch.chdir(tmp_path)
    xyz = tmp_path / "ar3.xyz"
    xyz.write_text("3\n\nAr 0 0 0\nAr 3.9 0.1 0\nAr 1.8 3.3 0.2\n")
    xyz = str(xyz)
    args = [xyz, "-os", "lj", "-opt", "fire", "-elec", "0"]
    assert _core.run_cli("optmain", [*args, "--dump-config", "s.json"]) == 0
    doc = json.loads((tmp_path / "s.json").read_text())
    assert doc["INPUT"] == [xyz] and doc["othersoft"] == "lj" and doc["opt_method"] == ["fire"]
    assert not _optimized(tmp_path), "--dump-config does not run"
    assert _core.run_cli("optmain", ["--config", "s.json", "-opt", "rsirfo_bfgs"]) == 0
    runs = _optimized(tmp_path)
    assert len(runs) == 1
    kept = json.loads((runs[0].parent / "settings.json").read_text())
    assert kept["opt_method"] == ["rsirfo_bfgs"], "the command line overrides the file"
    assert kept["othersoft"] == "lj"
    (tmp_path / "bad.json").write_text('{"no_such_option": 1}')
    assert _core.run_cli("optmain", ["--config", "bad.json"]) != 0


def test_program_options_reach_the_program() -> None:
    from types import SimpleNamespace

    from mopt_rs.calculators.program_options import ProgramSetup, set_path, split_program_options

    def tag(obj: Any, label: str = "") -> None:
        obj.tags.append(label)

    plain, pre, setup = split_program_options(
        {"conv_tol": 1e-9, "scf.with_solvent.eps": 78.4, "hook": [tag], "hook.label": "x"},
        ("scf", "mol"),
    )
    assert plain == {"conv_tol": 1e-9}
    assert pre == {"scf": {"with_solvent.eps": 78.4}, "mol": {}}
    obj = SimpleNamespace(tags=[], with_solvent=SimpleNamespace(eps=1.0), opts={"a": 1})
    assert setup.run(obj) is obj and obj.tags == ["x"]
    set_path(obj, "with_solvent.eps", 78.4)
    set_path(obj, "opts.a", 2)
    assert obj.with_solvent.eps == 78.4 and obj.opts == {"a": 2}
    with pytest.raises(ValueError, match="without hook"):
        split_program_options({"hook.label": "x"})
    with pytest.raises(ImportError):
        split_program_options({"hook": "no_such_module_xyz:f"})
    # A hook may return a replacement object.
    assert ProgramSetup((lambda o: o + 1,)).run(1) == 2


@pytest.mark.skipif(not has("ase"), reason="ASE not installed")
def test_hooks_change_any_ase_calculator_from_the_command_line(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "lj_hooks.py").write_text(
        "def set_sigma(calc, sigma):\n    calc.parameters.sigma = float(sigma)\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.chdir(tmp_path)
    xyz = tmp_path / "ar2.xyz"
    xyz.write_text("2\n\nAr 0.0 0.0 0.0\nAr 3.9 0.0 0.0\n")
    args = [str(xyz), "-os", "ase:ase.calculators.lj:LennardJones", "-calc_opt"]
    args += ["sigma=3.4", "epsilon=1.0", "rc=12.0", "hook=lj_hooks:set_sigma", "hook.sigma=3.0"]
    assert _core.run_cli("optmain", args) == 0
    rows = [line.split() for line in _optimized(tmp_path)[0].read_text().splitlines()[2:]]
    pos = np.array([[float(v) for v in r[1:4]] for r in rows])
    # The minimum of the hooked potential (sigma = 3.0), not of sigma = 3.4.
    assert np.isclose(np.linalg.norm(pos[1] - pos[0]), 2 ** (1 / 6) * 3.0, atol=2e-3)


@pytest.mark.skipif(not has("pyscf"), reason="PySCF not installed")
def test_pyscf_solvent_through_program_options() -> None:
    water = Structure.build(["O", "H", "H"], WATER)
    gas = create("pyscf", CalculatorOptions(method="hf", basis="sto-3g"))
    solv = create(
        "pyscf",
        CalculatorOptions(
            method="hf",
            basis="sto-3g",
            extra={"apply": "ddCOSMO", "scf.with_solvent.eps": 78.4, "mol.cart": True},
        ),
    )
    e_gas = gas.evaluate_structure(water).energy
    ev = solv.evaluate_structure(water)
    mf = solv.last_extra()["scf"]
    assert mf.with_solvent.eps == 78.4 and mf.mol.cart
    # Solvation stabilizes water by a few kcal/mol.
    assert -0.02 < ev.energy - e_gas < -1e-4


@pytest.mark.skipif(not has("tblite"), reason="tblite not installed")
def test_tblite_external_field_through_program_options() -> None:
    water = Structure.build(["O", "H", "H"], WATER)
    plain = create("tblite:GFN2-xTB").evaluate_structure(water).energy
    field = create(
        "tblite:GFN2-xTB", CalculatorOptions(extra={"add.electric-field": [0.0, 0.0, 0.01]})
    )
    assert abs(field.evaluate_structure(water).energy - plain) > 1e-6


def test_ase_hessians_fall_back_to_finite_differences(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import warnings

    import fake_ase
    from mopt_rs.calculators.ase import FINITE_DIFFERENCES

    if not has("ase"):
        monkeypatch.setitem(sys.modules, "ase", fake_ase.module())
    x0 = WATER * BOHR_TO_ANGSTROM
    s = Structure.build(["O", "H", "H"], WATER + 0.05)
    other = Structure.build(["O", "H", "H"], WATER - 0.04)
    k_au = 2.0 / HARTREE_TO_EV * BOHR_TO_ANGSTROM**2
    expected = k_au * np.eye(9)

    def engine(cls: Any) -> AseEngine:
        return AseEngine(cls, kwargs={"x0": x0}, electrons=False)

    # ASE defines no Hessian: most calculators get finite differences of the forces.
    plain = engine(fake_ase.Harmonic)
    assert plain.hessian_route == FINITE_DIFFERENCES
    assert not plain.capabilities.analytic_hessian
    assert plain.describe().endswith(f"Hessian: {FINITE_DIFFERENCES}")
    h = EngineCalculator(plain).evaluate_structure(s, need_hessian=True).hessian
    assert h is not None and np.allclose(h, expected, atol=1e-8)

    # A Hessian method of the calculator is used when it has one.
    analytic = engine(fake_ase.HarmonicWithHessian)
    assert analytic.hessian_route == "get_hessian()" and analytic.capabilities.analytic_hessian
    h = EngineCalculator(analytic).evaluate_structure(s, need_hessian=True).hessian
    assert h is not None and np.allclose(h, expected, atol=1e-12)

    # A declared method that fails, or returns an unusable matrix, falls back once, with a
    # warning, and stays on finite differences.
    for cls in (fake_ase.FailingHessian, fake_ase.MisshapenHessian):
        eng = engine(cls)
        calc = EngineCalculator(eng)
        with pytest.warns(RuntimeWarning, match="finite differences of the forces"):
            h = calc.evaluate_structure(s, need_hessian=True).hessian
        assert h is not None and np.allclose(h, expected, atol=1e-8)
        assert eng.hessian_route == FINITE_DIFFERENCES and not eng.capabilities.analytic_hessian
        assert "get_hessian() failed" in eng.describe()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            h2 = calc.evaluate_structure(other, need_hessian=True).hessian
        assert h2 is not None and np.allclose(h2, expected, atol=1e-8)

    # A Hessian method named by the user that does not exist falls back the same way.
    named = AseEngine(fake_ase.Harmonic, kwargs={"x0": x0}, electrons=False, hessian="no_such")
    with pytest.warns(RuntimeWarning, match="no_such"):
        h = EngineCalculator(named).evaluate_structure(s, need_hessian=True).hessian
    assert h is not None and np.allclose(h, expected, atol=1e-8)


def test_spin_as_unpaired_electrons() -> None:
    s = Structure.build(["C", "H", "H"], WATER, unpaired_electrons=2)
    assert s.state.multiplicity == 3
    assert Structure.build(["O"], [[0.0, 0.0, 0.0]], multiplicity=3).state.multiplicity == 3
    with pytest.raises(ValueError, match="disagree"):
        Structure.build(["C", "H", "H"], WATER, multiplicity=1, unpaired_electrons=2)
    opt = Optimizer(["O", "H"], np.ravel(WATER[:2]).tolist(), unpaired_electrons=1)
    tasks = opt.ask()
    assert tasks is not None and tasks[0]["multiplicity"] == 2
    with pytest.raises(ValueError, match="disagree"):
        Optimizer(["O", "H"], np.ravel(WATER[:2]).tolist(), multiplicity=1, unpaired_electrons=1)
    e1 = _core.evaluate_native("gfn0xtb", ["O", "H"], np.ravel(WATER[:2]).tolist(), 0, 2)
    e2 = _core.evaluate_native(
        "gfn0xtb", ["O", "H"], np.ravel(WATER[:2]).tolist(), unpaired_electrons=1
    )
    assert e1[0] == e2[0]


def test_gfnff_built_in_matches_xtb() -> None:
    import json
    from pathlib import Path

    golden = json.loads(
        (Path(__file__).parent / "golden" / "gfnff.json").read_text(encoding="utf-8")
    )
    case = next(c for c in golden["cases"] if c["molecule"] == "acetate_water")
    e, g, h = _core.evaluate_native(
        "gfnff", case["symbols"], case["positions_bohr"], case["charge"], None, True
    )
    assert e == pytest.approx(case["terms"]["total"], abs=1e-10)
    assert np.allclose(g, case["gradient"], atol=1e-9)
    assert h is not None
    hm = np.reshape(h, (len(g), len(g)))
    assert np.allclose(hm, hm.T, atol=1e-7)
    engine = create_engine("gfnff")
    assert engine.capabilities.pbc and not engine.capabilities.electrons
    pos = np.reshape(case["positions_bohr"], (-1, 3))
    result = engine.compute(
        Structure.build(case["symbols"], pos, charge=case["charge"]), hessian=False
    )
    assert result.energy == pytest.approx(case["terms"]["total"], abs=1e-10)


def test_gfnff_keeps_its_topology_and_warns_on_bonding_changes() -> None:
    import json
    from pathlib import Path

    golden = json.loads(
        (Path(__file__).parent / "golden" / "gfnff.json").read_text(encoding="utf-8")
    )
    case = next(c for c in golden["cases"] if c["molecule"] == "methanethiol")
    pos = np.reshape(case["positions_bohr"], (-1, 3))
    engine = create_engine("gfnff")
    first = engine.compute(Structure.build(case["symbols"], pos), hessian=False)
    assert "warnings" not in first.extra
    far = pos.copy()
    far[1] = pos[0] + 3.0 * (pos[1] - pos[0])  # break C-S
    with pytest.warns(RuntimeWarning, match="broken 1-2"):
        result = engine.compute(Structure.build(case["symbols"], far), hessian=False)
    assert "broken 1-2" in result.extra["warnings"][0]
    # A fresh calculator made at the far structure has no C-S bond (another energy).
    e_fresh = _core.evaluate_native("gfnff", case["symbols"], far.ravel().tolist())[0]
    assert abs(e_fresh - result.energy) > 1e-3


@pytest.mark.skipif(not has("pyscf"), reason="PySCF not installed")
def test_pyscf_tells_the_spin_square(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    calc = create("pyscf", CalculatorOptions(method="hf", basis="sto-3g"))
    radical = Structure.build(["O", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.83]], multiplicity=2)
    found = calc.evaluate_structure(radical).spin_square
    assert found is not None and 0.75 <= found < 0.80
    # A closed shell is restricted: nothing to tell.
    assert calc.evaluate_structure(Structure.build(["O", "H", "H"], WATER)).spin_square is None
    # The run log shows it under every iteration.
    monkeypatch.chdir(tmp_path)
    xyz = tmp_path / "oh.xyz"
    xyz.write_text("2\n\nO 0.0 0.0 0.0\nH 0.0 0.0 0.97\n", encoding="utf-8")
    args = [str(xyz), "-func", "hf", "-bs", "sto-3g", "-spin", "2", "-ns", "2"]
    assert _core.run_cli("optmain", args) == 0
    (log,) = tmp_path.rglob("mopt.log")
    text = log.read_text(encoding="utf-8")
    assert text.count("<S**2>  ") == text.count("# ITR. ") >= 1
    assert "(S(S+1) = 0.7500;" in text
    # The papers of the program and what it printed itself are in the log.
    assert "Reference: Q. Sun et al., WIREs Comput. Mol. Sci. 2018" in text
    assert "converged SCF energy" in text
    assert text.index("converged SCF energy") < text.index("caluculation results")


class OneAtATime(Engine):
    """A harmonic well whose calls take a moment without holding the interpreter, as a
    program behind an engine does; it notices two calls at once."""

    name = "one_at_a_time"
    units = ASE_UNITS
    capabilities = Capabilities(electrons=False, rigid_body_invariant=False)

    def __init__(self) -> None:
        self.inside = 0
        self.overlaps = 0
        self.calls = 0

    def compute(self, structure: Structure, hessian: bool) -> EngineResult:
        import time

        self.inside += 1
        self.calls += 1
        if self.inside > 1:
            self.overlaps += 1
        time.sleep(0.002)
        self.inside -= 1
        d = structure.positions_angstrom
        return EngineResult(0.5 * float(np.sum(d * d)), forces=-d)


def test_an_engine_is_called_by_one_thread_at_a_time(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mopt_rs.calculators import register

    engine = OneAtATime()
    register("one_at_a_time_test", lambda arg, o: engine, replace_existing=True)
    monkeypatch.chdir(tmp_path)
    # A chain of images: the commands evaluate its images in parallel threads.
    frames = "".join(f"2\n\nAr 0.0 0.0 0.0\nAr {1.0 + 0.2 * k:.3f} 0.0 0.0\n" for k in range(8))
    path = tmp_path / "chain.xyz"
    path.write_text(frames)
    code = _core.run_cli(
        "nebmain",
        [str(path), "-os", "one_at_a_time_test", "-ns", "3", "-nd", "0", "-adpred", "0"],
    )
    assert code == 0
    assert engine.calls >= 16
    assert engine.overlaps == 0
