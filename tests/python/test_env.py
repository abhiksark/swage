# tests/python/test_env.py
"""Tests for the swage package metadata and environment diagnostics."""

import subprocess
import sys
import types

import pytest
import swage
from swage import env

_NATIVE_MODULES = (
    "mlir_swage",
    "mlir_swage._mlir_libs",
    "mlir_swage._mlir_libs._swageDialectsNanobind",
)

_REVISION = "0123456789abcdef0123456789abcdef01234567"


def _install_fake_bindings(monkeypatch, **native_attributes):
    """Place a fake build-tree `mlir_swage` package in `sys.modules`."""
    package, libs, native = (types.ModuleType(name) for name in _NATIVE_MODULES)
    native.swage = types.SimpleNamespace(**native_attributes)
    libs._swageDialectsNanobind = native
    package._mlir_libs = libs
    for name, module in zip(_NATIVE_MODULES, (package, libs, native)):
        monkeypatch.setitem(sys.modules, name, module)


def _remove_bindings(monkeypatch):
    """Make `mlir_swage` unimportable even when a build tree is on the path."""
    # Blocking the parent is not enough: a fully dotted module already in
    # sys.modules is returned without consulting the parent.
    monkeypatch.setitem(sys.modules, "mlir_swage", None)
    for name in _NATIVE_MODULES[1:]:
        monkeypatch.delitem(sys.modules, name, raising=False)


def _stub_identity(monkeypatch, identity):
    """Replace the compiler identity the report reads its revision from."""
    from swage import _runtime

    monkeypatch.setattr(_runtime, "_cached_identity", lambda: identity)


def test_version_present():
    """The package exposes a PEP 440 version string."""
    assert swage.__version__
    assert swage.__version__[0].isdigit()


def test_report_keys():
    """The environment report contains every documented field."""
    result = env.report()
    for key in (
        "swage",
        "python",
        "platform",
        "torch",
        "torch_cuda_build",
        "cuda_driver",
        "cuda",
        "gpu",
        "llvm_pin",
        "llvm_linked",
        "revision",
        "backends",
    ):
        assert key in result


def test_report_separates_torch_build_from_cuda_driver(monkeypatch):
    """Do not misreport the build-time CUDA version as the driver."""
    from swage import _runtime

    torch = types.SimpleNamespace(
        __version__="2.8.0",
        version=types.SimpleNamespace(cuda="12.8"),
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            get_device_capability=lambda: (8, 6),
            get_device_name=lambda: "RTX A6000",
        ),
    )
    monkeypatch.setattr(env.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(_runtime, "driver_version", lambda: "13.0")

    result = env.report()

    assert result["torch_cuda_build"] == "12.8"
    assert result["cuda_driver"] == "13.0"


def test_report_says_unavailable_without_the_bindings(monkeypatch):
    """A missing `mlir_swage` package is reported, not raised."""
    _remove_bindings(monkeypatch)

    result = env.report()

    assert result["backends"]["mlir"] == (
        "unavailable (build-tree mlir_swage bindings not importable)"
    )
    assert result["llvm_linked"] is None


def test_report_says_unavailable_when_the_bindings_fail_to_load(monkeypatch):
    """A binding import that fails for any reason is reported, not raised."""

    class _BrokenBindings(types.ModuleType):
        def __getattr__(self, name):
            raise OSError("libSwagePythonCAPI.so: cannot open shared object")

    _remove_bindings(monkeypatch)
    broken = _BrokenBindings("mlir_swage")
    monkeypatch.setitem(sys.modules, "mlir_swage", broken)

    result = env.report()

    assert "unavailable" in result["backends"]["mlir"]
    assert result["llvm_linked"] is None


def test_report_says_available_with_the_bindings(monkeypatch):
    """An importable `mlir_swage` reports the LLVM it was linked against."""
    _install_fake_bindings(monkeypatch, __llvm_version__="22.1.8")

    result = env.report()

    assert result["backends"]["mlir"] == "available (linked LLVM 22.1.8)"
    assert result["llvm_linked"] == "22.1.8"


def test_report_does_not_guess_the_llvm_of_unversioned_bindings(monkeypatch):
    """Bindings built before the version attribute report no linked LLVM."""
    _install_fake_bindings(monkeypatch)

    result = env.report()

    assert result["backends"]["mlir"] == "available (linked LLVM unknown)"
    assert result["llvm_linked"] is None


@pytest.mark.parametrize(
    ("identity", "expected"),
    [
        ({"revision": None, "clean": False, "llvm": None}, None),
        ({"revision": _REVISION, "clean": True, "llvm": None}, "0123456789ab"),
        (
            {"revision": _REVISION, "clean": False, "llvm": None},
            "0123456789ab-dirty",
        ),
    ],
)
def test_report_revision_follows_the_compiler_identity(
    monkeypatch, identity, expected
):
    """Report the short HEAD, mark a dirty tree, and say None otherwise."""
    _stub_identity(monkeypatch, identity)

    assert env.report()["revision"] == expected


def test_report_revision_is_none_when_the_identity_fails(monkeypatch):
    """An identity lookup that raises must not break the report."""
    from swage import _runtime

    def _raise():
        raise OSError("git is not installed")

    monkeypatch.setattr(_runtime, "_cached_identity", _raise)

    assert env.report()["revision"] is None


def test_importing_env_does_not_import_optional_dependencies():
    """Probing happens when the report is built, not at import time."""
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "import swage.env\n"
            "assert 'mlir_swage' not in sys.modules\n"
            "assert 'torch' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


def test_module_entrypoint():
    """`python -m swage.env` prints the report and exits cleanly."""
    proc = subprocess.run(
        [sys.executable, "-m", "swage.env"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "swage:" in proc.stdout
    assert "python:" in proc.stdout
    assert "llvm_linked:" in proc.stdout
    assert "revision:" in proc.stdout
    assert "backends: {'mlir': '" in proc.stdout


def test_module_entrypoint_without_optional_components():
    """The command exits cleanly with neither PyTorch nor the bindings."""
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy, sys\n"
            "sys.modules['torch'] = None\n"
            "sys.modules['mlir_swage'] = None\n"
            "runpy.run_module('swage.env', run_name='__main__')",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "torch: None" in proc.stdout
    assert "cuda: False" in proc.stdout
    assert "llvm_linked: None" in proc.stdout
    assert (
        "backends: {'mlir': 'unavailable (build-tree mlir_swage bindings "
        "not importable)'}"
    ) in proc.stdout
