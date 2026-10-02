# tests/python/test_env.py
"""Tests for the swage package metadata and environment diagnostics."""

import os
import pathlib
import re
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
_CACHE_VARIABLES = (
    "SWAGE_CACHE_MAX_ENTRIES",
    "SWAGE_CACHE_READ_ONLY",
    "SWAGE_NO_COMPILE",
)


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    """Keep every report away from the user's cache and its settings."""
    from swage import _runtime

    monkeypatch.setenv("SWAGE_CACHE_DIR", str(tmp_path / "isolated-cache"))
    monkeypatch.setattr(_runtime, "_cache_off", {})
    for name in _CACHE_VARIABLES:
        monkeypatch.delenv(name, raising=False)


def _subprocess_environment(tmp_path, **overrides):
    """Return this environment with an isolated, default cache setup."""
    environment = {
        name: value
        for name, value in os.environ.items()
        if name not in _CACHE_VARIABLES
    }
    environment["SWAGE_CACHE_DIR"] = str(tmp_path / "cache")
    environment.update(overrides)
    return environment


def _install_fake_bindings(monkeypatch, **native_attributes):
    """Place a fake `mlir_swage` package built for this `swage` in place.

    Args:
        monkeypatch: The pytest fixture that undoes the installation.
        **native_attributes: Attributes of the native `swage` module. They
            replace the build identity the fake records by default; a value
            of None leaves the attribute out.
    """
    from swage import _runtime

    attributes = {
        "__version__": swage.__version__,
        "__source_revision__": "unknown",
        **native_attributes,
    }
    package, libs, native = (types.ModuleType(name) for name in _NATIVE_MODULES)
    native.swage = types.SimpleNamespace(
        **{
            name: value
            for name, value in attributes.items()
            if value is not None
        }
    )
    monkeypatch.setattr(_runtime, "_verified_bindings", None)
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


def _identified_compiler(monkeypatch, **overrides):
    """Stub an identity that the disk cache accepts, adjusted by overrides."""
    from swage import _runtime

    identity = {
        "revision": _REVISION,
        "clean": True,
        "llvm": "llvmorg-test",
        "frontend": "f" * 64,
        "native": [["_swageDialectsNanobind.so", 1, 2]],
    }
    identity.update(overrides)
    _stub_identity(monkeypatch, identity)
    # A stubbed identity describes no files, so the check that ties it to
    # the loaded code is stubbed with it.
    monkeypatch.setattr(_runtime, "_stale_identity", lambda _identity: None)


def _fake_cuda_torch(capability):
    """Return a fake PyTorch whose current device has `capability`."""
    return types.SimpleNamespace(
        __version__="2.8.0",
        version=types.SimpleNamespace(cuda="12.8"),
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            get_device_capability=lambda: capability,
            get_device_name=lambda: "A device",
        ),
    )


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
        "native_version",
        "native_revision",
        "revision",
        "backends",
        "swage_file",
        "mlir_swage_file",
        "target",
        "cache_dir",
        "cache",
        "compile_on_miss",
    ):
        assert key in result


def test_report_names_the_imported_package_file():
    """Tell two checkouts apart by the file `swage` was imported from."""
    assert env.report()["swage_file"] == swage.__file__


def test_report_names_the_loaded_bindings_file(monkeypatch):
    """Name the native extension that was loaded, not a search path."""
    _install_fake_bindings(monkeypatch, __llvm_version__="22.1.8")
    native = sys.modules[_NATIVE_MODULES[2]]
    native.__file__ = "/build/mlir_swage/_mlir_libs/_swageDialectsNanobind.so"

    assert env.report()["mlir_swage_file"] == native.__file__


def test_report_has_no_bindings_file_without_the_bindings(monkeypatch):
    """Report no file when nothing was loaded."""
    _remove_bindings(monkeypatch)

    assert env.report()["mlir_swage_file"] is None


@pytest.mark.parametrize(
    ("capability", "expected"),
    [
        ((8, 6), "sm_86 (qualified)"),
        ((8, 0), "sm_80 (admitted, not qualified)"),
        ((8, 9), "sm_89 (admitted, not qualified)"),
        ((9, 0), "sm_90 (admitted, not qualified)"),
        ((12, 0), "sm_120 (admitted, not qualified)"),
        ((7, 5), "sm_75 (not admitted)"),
        ((8, 5), "sm_85 (not admitted)"),
    ],
)
def test_report_says_whether_the_device_target_is_qualified(
    monkeypatch, capability, expected
):
    """Separate executed targets from targets that only compile."""
    from swage import _runtime

    monkeypatch.setattr(env.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setitem(sys.modules, "torch", _fake_cuda_torch(capability))
    monkeypatch.setattr(_runtime, "driver_version", lambda: "13.0")

    assert env.report()["target"] == expected


def test_report_has_no_target_without_a_cuda_device(monkeypatch):
    """Report no target when PyTorch sees no CUDA device."""
    torch = _fake_cuda_torch((8, 6))
    torch.cuda.is_available = lambda: False
    monkeypatch.setattr(env.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setitem(sys.modules, "torch", torch)

    result = env.report()

    assert result["target"] is None
    assert result["gpu"] is None


def test_admitted_targets_match_the_compiler():
    """Keep the report's admitted list equal to the one the compiler uses."""
    root = pathlib.Path(__file__).parents[2]
    source = root / "lib" / "Target" / "NVIDIATarget.cpp"
    if not source.is_file():
        pytest.skip("the compiler sources are not present")
    # The processor list of the target description, which code generation
    # and the native bindings read.
    body = re.search(
        r"constexpr uint16_t processors\[\] = \{(.*?)\};",
        source.read_text(),
        re.DOTALL,
    )

    assert body is not None
    numbers = re.findall(r"\d+", body[1])
    admitted = {f"sm_{number}" for number in numbers}
    assert admitted == env._ADMITTED_TARGETS
    assert env._QUALIFIED_TARGETS <= env._ADMITTED_TARGETS


def test_report_reads_the_cuda_driver_without_pytorch(monkeypatch):
    """Report the driver from `libcuda`, which needs no PyTorch."""
    from swage import _runtime

    monkeypatch.setattr(env.importlib.util, "find_spec", lambda _name: None)
    monkeypatch.setattr(_runtime, "driver_version", lambda: "13.0")

    result = env.report()

    assert result["torch"] is None
    assert result["cuda"] is False
    assert result["cuda_driver"] == "13.0"


def test_report_has_no_cuda_driver_when_the_lookup_fails(monkeypatch):
    """A driver lookup that raises must not break the report."""
    from swage import _runtime

    def _raise():
        raise OSError("libcuda.so.1: cannot open shared object file")

    monkeypatch.setattr(_runtime, "driver_version", _raise)

    assert env.report()["cuda_driver"] is None


def test_report_describes_an_active_cache(tmp_path, monkeypatch):
    """Say where the cache is, that it is used, and how full it is."""
    root = tmp_path / "cache"
    monkeypatch.setenv("SWAGE_CACHE_DIR", str(root))
    _identified_compiler(monkeypatch)

    absent = env.report()
    root.mkdir(mode=0o700)
    (root / ("a" * 64)).mkdir()
    (root / ".staging-leftover").mkdir()
    present = env.report()

    assert absent["cache_dir"] == present["cache_dir"] == str(root)
    assert absent["cache"] == (
        "active (reads and writes; 0 of at most 1024 entries)"
    )
    assert present["cache"] == (
        "active (reads and writes; 1 of at most 1024 entries)"
    )
    assert present["compile_on_miss"] == "allowed"
    assert sorted(path.name for path in root.iterdir()) == sorted(
        ["a" * 64, ".staging-leftover"]
    )


def test_report_describes_the_cache_modes(tmp_path, monkeypatch):
    """Show the read-only mode, the bound, and the no-compile mode."""
    monkeypatch.setenv("SWAGE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("SWAGE_CACHE_MAX_ENTRIES", "64")
    _identified_compiler(monkeypatch)

    bounded = env.report()
    monkeypatch.setenv("SWAGE_CACHE_READ_ONLY", "1")
    monkeypatch.setenv("SWAGE_NO_COMPILE", "1")
    restricted = env.report()

    assert bounded["cache"] == (
        "active (reads and writes; 0 of at most 64 entries)"
    )
    assert restricted["cache"] == "active (reads only; 0 entries)"
    assert restricted["compile_on_miss"] == "refused (SWAGE_NO_COMPILE=1)"
    assert not (tmp_path / "cache").exists()


def test_report_says_why_the_cache_is_off(monkeypatch):
    """Give the reason instead of only saying that the cache is unused."""
    _identified_compiler(monkeypatch, native=None)

    result = env.report()

    assert result["cache"] == (
        "off (the native compiler libraries are not found)"
    )
    assert result["compile_on_miss"] == "allowed"


def test_report_says_when_the_cache_root_is_rejected(tmp_path, monkeypatch):
    """Separate a root that fails every lookup from a cache that is off."""
    monkeypatch.setenv("SWAGE_CACHE_DIR", str(tmp_path))
    _identified_compiler(monkeypatch)
    tmp_path.chmod(0o707)

    try:
        result = env.report()
    finally:
        tmp_path.chmod(0o700)

    assert result["cache"] == (
        f"rejected (cache entry is world-writable: {tmp_path})"
    )


@pytest.mark.parametrize(
    ("name", "value", "reason"),
    [
        ("SWAGE_NO_COMPILE", "yes", "SWAGE_NO_COMPILE must be 0 or 1"),
        ("SWAGE_CACHE_READ_ONLY", "2", "SWAGE_CACHE_READ_ONLY must be 0 or 1"),
        ("SWAGE_CACHE_MAX_ENTRIES", "0", "must be a positive integer"),
    ],
)
def test_report_names_a_mistyped_cache_variable(
    monkeypatch, name, value, reason
):
    """Report the setting a launch would reject, without raising."""
    monkeypatch.setenv(name, value)

    result = env.report()

    assert result["cache"].startswith("unknown (")
    assert reason in result["cache"]
    assert result["compile_on_miss"] is None


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
        "unavailable (mlir_swage bindings not importable)"
    )
    assert result["llvm_linked"] is None
    assert result["native_version"] is None
    assert result["native_revision"] is None


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
    """Matching bindings report what they were built from and linked."""
    _install_fake_bindings(
        monkeypatch,
        __llvm_version__="22.1.8",
        __source_revision__=f"{_REVISION}-dirty",
    )

    result = env.report()

    assert result["backends"]["mlir"] == "available (linked LLVM 22.1.8)"
    assert result["native_version"] == swage.__version__
    assert result["native_revision"] == f"{_REVISION}-dirty"
    assert result["llvm_linked"] == "22.1.8"


@pytest.mark.parametrize(
    ("built_for", "reason"),
    [("0.0.1", "were built for swage 0.0.1"), (None, "record no swage")],
)
def test_report_names_bindings_built_for_another_swage(
    monkeypatch, built_for, reason
):
    """Bindings that `swage` refuses are reported as rejected, not raised."""
    _install_fake_bindings(
        monkeypatch, __version__=built_for, __llvm_version__="22.1.8"
    )

    result = env.report()

    assert result["backends"]["mlir"].startswith("rejected (")
    assert reason in result["backends"]["mlir"]
    assert swage.__version__ in result["backends"]["mlir"]
    assert result["native_version"] is None
    assert result["llvm_linked"] is None
    assert result["mlir_swage_file"] is None


def test_report_does_not_guess_the_llvm_of_unversioned_bindings(monkeypatch):
    """Bindings that record no LLVM version report no linked LLVM."""
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


def test_module_entrypoint(tmp_path):
    """`python -m swage.env` prints the report and exits cleanly."""
    proc = subprocess.run(
        [sys.executable, "-m", "swage.env"],
        env=_subprocess_environment(tmp_path),
        capture_output=True,
        text=True,
        check=True,
    )
    assert "swage:" in proc.stdout
    assert "python:" in proc.stdout
    assert "llvm_linked:" in proc.stdout
    assert "revision:" in proc.stdout
    assert "backends: {'mlir': '" in proc.stdout
    assert f"swage_file: {swage.__file__}\n" in proc.stdout
    assert "mlir_swage_file:" in proc.stdout
    assert "target:" in proc.stdout
    assert f"cache_dir: {tmp_path / 'cache'}\n" in proc.stdout
    assert "\ncache: " in proc.stdout
    assert "compile_on_miss: allowed\n" in proc.stdout
    assert not (tmp_path / "cache").exists()


def test_module_entrypoint_reports_a_mistyped_cache_variable(tmp_path):
    """The command exits cleanly on a setting that a launch rejects."""
    proc = subprocess.run(
        [sys.executable, "-m", "swage.env"],
        env=_subprocess_environment(tmp_path, SWAGE_NO_COMPILE="yes"),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert (
        "cache: unknown (SWAGE_NO_COMPILE must be 0 or 1; found 'yes')"
    ) in proc.stdout


def test_module_entrypoint_without_optional_components(tmp_path):
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
        env=_subprocess_environment(tmp_path),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "torch: None" in proc.stdout
    assert "cuda: False" in proc.stdout
    assert "target: None" in proc.stdout
    assert "llvm_linked: None" in proc.stdout
    assert "mlir_swage_file: None" in proc.stdout
    assert "cache: off (the native compiler libraries are not found)" in (
        proc.stdout
    )
    assert (
        "backends: {'mlir': 'unavailable (mlir_swage bindings not "
        "importable)'}"
    ) in proc.stdout
