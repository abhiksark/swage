# tests/python/test_env.py
"""Tests for the package metadata, environment report, and health checks."""

import json
import os
import pathlib
import re
import subprocess
import sys
import types

import pytest
import swage
from swage import _cuda_backend, _native, _runtime, env
from swage._errors import BackendUnavailableError

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
# A valid schema 2 build record of the native package.
_BUILD_INFO = {
    "schema_version": 2,
    "package_version": "0.5.2",
    "source_revision": "a" * 40,
    "source_clean": True,
    "frontend_digest": "b" * 64,
    "llvm_version": "llvmorg-22.1.8",
    "build_type": "Release",
}


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    """Keep every report away from the user's cache and its settings."""
    monkeypatch.setenv("SWAGE_CACHE_DIR", str(tmp_path / "isolated-cache"))
    monkeypatch.setattr(_runtime, "_cache_off", {})
    for name in _CACHE_VARIABLES:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def probes(monkeypatch):
    """Make prerequisite probes deterministic without native code or a GPU.

    The build record is valid, the bindings load and were built for this
    `swage`, the driver reports 13.0, and PyTorch sees the qualified
    NVIDIA RTX A6000 (sm_86).

    Returns:
        The fake PyTorch module and the build record.
    """
    info = dict(_BUILD_INFO)
    extension = types.SimpleNamespace(
        __version__=swage.__version__,
        __source_revision__="unknown",
        __llvm_version__="22.1.8",
    )
    monkeypatch.setattr(_native, "build_info", lambda: info)
    monkeypatch.setattr(_native, "load_ir", lambda: None)
    monkeypatch.setattr(_native, "load_extension", lambda: extension)
    monkeypatch.setattr(_runtime, "_verified_bindings", None)
    monkeypatch.setattr(env, "_driver_info", lambda: ("13.0", None))
    torch = types.SimpleNamespace(
        __version__="2.6.0",
        version=types.SimpleNamespace(cuda="12.4"),
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            get_device_capability=lambda: (8, 6),
            get_device_name=lambda: "NVIDIA RTX A6000",
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch, info


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

    The report loads the fake extension and checks it against `swage`. The
    IR bindings, which the fake package lacks, and the packaged build
    record are stubbed out, so the report describes the extension alone.

    Args:
        monkeypatch: The pytest fixture that undoes the installation.
        **native_attributes: Attributes of the native `swage` module. They
            replace the build identity the fake records by default; a value
            of None leaves the attribute out.
    """
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
    monkeypatch.setattr(_native, "build_info", lambda: None)
    monkeypatch.setattr(_native, "load_ir", lambda: None)
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
    monkeypatch.setattr(_runtime, "_cached_identity", lambda: identity)


def _identified_compiler(monkeypatch, **overrides):
    """Stub an identity that the disk cache accepts, adjusted by overrides."""
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


def test_version_present():
    """The package exposes a PEP 440 version string."""
    assert swage.__version__
    assert swage.__version__[0].isdigit()


def test_report_keys(probes):
    """The environment report contains every documented field."""
    result = env.report()

    assert set(result) == {
        "schema_version",
        "swage",
        "source",
        "python",
        "implementation",
        "machine",
        "platform",
        "torch",
        "torch_cuda_build",
        "cuda_driver",
        "cuda",
        "gpu",
        "llvm_pin",
        "native",
        "backends",
        "cache",
        "artifact",
    }
    assert result["schema_version"] == 2
    assert set(result["source"]) == {"file", "revision"}
    assert set(result["native"]) == {
        "package_version",
        "source_revision",
        "source_clean",
        "frontend_digest",
        "llvm_version",
        "build_type",
        "available",
        "error",
        "bindings",
    }
    assert set(result["native"]["bindings"]) == {
        "version",
        "revision",
        "llvm_linked",
        "file",
        "problem",
    }
    assert set(result["backends"]) == {"cpu", "cuda"}
    assert set(result["backends"]["cpu"]) == {"available", "reason"}
    assert set(result["backends"]["cuda"]) == {
        "available",
        "qualified",
        "target",
        "reason",
    }
    assert set(result["cache"]) == {"directory", "state", "compile_on_miss"}


def test_report_names_the_imported_package_file():
    """Tell two checkouts apart by the file `swage` was imported from."""
    assert env.report()["source"]["file"] == swage.__file__


def test_report_names_the_loaded_bindings_file(monkeypatch):
    """Name the native extension that was loaded, not a search path."""
    _install_fake_bindings(monkeypatch, __llvm_version__="22.1.8")
    native = sys.modules[_NATIVE_MODULES[2]]
    native.__file__ = "/build/mlir_swage/_mlir_libs/_swageDialectsNanobind.so"

    assert env.report()["native"]["bindings"]["file"] == native.__file__


@pytest.mark.parametrize(
    ("capability", "target", "admitted", "qualified"),
    [
        ((8, 6), "sm_86", True, True),
        ((8, 0), "sm_80", True, False),
        ((8, 9), "sm_89", True, False),
        ((9, 0), "sm_90", True, False),
        ((12, 0), "sm_120", True, False),
        ((7, 5), "sm_75", False, False),
        ((8, 5), "sm_85", False, False),
    ],
)
def test_report_says_whether_the_device_target_is_qualified(
    probes, monkeypatch, capability, target, admitted, qualified
):
    """Separate executed targets from targets that only compile."""
    torch, _ = probes
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: capability)

    cuda = env.report()["backends"]["cuda"]

    assert cuda["target"] == target
    assert cuda["available"] is admitted
    assert cuda["qualified"] is qualified
    not_admitted = "CUDA target is not admitted by the pinned compiler"
    assert cuda["reason"] == (None if admitted else not_admitted)


def test_report_has_no_target_without_a_cuda_device(probes, monkeypatch):
    """Report no target when PyTorch sees no CUDA device."""
    torch, _ = probes
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    result = env.report()

    assert result["cuda"] is False
    assert result["gpu"] is None
    assert result["backends"]["cuda"]["target"] is None
    assert not result["backends"]["cuda"]["available"]
    assert not result["backends"]["cuda"]["qualified"]
    assert "cuda-unavailable" in result["backends"]["cuda"]["reason"]
    assert result["backends"]["cpu"]["available"]


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
    assert admitted == env._ADMITTED_CUDA_TARGETS
    # The one qualified target is admitted.
    assert "sm_86" in env._ADMITTED_CUDA_TARGETS


def test_report_reads_the_cuda_driver_without_pytorch(monkeypatch):
    """Report the driver from `libcuda`, which needs no PyTorch."""
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setattr(_cuda_backend, "driver_version", lambda: "13.0")

    result = env.report()

    assert result["torch"] is None
    assert result["cuda"] is False
    assert result["cuda_driver"] == "13.0"


def test_report_has_no_cuda_driver_when_the_lookup_fails(monkeypatch):
    """A driver lookup that raises must not break the report."""

    def _raise():
        raise OSError("libcuda.so.1: cannot open shared object file")

    monkeypatch.setattr(_cuda_backend, "driver_version", _raise)

    result = env.report()

    reason = result["backends"]["cuda"]["reason"]
    assert result["cuda_driver"] is None
    assert "CUDA driver probe failed (OSError)" in reason


def test_report_describes_an_active_cache(tmp_path, monkeypatch):
    """Say where the cache is, that it is used, and how full it is."""
    root = tmp_path / "cache"
    monkeypatch.setenv("SWAGE_CACHE_DIR", str(root))
    _identified_compiler(monkeypatch)

    absent = env.report()["cache"]
    root.mkdir(mode=0o700)
    (root / ("a" * 64)).mkdir()
    (root / ".staging-leftover").mkdir()
    present = env.report()["cache"]

    assert absent["directory"] == present["directory"] == str(root)
    assert absent["state"] == (
        "active (reads and writes; 0 of at most 1024 entries)"
    )
    assert present["state"] == (
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

    bounded = env.report()["cache"]
    monkeypatch.setenv("SWAGE_CACHE_READ_ONLY", "1")
    monkeypatch.setenv("SWAGE_NO_COMPILE", "1")
    restricted = env.report()["cache"]

    assert bounded["state"] == (
        "active (reads and writes; 0 of at most 64 entries)"
    )
    assert restricted["state"] == "active (reads only; 0 entries)"
    assert restricted["compile_on_miss"] == "refused (SWAGE_NO_COMPILE=1)"
    assert not (tmp_path / "cache").exists()


def test_report_says_why_the_cache_is_off(monkeypatch):
    """Give the reason instead of only saying that the cache is unused."""
    _identified_compiler(monkeypatch, native=None)

    cache = env.report()["cache"]

    assert cache["state"] == (
        "off (the native compiler libraries are not found)"
    )
    assert cache["compile_on_miss"] == "allowed"


def test_report_says_when_the_cache_root_is_rejected(tmp_path, monkeypatch):
    """Separate a root that fails every lookup from a cache that is off."""
    monkeypatch.setenv("SWAGE_CACHE_DIR", str(tmp_path))
    _identified_compiler(monkeypatch)
    tmp_path.chmod(0o707)

    try:
        cache = env.report()["cache"]
    finally:
        tmp_path.chmod(0o700)

    assert cache["state"] == (
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

    cache = env.report()["cache"]

    assert cache["directory"] is None
    assert cache["state"].startswith("unknown (")
    assert reason in cache["state"]
    assert cache["compile_on_miss"] is None


def test_report_separates_torch_build_from_cuda_driver(monkeypatch):
    """Do not misreport the build-time CUDA version as the driver."""
    torch = types.SimpleNamespace(
        __version__="2.8.0",
        version=types.SimpleNamespace(cuda="12.8"),
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            get_device_capability=lambda: (8, 6),
            get_device_name=lambda: "RTX A6000",
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(_cuda_backend, "driver_version", lambda: "13.0")

    result = env.report()

    assert result["torch_cuda_build"] == "12.8"
    assert result["cuda_driver"] == "13.0"


def test_report_says_unavailable_without_the_bindings(monkeypatch):
    """A missing `mlir_swage` package is reported, not raised."""
    _remove_bindings(monkeypatch)

    result = env.report()

    assert not result["native"]["available"]
    assert result["native"]["error"] == "native-unavailable"
    # Nothing was loaded, so no bindings identity or file is reported.
    assert result["native"]["bindings"] is None
    assert not result["backends"]["cpu"]["available"]
    assert "native-unavailable" in result["backends"]["cpu"]["reason"]


def test_report_says_unavailable_when_the_bindings_fail_to_load(monkeypatch):
    """A binding import that fails for any reason is reported, not raised."""

    class _BrokenBindings(types.ModuleType):
        def __getattr__(self, name):
            raise OSError("libSwagePythonCAPI.so: cannot open shared object")

    _remove_bindings(monkeypatch)
    broken = _BrokenBindings("mlir_swage")
    monkeypatch.setitem(sys.modules, "mlir_swage", broken)

    result = env.report()

    assert not result["native"]["available"]
    assert result["native"]["error"] is not None
    assert result["native"]["bindings"] is None
    assert not result["backends"]["cpu"]["available"]


def test_report_says_available_with_the_bindings(monkeypatch):
    """Matching bindings report what they were built from and linked."""
    _install_fake_bindings(
        monkeypatch,
        __llvm_version__="22.1.8",
        __source_revision__=f"{_REVISION}-dirty",
    )

    native = env.report()["native"]

    assert native["available"]
    assert native["error"] is None
    assert native["bindings"] == {
        "version": swage.__version__,
        "revision": f"{_REVISION}-dirty",
        "llvm_linked": "22.1.8",
        "file": None,
        "problem": None,
    }


@pytest.mark.parametrize(
    ("built_for", "reason"),
    [("0.0.1", "were built for swage 0.0.1"), (None, "record no swage")],
)
def test_report_names_bindings_built_for_another_swage(
    monkeypatch, built_for, reason
):
    """Bindings that `swage` refuses are reported as refused, not raised."""
    _install_fake_bindings(
        monkeypatch, __version__=built_for, __llvm_version__="22.1.8"
    )

    result = env.report()

    native = result["native"]
    assert not native["available"]
    assert native["error"] == "native-mismatch"
    # The refused bindings still say what they were built for and linked.
    assert native["bindings"]["version"] == built_for
    assert native["bindings"]["llvm_linked"] == "22.1.8"
    assert reason in native["bindings"]["problem"]
    assert swage.__version__ in native["bindings"]["problem"]
    assert not result["backends"]["cpu"]["available"]


def test_report_does_not_guess_the_llvm_of_unversioned_bindings(monkeypatch):
    """Bindings that record no LLVM version report no linked LLVM."""
    _install_fake_bindings(monkeypatch)

    native = env.report()["native"]

    assert native["available"]
    assert native["bindings"]["llvm_linked"] is None


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

    assert env.report()["source"]["revision"] == expected


def test_report_revision_is_none_when_the_identity_fails(monkeypatch):
    """An identity lookup that raises must not break the report."""

    def _raise():
        raise OSError("git is not installed")

    monkeypatch.setattr(_runtime, "_cached_identity", _raise)

    assert env.report()["source"]["revision"] is None


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
    assert "schema_version: 2\n" in proc.stdout
    assert "swage:" in proc.stdout
    assert "python:" in proc.stdout
    assert f"source: {{'file': {swage.__file__!r}, 'revision': " in (
        proc.stdout
    )
    assert "\nnative: {" in proc.stdout
    assert "backends: {'cpu': {'available': " in proc.stdout
    assert f"cache: {{'directory': {str(tmp_path / 'cache')!r}, " in (
        proc.stdout
    )
    assert "'compile_on_miss': 'allowed'}\n" in proc.stdout
    assert "\nartifact: " in proc.stdout
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
        "unknown (SWAGE_NO_COMPILE must be 0 or 1; found 'yes')"
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
    assert "gpu: None" in proc.stdout
    assert (
        "'available': False, 'error': 'native-unavailable', 'bindings': None}"
    ) in proc.stdout
    assert "'state': 'off (the native compiler libraries are not found)'" in (
        proc.stdout
    )
    assert (
        "'cpu': {'available': False, 'reason': 'native-unavailable: install "
        "a supported native wheel; pytorch-unavailable: install "
        "swage-compiler[pytorch]'}"
    ) in proc.stdout
    cuda = "'cuda': {'available': False, 'qualified': False, 'target': None"
    assert cuda in proc.stdout


def test_report_identity_and_driver_build_separation(probes):
    """Packaged identity wins and CUDA build version is not the driver."""
    result = env.report()
    assert result["llvm_pin"] == "llvmorg-22.1.8"
    assert result["native"]["source_revision"] == "a" * 40
    assert result["native"]["source_clean"] is True
    assert result["native"]["frontend_digest"] == "b" * 64
    assert result["torch_cuda_build"] == "12.4"
    assert result["cuda_driver"] == "13.0"
    assert result["backends"]["cpu"]["available"]
    assert result["backends"]["cuda"] == {
        "available": True,
        "qualified": True,
        "target": "sm_86",
        "reason": None,
    }


def test_no_torch_keeps_native_and_driver_facts(probes, monkeypatch):
    """Missing PyTorch cannot hide successful native or driver probes."""
    monkeypatch.setitem(sys.modules, "torch", None)
    result = env.report()
    assert result["torch"] is None
    assert result["native"]["available"]
    assert result["cuda_driver"] == "13.0"
    assert not result["backends"]["cpu"]["available"]
    assert not result["backends"]["cuda"]["available"]


def test_no_native_keeps_torch_and_driver_facts(probes, monkeypatch):
    """Missing bindings disable both adapters, not independent GPU facts."""

    def unavailable():
        raise BackendUnavailableError(
            "missing",
            code="native-unavailable",
            backend="native",
            remediation="install wheel",
        )

    monkeypatch.setattr(_native, "load_ir", unavailable)
    result = env.report()
    assert not result["native"]["available"]
    assert result["torch"] == "2.6.0"
    assert result["gpu"]["compute_capability"] == "8.6"
    assert not result["backends"]["cpu"]["available"]
    assert not result["backends"]["cuda"]["available"]


def test_invalid_metadata_is_visible_without_disabling_compilation(
    probes,
    monkeypatch,
):
    """Malformed provenance is diagnosable while native imports still work."""

    def invalid():
        raise ValueError("invalid native build metadata: source_revision")

    monkeypatch.setattr(_native, "build_info", invalid)
    result = env.report()
    assert result["native"]["available"]
    assert result["native"]["source_revision"] is None
    assert "source_revision" in result["native"]["error"]
    assert result["backends"]["cpu"]["available"]


def test_cuda_target_admission_is_not_qualification(probes, monkeypatch):
    """Admitted non-sm86 targets remain available but are not qualified."""
    torch, _ = probes
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (9, 0))
    result = env.report()["backends"]["cuda"]
    assert result == {
        "available": True,
        "qualified": False,
        "target": "sm_90",
        "reason": None,
    }
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (7, 5))
    result = env.report()["backends"]["cuda"]
    assert not result["available"]
    assert not result["qualified"]
    assert "not admitted" in result["reason"]


def test_same_sm_does_not_qualify_another_gpu(probes, monkeypatch):
    """An admitted sm86 device is usable but not the qualified A6000."""
    torch, _ = probes
    monkeypatch.setattr(
        torch.cuda, "get_device_name", lambda: "NVIDIA GeForce RTX 3090"
    )
    cuda = env.report()["backends"]["cuda"]
    assert cuda["available"] and cuda["target"] == "sm_86"
    assert not cuda["qualified"]
    assert env.main(["--json", "--check", "cuda"]) == 0


def test_driver_failure_does_not_disable_cpu(probes, monkeypatch):
    """A driver problem is never a CPU prerequisite."""
    monkeypatch.setattr(
        env, "_driver_info", lambda: (None, "driver unavailable")
    )
    result = env.report()
    assert result["backends"]["cpu"]["available"]
    assert not result["backends"]["cuda"]["available"]
    assert result["torch_cuda_build"] == "12.4"


def test_probe_exceptions_do_not_disclose_payloads(probes, monkeypatch):
    """Diagnostics catch arbitrary probe failures without printing secrets."""

    def broken():
        raise RuntimeError("tensor([1234]); secret=do-not-log")

    torch, _ = probes
    monkeypatch.setattr(torch.cuda, "get_device_capability", broken)
    result = env.report()
    assert not result["backends"]["cuda"]["available"]
    assert "do-not-log" not in json.dumps(result)
    assert "RuntimeError" in result["backends"]["cuda"]["reason"]


@pytest.mark.parametrize("check", (None, "native", "cpu", "cuda"))
def test_json_health_exit_contract(probes, monkeypatch, capsys, check):
    """A complete sorted JSON report is printed even when checks fail."""
    monkeypatch.setitem(sys.modules, "torch", None)
    args = ["--json"] + ([] if check is None else ["--check", check])
    assert env.main(args) == (1 if check in ("cpu", "cuda") else 0)
    out = capsys.readouterr()
    result = json.loads(out.out)
    assert out.out == json.dumps(result, sort_keys=True) + "\n"
    assert not out.err
    assert result["schema_version"] == 2
    assert "implementation" in result and "machine" in result


def test_module_entrypoint_json_without_check(tmp_path):
    """The real CLI emits exactly one JSON object and exits zero by default."""
    proc = subprocess.run(
        [sys.executable, "-m", "swage.env", "--json"],
        env=_subprocess_environment(tmp_path),
        capture_output=True,
        text=True,
        check=True,
    )
    result = json.loads(proc.stdout)
    assert proc.stdout == json.dumps(result, sort_keys=True) + "\n"
    assert result["schema_version"] == 2


def test_metadata_resource_validation_is_fail_closed(tmp_path, monkeypatch):
    """Absent metadata differs from malformed bytes and invalid field types."""
    monkeypatch.setattr(
        _native.importlib.resources, "files", lambda _: tmp_path
    )
    assert _native.build_info() is None
    resource = tmp_path / "_build_info.json"
    resource.write_text('{"secret": "never-print-this"}')
    with pytest.raises(ValueError) as caught:
        _native.build_info()
    assert "never-print-this" not in str(caught.value)
    valid = dict(_BUILD_INFO)
    resource.write_text(json.dumps(valid))
    assert _native.build_info()["source_revision"] == "a" * 40
    # A build from sources without git records no revision.
    resource.write_text(json.dumps({**valid, "source_revision": None}))
    assert _native.build_info()["source_revision"] is None
    resource.write_text(json.dumps({**valid, "source_clean": "true"}))
    with pytest.raises(ValueError, match="source_clean"):
        _native.build_info()
    resource.write_text(json.dumps({**valid, "schema_version": True}))
    with pytest.raises(ValueError, match="schema_version"):
        _native.build_info()
    resource.write_text(json.dumps({**valid, "frontend_digest": "B" * 64}))
    with pytest.raises(ValueError, match="frontend_digest"):
        _native.build_info()
    # A schema 1 record has no frontend digest and is refused.
    schema_one = {**valid, "schema_version": 1}
    del schema_one["frontend_digest"]
    resource.write_text(json.dumps(schema_one))
    with pytest.raises(ValueError, match="schema fields"):
        _native.build_info()
