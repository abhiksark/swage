# tests/python/test_env.py
"""Tests for the swage package metadata and environment diagnostics."""

import subprocess
import sys
import types

import swage
from swage import env


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


def test_report_is_honest_about_backends():
    """No backend may claim availability until one can execute kernels."""
    result = env.report()
    assert "unavailable" in result["backends"]["mlir"]


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
