# tests/python/test_env.py
"""Tests for the swage package metadata and environment diagnostics."""

import subprocess
import sys

import swage
from swage import env


def test_version_present():
    """The package exposes a PEP 440 version string."""
    assert swage.__version__
    assert swage.__version__[0].isdigit()


def test_report_keys():
    """The environment report contains every documented field."""
    result = env.report()
    for key in ("swage", "python", "platform", "torch", "cuda", "gpu",
                "llvm_pin", "backends"):
        assert key in result


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
