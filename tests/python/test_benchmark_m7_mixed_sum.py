# tests/python/test_benchmark_m7_mixed_sum.py
"""Tests for the fixed M7 mixed-policy benchmark contract."""

import importlib
import pathlib
import subprocess

import pytest


@pytest.fixture
def m7_benchmark(monkeypatch):
    """Import the standalone benchmark as its script entry point does."""
    root = pathlib.Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(root / "benchmarks"))
    return importlib.import_module("benchmark_m7_mixed_sum")


def test_gate_boundary(m7_benchmark):
    """Accept the declared boundary and reject a larger mixed ratio."""
    ratio, passed = m7_benchmark._evaluate_gate(
        {"warp": 2.0, "cta": 3.0, "mixed": 2.1}
    )
    assert ratio == pytest.approx(1.05)
    assert passed
    assert not m7_benchmark._evaluate_gate(
        {"warp": 2.0, "cta": 3.0, "mixed": 2.1001}
    )[1]


def test_configuration_records_fused_mixed_schedule(m7_benchmark):
    """Keep the predeclared one-launch fused schedule in every result."""
    assert m7_benchmark._configuration()["mixed_schedule"] == {
        "kind": "fused",
        "kernel_launches": 1,
        "block_threads": 128,
        "warp_slots_per_block": 4,
    }


def test_git_metadata_requires_a_clean_worktree(m7_benchmark, monkeypatch):
    """Do not label measurements from modified or untracked sources."""
    results = iter(
        [
            subprocess.CompletedProcess([], 0, "abc123\n"),
            subprocess.CompletedProcess([], 0, "?? scratch.txt\n"),
        ]
    )
    monkeypatch.setattr(
        m7_benchmark.subprocess, "run", lambda *args, **kw: next(results)
    )

    with pytest.raises(RuntimeError, match="clean source worktree"):
        m7_benchmark._git_metadata(pathlib.Path("."))
