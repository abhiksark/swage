# tests/python/test_benchmark_mixed_sum.py
"""Tests for the frozen mixed-policy benchmark contract."""

import importlib
import pathlib
import subprocess

import pytest


@pytest.fixture
def mixed_sum_benchmark(monkeypatch):
    """Import the standalone benchmark as its script entry point does."""
    root = pathlib.Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(root / "benchmarks"))
    return importlib.import_module("benchmark_mixed_sum")


def test_gate_boundary(mixed_sum_benchmark):
    """Accept the declared boundary and reject a larger mixed ratio."""
    ratio, passed = mixed_sum_benchmark._evaluate_gate(
        {"warp": 2.0, "cta": 3.0, "mixed": 2.1}
    )
    assert ratio == pytest.approx(1.05)
    assert passed
    assert not mixed_sum_benchmark._evaluate_gate(
        {"warp": 2.0, "cta": 3.0, "mixed": 2.1001}
    )[1]


def test_configuration_records_fused_mixed_schedule(mixed_sum_benchmark):
    """Keep the predeclared one-launch fused schedule in every result."""
    assert mixed_sum_benchmark._configuration()["mixed_schedule"] == {
        "kind": "fused",
        "kernel_launches": 1,
        "block_threads": 128,
        "warp_slots_per_block": 4,
    }


def test_git_metadata_requires_a_clean_worktree(
    mixed_sum_benchmark, monkeypatch
):
    """Do not label measurements from modified or untracked sources."""
    results = iter(
        [
            subprocess.CompletedProcess([], 0, "abc123\n"),
            subprocess.CompletedProcess([], 0, "?? scratch.txt\n"),
        ]
    )
    monkeypatch.setattr(
        mixed_sum_benchmark.subprocess,
        "run",
        lambda *args, **kw: next(results),
    )

    with pytest.raises(RuntimeError, match="clean source worktree"):
        mixed_sum_benchmark._git_metadata(pathlib.Path("."))
