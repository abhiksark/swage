# tests/python/test_benchmark_persistent_sum.py
"""Tests for the frozen persistent-scheduling benchmark contract."""

import importlib
import json
import pathlib
import subprocess

import pytest


@pytest.fixture
def persistent_benchmark(monkeypatch):
    """Import the standalone benchmark as its script entry point does."""
    root = pathlib.Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(root / "benchmarks"))
    return importlib.import_module("benchmark_persistent_sum")


def test_distribution_is_frozen_and_extremely_skewed(persistent_benchmark):
    """Keep one exact split outlier after all deterministic short segments."""
    lengths = persistent_benchmark._generate_lengths()

    assert len(lengths) == 32_768
    assert all(1 <= length <= 32 for length in lengths[:-1])
    assert lengths[-1] == 16_777_216
    assert lengths[:5] == [21, 10, 26, 4, 5]


def test_configuration_records_resident_schedule(persistent_benchmark):
    """Keep the predeclared block, residency, and timed sequences fixed."""
    configuration = persistent_benchmark._configuration()

    assert configuration["persistent_block_threads"] == 512
    assert configuration["persistent_resident_blocks"] == 168
    assert configuration["timed_static_sequence"] == [
        "fused direct",
        "split partial",
        "split merge",
    ]
    assert configuration["timed_persistent_sequence"] == [
        "counter reset",
        "resident kernel",
    ]


def test_gate_boundary(persistent_benchmark):
    """Accept the fixed 0.95 boundary and reject a larger ratio."""
    ratio, passed = persistent_benchmark._evaluate_gate(
        {"static_mixed": 2.0, "persistent": 1.9}
    )

    assert ratio == pytest.approx(0.95)
    assert passed
    assert not persistent_benchmark._evaluate_gate(
        {"static_mixed": 2.0, "persistent": 1.9001}
    )[1]


def test_recorded_a6000_result_preserves_the_failed_gate():
    """Keep the committed evidence honest about the predeclared near miss."""
    root = pathlib.Path(__file__).resolve().parents[2]
    result = json.loads(
        (root / "benchmarks/results/persistent-sum-a6000-sm86.json").read_text()
    )

    assert result["source"] == {
        "revision": "205f629f208cb85035f8f0baa161c3f233e4fd68",
        "worktree_clean": True,
    }
    assert result["medians_ms"] == {
        "persistent": 0.11752000078558922,
        "static_mixed": 0.11878400295972824,
    }
    assert result["persistent_to_static_ratio"] == pytest.approx(
        0.9893588181687432
    )
    assert result["gate"] == {"maximum_ratio": 0.95, "passed": False}
    assert all(
        len(samples) == 100
        for samples in result["raw_samples_ms"].values()
    )


def test_git_metadata_requires_clean_worktree(
    persistent_benchmark, monkeypatch
):
    """Reject performance evidence produced from modified sources."""
    results = iter(
        [
            subprocess.CompletedProcess([], 0, "abc123\n"),
            subprocess.CompletedProcess([], 0, " M kernel.cpp\n"),
        ]
    )
    monkeypatch.setattr(
        persistent_benchmark.subprocess,
        "run",
        lambda *args, **kwargs: next(results),
    )

    with pytest.raises(RuntimeError, match="clean source worktree"):
        persistent_benchmark._git_metadata(pathlib.Path("."))
