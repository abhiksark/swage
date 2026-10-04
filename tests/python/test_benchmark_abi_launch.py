# tests/python/test_benchmark_abi_launch.py
"""Tests for the fixed hot-launch ABI migration gate."""

import json
from pathlib import Path

import pytest

_RESULT = (
    Path(__file__).resolve().parents[2]
    / "benchmarks"
    / "results"
    / "abi-launch-a6000-sm86.json"
)


def _result():
    return json.loads(_RESULT.read_text())


def test_host_marshalling_gate_passes_recorded_measurement():
    """Keep the generic launcher within five percent of the legacy path."""
    result = _result()
    medians = result["medians_us_per_call"]
    ratio = medians["generic"] / medians["legacy"]

    assert result["generic_to_legacy_ratio"] == pytest.approx(ratio)
    assert result["gate"] == {"maximum_ratio": 1.05, "passed": True}
    assert ratio <= result["gate"]["maximum_ratio"]


def test_host_marshalling_measurement_is_frozen():
    """Retain the predeclared workload and batching methodology."""
    result = _result()

    assert result["configuration"] == {
        "batch_count": 20,
        "calls_per_batch": 500,
        "constexpr_block": 128,
        "element_count": 129,
        "grid": [2],
        "synchronization": "before and after each batch",
        "warmup_calls": 200,
    }
    assert len(result["raw_samples_us_per_call"]["generic"]) == 20
    assert len(result["raw_samples_us_per_call"]["legacy"]) == 20
