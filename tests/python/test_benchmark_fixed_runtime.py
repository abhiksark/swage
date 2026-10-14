# tests/python/test_benchmark_fixed_runtime.py
"""Frozen SLO thresholds, evidence completeness, and enforcement eligibility."""

import json

import pytest
from swage import _benchmark


@pytest.fixture
def slo_harness():
    """Load the measurement harness without importing PyTorch or native code."""
    return _benchmark


def _evidence():
    return {
        "correctness": dict.fromkeys(
            ("cold", "warm", "throughput_262144", "throughput_1048576"), True
        ),
        "measurements": {
            "cold": [
                {
                    "elapsed_ms": ms,
                    "rss_delta_bytes": 512 * 1024 * 1024,
                    "correct": True,
                }
                for ms in (100, 250, 250, 250, 400)
            ],
            "warm_host_us": [10.0] * 9 + [15.0] * 2 + [20.0] * 9,
            # Each size sits exactly at its ratio ceiling.
            "throughput": {
                str(n): {"swage_us": [swage] * 100, "torch_us": [10.0] * 100}
                for n, swage in ((1 << 18, 18.0), (1 << 20, 15.0))
            },
        },
    }


def test_inclusive_threshold_equality_passes(slo_harness):
    """Equality at every frozen upper threshold is a passing observation."""
    record = _evidence()
    assert slo_harness.evaluate(record)
    assert record["gates"]["warm_host_us"]["statistics"]["p95"] == 20
    assert {
        n: gate["maximum_ratio"]
        for n, gate in record["gates"]["throughput"].items()
    } == {"262144": 1.80, "1048576": 1.50}
    assert record["valid"] and record["passed"]


@pytest.mark.parametrize(
    "gate",
    ("cold", "warm", "memory", "throughput_262144", "throughput_1048576"),
)
def test_exceeding_one_threshold_fails_the_record(slo_harness, gate):
    """An otherwise valid campaign cannot hide a single failed release SLO."""
    record = _evidence()
    raw = record["measurements"]
    if gate == "cold":
        raw["cold"][-1]["elapsed_ms"] = 400.001
    elif gate == "warm":
        raw["warm_host_us"] = [15.001] * 20
    elif gate == "memory":
        raw["cold"][-1]["rss_delta_bytes"] += 1
    elif gate == "throughput_262144":
        raw["throughput"]["262144"]["swage_us"] = [18.001] * 100
    else:
        raw["throughput"]["1048576"]["swage_us"] = [15.001] * 100
    assert not slo_harness.evaluate(record)
    assert record["valid"] and not record["passed"]


def test_missing_samples_or_correctness_cannot_qualify(slo_harness):
    """Partial raw measurements and failed correctness are invalid evidence."""
    record = _evidence()
    record["measurements"]["cold"].pop()
    with pytest.raises(ValueError, match="five correct"):
        slo_harness.evaluate(record)
    record = _evidence()
    record["correctness"]["warm"] = False
    with pytest.raises(ValueError, match="correctness preflight"):
        slo_harness.evaluate(record)


def test_hardware_enforcement_is_exact(slo_harness):
    """The same architecture on a different GPU is not production evidence."""
    slo_harness.require_qualified_hardware("NVIDIA RTX A6000", "sm_86")
    with pytest.raises(ValueError, match="requires NVIDIA RTX A6000"):
        slo_harness.require_qualified_hardware(
            "NVIDIA GeForce RTX 3090", "sm_86"
        )
    with pytest.raises(ValueError, match="requires NVIDIA RTX A6000"):
        slo_harness.require_qualified_hardware("NVIDIA RTX A6000", "sm_90")


def test_failure_writes_partial_raw_json_before_exit(
    slo_harness,
    monkeypatch,
    tmp_path,
):
    """A failed run retains observations and never claims a passed gate."""

    def fail(record, enforce):
        record["measurements"]["warm_host_us"].append(12.5)
        raise ValueError("fixed vector-add correctness mismatch")

    monkeypatch.setattr(slo_harness, "_measure", fail)
    output = tmp_path / "evidence.json"
    assert slo_harness.run_fixed_vector_add(output=output, enforce=True) == 1
    record = json.loads(output.read_text())
    assert record["benchmark"] == "fixed-runtime"
    assert record["schema_version"] == 1
    assert record["measurements"]["warm_host_us"] == [12.5]
    assert record["configuration"]["cold"]["processes"] == 5
    assert record["error"]["type"] == "ValueError"
    assert not record["valid"] and not record["passed"]


@pytest.mark.parametrize("enforce", (False, True))
def test_threshold_failure_keeps_complete_evidence(
    slo_harness,
    monkeypatch,
    tmp_path,
    enforce,
):
    """Complete evidence survives failure; only enforcement fails the exit."""

    def slow(record, enforce):
        record.update(_evidence())
        record["measurements"]["warm_host_us"] = [21.0] * 20

    monkeypatch.setattr(slo_harness, "_measure", slow)
    output = tmp_path / "slow.json"
    assert slo_harness.run_fixed_vector_add(
        output=output, enforce=enforce
    ) == int(enforce)
    record = json.loads(output.read_text())
    assert record["gates"]["cold_ms"]["passed"]
    assert not record["gates"]["warm_host_us"]["passed"]
    assert (
        len(record["measurements"]["throughput"]["1048576"]["torch_us"]) == 100
    )
    assert record["valid"] and not record["passed"]
