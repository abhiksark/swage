# tests/python/test_benchmark_mixed_sum.py
"""Tests for the frozen mixed-policy benchmark contract."""

import importlib
import pathlib
import subprocess
import sys
import types

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


class _FakeEvent:
    """A CUDA event stand-in; the rotation test only needs it to exist."""

    def __init__(self, enable_timing):
        pass

    def record(self):
        pass

    def synchronize(self):
        pass

    def elapsed_time(self, end):
        return 0.0


def test_gate_runs_only_on_its_device_unless_asked(mixed_sum_benchmark):
    """Keep the gate on the A6000 and let another GPU rerun it by option."""
    arguments = mixed_sum_benchmark._arguments(["--output", "x.json"])
    rerun = mixed_sum_benchmark._arguments(
        ["--output", "x.json", "--any-device"]
    )

    assert arguments.any_device is False
    assert rerun.any_device is True
    assert mixed_sum_benchmark._gate_device(
        "NVIDIA RTX A6000", (8, 6), any_device=False
    )
    assert mixed_sum_benchmark._gate_device(
        "NVIDIA RTX A6000", (8, 6), any_device=True
    )
    with pytest.raises(RuntimeError, match="RTX 5090 at sm_120.*--any-device"):
        mixed_sum_benchmark._gate_device(
            "NVIDIA GeForce RTX 5090", (12, 0), any_device=False
        )
    assert not mixed_sum_benchmark._gate_device(
        "NVIDIA GeForce RTX 5090", (12, 0), any_device=True
    )


def test_status_separates_gate_evidence_from_a_rerun(mixed_sum_benchmark):
    """Label a record from another GPU so it cannot pass as the gate."""
    assert mixed_sum_benchmark._status(True) == "frozen gate run"
    assert mixed_sum_benchmark._status(False) == (
        "rerun on another device; not gate evidence"
    )


def test_positions_follow_the_rotation_of_the_timing_loop(
    mixed_sum_benchmark, monkeypatch
):
    """Derive each sample's place from the order the loop really used."""
    log = []
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            Event=_FakeEvent, synchronize=lambda: log.append("synchronize")
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    policies = ("warp", "cta", "mixed")

    samples = mixed_sum_benchmark._measure(
        {name: (lambda name=name: log.append(name)) for name in policies}
    )

    timed = log[log.index("synchronize") + 1 :]
    count = len(samples["warp"])
    assert count == 100
    assert len(timed) == 3 * count
    positions = mixed_sum_benchmark._positions(policies, count)
    for sample in range(count):
        order = timed[3 * sample : 3 * sample + 3]
        assert [order.index(name) for name in policies] == [
            positions[name][sample] for name in policies
        ]
    assert {name: positions[name][:4] for name in policies} == {
        "warp": [0, 2, 1, 0],
        "cta": [1, 0, 2, 1],
        "mixed": [2, 1, 0, 2],
    }


def test_position_medians_split_the_idle_stream_start(mixed_sum_benchmark):
    """Report each policy's median at each place in the rotation."""
    policies = ("warp", "cta", "mixed")
    positions = mixed_sum_benchmark._positions(policies, 9)
    base = {"warp": 60.0, "cta": 70.0, "mixed": 50.0}
    # The first policy of a rotation starts on an idle stream and pays ten
    # more; later places queue behind a running kernel.
    samples = {
        name: [
            base[name] + (10.0 if position == 0 else 0.0) + sample * 0.001
            for sample, position in enumerate(positions[name])
        ]
        for name in policies
    }

    medians = mixed_sum_benchmark._position_medians(samples)

    assert {name: [round(v) for v in medians[name]] for name in policies} == {
        "warp": [70, 60, 60],
        "cta": [80, 70, 70],
        "mixed": [60, 50, 50],
    }
    ratios = [
        mixed_sum_benchmark._evaluate_gate(
            {name: medians[name][position] for name in policies}
        )[0]
        for position in range(3)
    ]
    assert ratios == pytest.approx([60 / 70, 50 / 60, 50 / 60], abs=1e-3)


def test_ratio_is_bounded_by_the_timer_tick(mixed_sum_benchmark):
    """Show how far one tick moves a ratio of two short medians."""
    tick = 1.024
    low, high = mixed_sum_benchmark._ratio_tick_interval(
        62 * tick, 66 * tick, tick
    )

    assert low == pytest.approx(61.5 / 66.5)
    assert high == pytest.approx(62.5 / 65.5)
    assert mixed_sum_benchmark._ratio_tick_interval(1.0, 2.0, None) is None


def test_timer_block_reports_the_tick_and_the_ratio_range(
    mixed_sum_benchmark,
):
    """Record the observed tick beside the medians it limits."""
    tick = 0.001024
    samples = {
        "warp": [66 * tick, 67 * tick, 65 * tick],
        "cta": [70 * tick, 71 * tick, 73 * tick],
        "mixed": [62 * tick, 61 * tick, 63 * tick],
    }
    medians = {"warp": 66 * tick, "cta": 71 * tick, "mixed": 62 * tick}

    timer = mixed_sum_benchmark._timer(samples, medians, "mixed", "warp")

    assert timer["tick_ms"] == pytest.approx(tick)
    assert timer["ticks_per_median"] == pytest.approx(
        {"warp": 66, "cta": 71, "mixed": 62}
    )
    assert timer["ratio_tick_interval"] == pytest.approx(
        [61.5 / 66.5, 62.5 / 65.5]
    )
