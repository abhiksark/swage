# tests/python/test_benchmark_fresh_offsets.py
"""Tests for the fresh-offsets benchmark and the looped Triton baseline."""

import importlib
import pathlib

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_LOOPED_BLOCKS = (128, 256, 512, 1024)


@pytest.fixture
def triton_comparison(monkeypatch):
    """Import the comparison harness as its script entry point does."""
    monkeypatch.syspath_prepend(str(_ROOT / "benchmarks"))
    return importlib.import_module("benchmark_triton_comparison")


def test_looped_sweep_has_no_longest_segment_floor(triton_comparison):
    """Sweep small blocks whatever the longest segment is."""
    configs = triton_comparison._triton_looped_configs()

    assert configs == [
        *((128, warps) for warps in (1, 2, 4)),
        *((256, warps) for warps in (1, 2, 4, 8)),
        *((512, warps) for warps in (1, 2, 4, 8)),
        *((1024, warps) for warps in (1, 2, 4, 8)),
    ]


def test_fixed_sweep_still_covers_the_longest_segment(triton_comparison):
    """Keep the fixed-shape baseline exactly as the frozen record ran it."""
    assert triton_comparison._triton_sum_configs(4096) == [
        (4096, 1),
        (4096, 2),
        (4096, 4),
        (4096, 8),
    ]
    assert len(triton_comparison._triton_sum_configs(32)) == 26
    assert triton_comparison._triton_sum_configs(4097) == []
