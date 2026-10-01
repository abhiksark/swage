# python/tests/mlir/test_prepared_capture.py
"""Graph capture when the first launch had to wait for task storage.

A preparation that reuses cached kernels can reach its first launch before
the task upload event completes. That launch queues a device-side wait and
does not learn that the storage became ready. CUDA forbids querying or
synchronizing an event while a capture is open, so capture needs an earlier
launch that observed the storage ready.
"""

import pytest
import torch
from swage._segmented_qualification import (
    _prepare_persistent_sum,
    _prepare_planned_sum,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"
)

_LENGTHS = [0, 32, 33, 4096]


def _tensors():
    offsets = [0]
    for length in _LENGTHS:
        offsets.append(offsets[-1] + length)
    values = torch.ones(offsets[-1], device="cuda", dtype=torch.float32)
    device_offsets = torch.tensor(offsets, device="cuda", dtype=torch.int32)
    output = torch.full((len(_LENGTHS),), float("nan"), device="cuda")
    return values, device_offsets, output


def _report_pending_once(monkeypatch):
    """Make the first readiness query report pending."""
    real_query = torch.cuda.Event.query
    calls = {"count": 0}

    def query(event):
        calls["count"] += 1
        return False if calls["count"] == 1 else real_query(event)

    monkeypatch.setattr(torch.cuda.Event, "query", query)


def _replay_matches(launch, output):
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    output.fill_(float("nan"))
    graph.replay()
    torch.testing.assert_close(
        output.cpu(),
        torch.tensor(_LENGTHS, dtype=torch.float32),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("policy", ["warp", "cta"])
def test_planned_capture_after_waiting_then_ready_launch(policy, monkeypatch):
    """Launch, synchronize, launch again, then capture."""
    values, offsets, output = _tensors()
    prepared = _prepare_planned_sum(values, offsets, output)
    launch = getattr(prepared, policy)
    _report_pending_once(monkeypatch)

    launch()
    torch.cuda.synchronize()
    launch()
    torch.cuda.synchronize()

    _replay_matches(launch, output)


def test_persistent_capture_after_waiting_then_ready_launch(monkeypatch):
    """The persistent launch follows the same protocol."""
    values, offsets, output = _tensors()
    prepared = _prepare_persistent_sum(
        values, offsets, output, resident_blocks=2
    )
    _report_pending_once(monkeypatch)

    prepared.launch()
    torch.cuda.synchronize()
    prepared.launch()
    torch.cuda.synchronize()

    _replay_matches(prepared.launch, output)


@pytest.mark.parametrize("policy", ["warp", "cta"])
def test_planned_capture_after_only_a_waiting_launch_is_rejected(
    policy, monkeypatch
):
    """A launch that only queued the wait does not complete the handoff."""
    values, offsets, output = _tensors()
    prepared = _prepare_planned_sum(values, offsets, output)
    launch = getattr(prepared, policy)
    _report_pending_once(monkeypatch)
    launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()

    with pytest.raises(RuntimeError, match="launch again"):
        with torch.cuda.graph(graph):
            launch()


def test_planned_capture_before_any_launch_is_rejected():
    """Capture needs one launch after task initialization."""
    values, offsets, output = _tensors()
    prepared = _prepare_planned_sum(values, offsets, output)
    graph = torch.cuda.CUDAGraph()

    with pytest.raises(RuntimeError, match="must launch once"):
        with torch.cuda.graph(graph):
            prepared.cta()
