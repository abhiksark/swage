# python/tests/mlir/test_prepared_capture.py
"""Graph capture when the first launch had to wait for task storage.

A preparation that reuses cached kernels can reach its first launch before
the task upload event completes. That launch queues a device-side wait and
does not learn that the storage became ready. CUDA forbids querying or
synchronizing an event while a capture is open, so capture needs an earlier
launch that observed the storage ready.
"""

from contextlib import contextmanager
from itertools import accumulate, pairwise

import pytest
import torch
from swage._segmented_qualification import (
    _prepare_persistent_sum,
    _prepare_planned_sum,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"
)

# An empty segment, both sides of the warp limit, a full chunk, and one
# segment that splits, so a mixed replay covers the direct, partial, and
# merge launches.
_LENGTHS = [0, 32, 33, 4096, 4097]


def _tensors():
    """Build position-dependent values and their exact segment sums.

    Element ``index`` is ``(2 * (index % 67) - 65) / 4``, the pattern of
    test_segmented_runtime.py: nonzero multiples of 0.25 whose segment sums
    are exact in f32 under any order at these lengths. A replay that read a
    shifted window, or dropped or repeated an element, would change a sum,
    which all-one values cannot show.
    """
    offsets = list(accumulate(_LENGTHS, initial=0))
    index = torch.arange(offsets[-1])
    host_values = (2 * (index % 67) - 65).to(torch.float32) / 4
    exact = host_values.double()
    expected = torch.stack(
        [exact[begin:end].sum() for begin, end in pairwise(offsets)]
    ).float()
    values = host_values.cuda()
    device_offsets = torch.tensor(offsets, device="cuda", dtype=torch.int32)
    output = torch.full((len(_LENGTHS),), float("nan"), device="cuda")
    return values, device_offsets, output, expected


@contextmanager
def _storage_pending():
    """Report the task storage as not ready for every query in the block.

    The answer does not depend on how many readiness queries one launch
    makes, so it models a launch that ran entirely before the upload event
    completed.
    """
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torch.cuda.Event, "query", lambda event: False)
        yield


def _launch_and_check(launch, output, expected):
    output.fill_(float("nan"))
    launch()
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


def _replay_matches(launch, output, expected):
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    output.fill_(float("nan"))
    graph.replay()
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("policy", ["warp", "cta", "mixed"])
def test_planned_capture_after_waiting_then_ready_launch(policy):
    """Launch, synchronize, launch again, then capture."""
    values, offsets, output, expected = _tensors()
    prepared = _prepare_planned_sum(values, offsets, output)
    launch = getattr(prepared, policy)

    with _storage_pending():
        _launch_and_check(launch, output, expected)
    _launch_and_check(launch, output, expected)

    _replay_matches(launch, output, expected)


def test_persistent_capture_after_waiting_then_ready_launch():
    """The persistent launch follows the same protocol."""
    values, offsets, output, expected = _tensors()
    prepared = _prepare_persistent_sum(
        values, offsets, output, resident_blocks=2
    )

    with _storage_pending():
        _launch_and_check(prepared.launch, output, expected)
    _launch_and_check(prepared.launch, output, expected)

    _replay_matches(prepared.launch, output, expected)


@pytest.mark.parametrize("policy", ["warp", "cta", "mixed"])
def test_planned_capture_after_only_a_waiting_launch_is_rejected(policy):
    """A launch that only queued the wait does not complete the handoff."""
    values, offsets, output, expected = _tensors()
    prepared = _prepare_planned_sum(values, offsets, output)
    launch = getattr(prepared, policy)
    with _storage_pending():
        _launch_and_check(launch, output, expected)
    graph = torch.cuda.CUDAGraph()

    with pytest.raises(RuntimeError, match="launch again"):
        with torch.cuda.graph(graph):
            launch()


def test_planned_capture_before_any_launch_is_rejected():
    """Capture needs one launch after task initialization."""
    values, offsets, output, _ = _tensors()
    prepared = _prepare_planned_sum(values, offsets, output)
    graph = torch.cuda.CUDAGraph()

    with pytest.raises(RuntimeError, match="must launch once"):
        with torch.cuda.graph(graph):
            prepared.cta()
