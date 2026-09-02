# python/tests/mlir/test_persistent_runtime.py
"""Adversarial GPU qualification for the private persistent scheduler."""

import random

import pytest
import torch
from swage._segmented_qualification import _prepare_persistent_sum

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"
)


def _offsets(lengths):
    """Return canonical host offsets for segment lengths."""
    result = [0]
    for length in lengths:
        result.append(result[-1] + length)
    return result


def _integer_case(lengths):
    """Create segment-specific integral values and exact f32 sums."""
    offsets = _offsets(lengths)
    values = torch.empty(offsets[-1], dtype=torch.float32)
    expected = []
    for segment_id, (begin, end) in enumerate(
        zip(offsets, offsets[1:])
    ):
        value = float(segment_id % 7 + 1)
        values[begin:end] = value
        expected.append((end - begin) * value)
    return values, offsets, torch.tensor(expected, dtype=torch.float32)


def _closure_values(function):
    """Expose retained private allocations for white-box race poisoning."""
    return {
        name: cell.cell_contents
        for name, cell in zip(
            function.__code__.co_freevars, function.__closure__
        )
    }


@pytest.mark.parametrize("resident_blocks", [1, 2, 3, 7, 168, 336])
def test_persistent_batch_boundaries_across_residencies(resident_blocks):
    """Cover both claim batch edges with under- and over-subscribed grids."""
    lengths = (
        [0]
        + [1] * 17
        + [33, 34, 4095, 4096]
        + [33] * 337
        + [4097, 8193, 12289, 16385, 32769]
    )
    host_values, host_offsets, expected = _integer_case(lengths)
    base_values = host_values.cuda()
    values = base_values.clone()
    offsets = torch.tensor(host_offsets, device="cuda", dtype=torch.int32)
    guarded_output = torch.full(
        (len(lengths) + 2,), -123456.0, device="cuda"
    )
    output = guarded_output[1:-1]
    prepared = _prepare_persistent_sum(
        values, offsets, output, resident_blocks=resident_blocks
    )

    assert prepared.warp_tasks == 18
    assert prepared.cta_tasks == 341
    assert prepared.partial_tasks == 23
    assert prepared.merge_tasks == 5
    assert prepared.resident_blocks == min(resident_blocks, 366)

    for factor in (1, 2, 3):
        values.copy_(base_values)
        values.mul_(factor)
        output.fill_(float("nan"))
        prepared.launch()
        torch.cuda.synchronize()

        torch.testing.assert_close(
            output.cpu(), expected * factor, rtol=0, atol=0
        )
        torch.testing.assert_close(
            guarded_output[[0, -1]].cpu(),
            torch.tensor([-123456.0, -123456.0]),
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize("resident_blocks", [2, 3])
def test_persistent_merge_never_observes_poisoned_scratch(resident_blocks):
    """Require device-wide scratch publication before the final merge."""
    lengths = [4097, 8193, 12289, 16385, 20481, 24577, 28673, 32769]
    offsets = _offsets(lengths)
    values = torch.ones(offsets[-1], device="cuda")
    device_offsets = torch.tensor(offsets, device="cuda", dtype=torch.int32)
    output = torch.empty(len(lengths), device="cuda")
    prepared = _prepare_persistent_sum(
        values,
        device_offsets,
        output,
        resident_blocks=resident_blocks,
    )
    scratch = _closure_values(prepared.launch)["scratch"]
    expected = torch.tensor(lengths, dtype=torch.float32)

    for _ in range(100):
        scratch.fill_(float("nan"))
        output.fill_(float("nan"))
        prepared.launch()
        torch.cuda.synchronize()
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("resident_blocks", [2, 7])
def test_persistent_poisoned_scratch_survives_graph_replay(resident_blocks):
    """Preserve publication ordering across repeated captured submissions."""
    lengths = [0, 1, 32, 33, 4097, 8193, 16385, 32769]
    offsets = _offsets(lengths)
    values = torch.ones(offsets[-1], device="cuda")
    device_offsets = torch.tensor(offsets, device="cuda", dtype=torch.int32)
    output = torch.empty(len(lengths), device="cuda")
    prepared = _prepare_persistent_sum(
        values,
        device_offsets,
        output,
        resident_blocks=resident_blocks,
    )
    scratch = _closure_values(prepared.launch)["scratch"]
    expected = torch.tensor(lengths, dtype=torch.float32)

    prepared.launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        prepared.launch()

    for _ in range(100):
        scratch.fill_(float("nan"))
        output.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


def test_persistent_randomized_plans_and_values():
    """Differentially stress empty, direct, split, and unequal merge plans."""
    rng = random.Random(180067)
    boundary_lengths = [
        0,
        1,
        2,
        31,
        32,
        33,
        127,
        4095,
        4096,
        4097,
        8193,
    ]
    residencies = [1, 2, 3, 7, 17, 84, 168, 169, 336]

    for _ in range(40):
        lengths = [
            (
                rng.choice(boundary_lengths)
                if rng.random() < 0.8
                else rng.randint(0, 20_000)
            )
            for _ in range(rng.randint(1, 180))
        ]
        host_values, host_offsets, expected = _integer_case(lengths)
        base_values = host_values.cuda()
        values = base_values.clone()
        offsets = torch.tensor(
            host_offsets, device="cuda", dtype=torch.int32
        )
        output = torch.full((len(lengths),), float("nan"), device="cuda")
        prepared = _prepare_persistent_sum(
            values,
            offsets,
            output,
            resident_blocks=rng.choice(residencies),
        )

        for factor in (1, 3):
            values.copy_(base_values)
            values.mul_(factor)
            output.fill_(float("nan"))
            prepared.launch()
            torch.cuda.synchronize()
            torch.testing.assert_close(
                output.cpu(), expected * factor, rtol=0, atol=0
            )


def test_persistent_capture_requires_initialized_task_storage(monkeypatch):
    """Reject capture before the one-time task-readiness handoff."""
    values = torch.ones(33, device="cuda")
    offsets = torch.tensor([0, 33], device="cuda", dtype=torch.int32)
    output = torch.empty(1, device="cuda")
    prepared = _prepare_persistent_sum(values, offsets, output)
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: True
    )

    with pytest.raises(RuntimeError, match="must launch once"):
        prepared.launch()
