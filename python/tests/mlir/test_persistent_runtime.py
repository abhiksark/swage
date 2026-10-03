# python/tests/mlir/test_persistent_runtime.py
"""Adversarial GPU qualification for the private persistent scheduler."""

import os
import pathlib
import random
import subprocess
import sys
import textwrap

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
    for segment_id, (begin, end) in enumerate(zip(offsets, offsets[1:])):
        value = float(segment_id % 7 + 1)
        values[begin:end] = value
        expected.append((end - begin) * value)
    return values, offsets, torch.tensor(expected, dtype=torch.float32)


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
    guarded_output = torch.full((len(lengths) + 2,), -123456.0, device="cuda")
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
        prepared.launch_persistent()
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
    scratch = prepared.scratch_buffers["partials"]
    expected = torch.tensor(lengths, dtype=torch.float32)

    for _ in range(100):
        scratch.fill_(float("nan"))
        output.fill_(float("nan"))
        prepared.launch_persistent()
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
    scratch = prepared.scratch_buffers["partials"]
    expected = torch.tensor(lengths, dtype=torch.float32)

    prepared.launch_persistent()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        prepared.launch_persistent()

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
        offsets = torch.tensor(host_offsets, device="cuda", dtype=torch.int32)
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
            prepared.launch_persistent()
            torch.cuda.synchronize()
            torch.testing.assert_close(
                output.cpu(), expected * factor, rtol=0, atol=0
            )


def test_persistent_nonfinite_values_cross_every_worker_policy():
    """Preserve IEEE sum behavior through direct, split, and merge work."""
    lengths = [0, 32, 33, 4097, 8193]
    offsets = _offsets(lengths)
    values = torch.zeros(offsets[-1], device="cuda")
    values[offsets[1]] = float("nan")
    values[offsets[2] : offsets[3]] = float("inf")
    values[offsets[3] : offsets[4]] = float("-inf")
    values[offsets[4]] = float("inf")
    values[offsets[4] + 4096] = float("-inf")
    device_offsets = torch.tensor(offsets, device="cuda", dtype=torch.int32)
    output = torch.empty(len(lengths), device="cuda")

    _prepare_persistent_sum(
        values, device_offsets, output, resident_blocks=2
    ).launch_persistent()

    expected = torch.tensor(
        [0.0, float("nan"), float("inf"), float("-inf"), float("nan")]
    )
    torch.testing.assert_close(
        output.cpu(), expected, rtol=0, atol=0, equal_nan=True
    )


def test_persistent_capture_requires_initialized_task_storage(monkeypatch):
    """Reject capture before the one-time task-readiness handoff."""
    values = torch.ones(33, device="cuda")
    offsets = torch.tensor([0, 33], device="cuda", dtype=torch.int32)
    output = torch.empty(1, device="cuda")
    prepared = _prepare_persistent_sum(values, offsets, output)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    with pytest.raises(RuntimeError, match="must launch once"):
        prepared.launch_persistent()


def test_persistent_rejects_overlapping_cross_stream_launches():
    """Fence one prepared object's mutable counters without synchronizing."""
    values = torch.ones(33, device="cuda")
    offsets = torch.tensor([0, 33], dtype=torch.int32, device="cuda")
    output = torch.empty(1, device="cuda")
    prepared = _prepare_persistent_sum(values, offsets, output)
    first = torch.cuda.Stream()
    second = torch.cuda.Stream()

    with torch.cuda.stream(first):
        torch.cuda._sleep(2_000_000_000)
        prepared.launch_persistent()
    with torch.cuda.stream(second):
        with pytest.raises(RuntimeError, match="cannot run concurrently"):
            prepared.launch_persistent()

    first.synchronize()
    with torch.cuda.stream(second):
        prepared.launch_persistent()


def test_persistent_cache_hit_launches_in_a_clean_child_process(
    tmp_path, monkeypatch
):
    """Populate version-3 PTX in the parent and launch it in a child."""
    from swage import _runtime

    cache_dir = tmp_path / "cache"
    identity = {
        "revision": "clean-child-cache-test",
        "clean": True,
        "llvm": "llvmorg-test",
    }
    monkeypatch.setenv("SWAGE_CACHE_DIR", str(cache_dir))
    monkeypatch.setattr(_runtime, "_cached_identity", lambda: identity)
    _runtime._artifact_cache.clear()
    _runtime._compilations.clear()

    lengths = [0, 1, 33, 4097]
    values = torch.ones(sum(lengths), device="cuda")
    offsets = torch.tensor(_offsets(lengths), device="cuda", dtype=torch.int32)
    output = torch.empty(len(lengths), device="cuda")
    prepared = _prepare_persistent_sum(values, offsets, output)
    prepared.launch_persistent()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        output.cpu(), torch.tensor(lengths, dtype=torch.float32)
    )
    assert any(cache_dir.iterdir())

    script = textwrap.dedent(
        """
        import torch

        from swage import _cuda_backend, _runtime
        from swage._segmented_qualification import _prepare_persistent_sum


        def fail_compile(*_args, **_kwargs):
            raise RuntimeError("compiler called on child-process cache hit")


        _runtime._cached_identity = lambda: {
            "revision": "clean-child-cache-test",
            "clean": True,
            "llvm": "llvmorg-test",
        }
        _cuda_backend.CUDA_BACKEND.compile = fail_compile
        lengths = [0, 1, 33, 4097]
        host_offsets = [0]
        for length in lengths:
            host_offsets.append(host_offsets[-1] + length)
        values = torch.ones(sum(lengths), device="cuda")
        offsets = torch.tensor(
            host_offsets, device="cuda", dtype=torch.int32
        )
        output = torch.empty(len(lengths), device="cuda")
        prepared = _prepare_persistent_sum(values, offsets, output)
        prepared.launch_persistent()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            output.cpu(), torch.tensor(lengths, dtype=torch.float32)
        )
        print("child CUDA cache result matches")
        """
    )
    root = pathlib.Path(__file__).parents[3]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(root / "python"), str(root / "build" / "python_packages"))
    )
    environment["SWAGE_CACHE_DIR"] = str(cache_dir)
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.stdout.strip() == "child CUDA cache result matches"
