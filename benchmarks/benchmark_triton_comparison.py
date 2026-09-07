# benchmarks/benchmark_triton_comparison.py
"""Compare Swage GPU paths with Triton and PyTorch baselines.

This is a research benchmark harness, not a CI gate. Triton is imported only
when the benchmark is executed; the project does not depend on Triton.
"""

import argparse
import json
import pathlib
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Iterable
from datetime import datetime, timezone

_WARMUPS = 25
_SAMPLES = 100
_BATCHED_LAUNCHES = 32
_SEGMENT_COUNT = 32_768
_SEED = 7
_WARP_MAX_ELEMENTS = 32
_SYNTHETIC_DISTRIBUTIONS = (
    "many-tiny",
    "uniform",
    "log-normal",
    "bimodal",
    "zipf-like",
    "few-huge",
    "one-outlier",
)
_REAL_TRACE_NAME = "soc-epinions1-outdegree-v1"


def _arguments():
    """Parse benchmark controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument(
        "--suite",
        choices=("all", "vadd", "segmented-sum"),
        default="all",
        help="Benchmark suite to run.",
    )
    parser.add_argument(
        "--samples", type=int, default=_SAMPLES, help="Timed samples per case."
    )
    parser.add_argument(
        "--warmups", type=int, default=_WARMUPS, help="Warmup launches."
    )
    return parser.parse_args()


def _git_metadata(root: pathlib.Path) -> dict[str, object]:
    """Return source provenance without requiring a clean worktree."""
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {"revision": revision, "worktree_clean": not dirty, "dirty": dirty}


def _median_iqr(values: Iterable[float]) -> dict[str, float]:
    """Return median and quartiles for one nonempty sample list."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("at least one timing sample is required")
    if len(ordered) == 1:
        quartiles = (ordered[0], ordered[0])
    else:
        inclusive = statistics.quantiles(ordered, n=4, method="inclusive")
        quartiles = (inclusive[0], inclusive[2])
    return {
        "median": statistics.median(ordered),
        "q1": quartiles[0],
        "q3": quartiles[1],
    }


def _rotating_orders(
    candidates: Iterable[str], rounds: int
) -> list[tuple[str, ...]]:
    """Return deterministic round-robin candidate orders."""
    names = tuple(candidates)
    if not names:
        raise ValueError("at least one timing candidate is required")
    if len(names) != len(set(names)):
        raise ValueError("timing candidate names must be unique")
    if rounds < 0:
        raise ValueError("timing rounds must be nonnegative")
    offsets = (
        round_index % len(names) for round_index in range(rounds)
    )
    return [names[offset:] + names[:offset] for offset in offsets]


def _order_position_counts(
    orders: Iterable[tuple[str, ...]]
) -> dict[str, list[int]]:
    """Count how often each candidate occupies each order position."""
    materialized = list(orders)
    if not materialized:
        return {}
    names = materialized[0]
    counts = {name: [0] * len(names) for name in names}
    expected = set(names)
    for order in materialized:
        if len(order) != len(names) or set(order) != expected:
            raise ValueError("every timing order must contain each candidate")
        for position, name in enumerate(order):
            counts[name][position] += 1
    return counts


def _warm_interleaved(
    torch,
    launches: dict[str, Callable[[], object]],
    warmups: int,
) -> None:
    """Warm candidates in the same deterministic rotating order."""
    for order in _rotating_orders(launches, warmups):
        for name in order:
            launches[name]()
    torch.cuda.synchronize()


def _interleaved_call_us(
    torch,
    launches: dict[str, Callable[[], object]],
    warmups: int,
    samples: int,
) -> dict[str, dict[str, object]]:
    """Measure synchronized calls in rotating candidate order."""
    _warm_interleaved(torch, launches, warmups)
    timings = {name: [] for name in launches}
    for order in _rotating_orders(launches, samples):
        for name in order:
            start = time.perf_counter_ns()
            launches[name]()
            torch.cuda.synchronize()
            end = time.perf_counter_ns()
            timings[name].append((end - start) / 1_000.0)
    return {
        name: {"samples_us": values, "summary_us": _median_iqr(values)}
        for name, values in timings.items()
    }


def _interleaved_batched_event_us(
    torch,
    launches: dict[str, Callable[[], object]],
    warmups: int,
    samples: int,
) -> dict[str, dict[str, object]]:
    """Measure back-to-back launch batches in rotating candidate order."""
    _warm_interleaved(torch, launches, warmups)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    timings = {name: [] for name in launches}
    for order in _rotating_orders(launches, samples):
        for name in order:
            start.record()
            for _ in range(_BATCHED_LAUNCHES):
                launches[name]()
            end.record()
            end.synchronize()
            elapsed_us = (
                start.elapsed_time(end) * 1_000.0 / _BATCHED_LAUNCHES
            )
            timings[name].append(elapsed_us)
    return {
        name: {"samples_us": values, "summary_us": _median_iqr(values)}
        for name, values in timings.items()
    }


def _capture_graphs(
    torch,
    launches: dict[str, Callable[[], object]],
    warmups: int,
) -> tuple[dict[str, object], dict[str, str]]:
    """Prepare every candidate graph before any timed replay."""
    _warm_interleaved(torch, launches, warmups)
    graphs = {}
    errors = {}
    for name, launch in launches.items():
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                for _ in range(_BATCHED_LAUNCHES):
                    launch()
        except RuntimeError as error:
            torch.cuda.synchronize()
            errors[name] = str(error)
            continue
        graphs[name] = graph
    return graphs, errors


def _interleaved_graph_us(
    torch,
    launches: dict[str, Callable[[], object]],
    warmups: int,
    samples: int,
) -> dict[str, dict[str, object]]:
    """Measure prepared 32-launch graphs in rotating candidate order."""
    graphs, errors = _capture_graphs(torch, launches, warmups)
    for order in _rotating_orders(launches, warmups):
        for name in order:
            if name in graphs:
                graphs[name].replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    timings = {name: [] for name in graphs}
    for order in _rotating_orders(launches, samples):
        for name in order:
            if name not in graphs:
                continue
            start.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            elapsed_us = (
                start.elapsed_time(end) * 1_000.0 / _BATCHED_LAUNCHES
            )
            timings[name].append(elapsed_us)
    results = {
        name: {
            "available": True,
            "samples_us": values,
            "summary_us": _median_iqr(values),
        }
        for name, values in timings.items()
    }
    results.update(
        {
            name: {"available": False, "error": error}
            for name, error in errors.items()
        }
    )
    return results


def _timings(
    torch,
    launches: dict[str, Callable[[], object]],
    warmups: int,
    samples: int,
) -> tuple[dict[str, object], dict[str, object]]:
    """Collect interleaved call, event, and graph measurements."""
    call = _interleaved_call_us(torch, launches, warmups, samples)
    batched = _interleaved_batched_event_us(
        torch, launches, warmups, samples
    )
    graph = _interleaved_graph_us(torch, launches, warmups, samples)
    results = {
        name: {
            "call": call[name],
            "batched_event": batched[name],
            "graph": graph[name],
        }
        for name in launches
    }
    orders = _rotating_orders(launches, samples)
    method = {
        "sampling": "deterministic_rotating_interleaved",
        "base_candidate_order": list(launches),
        "round_rotation": "left by round_index modulo candidate_count",
        "timed_rounds": samples,
        "order_position_counts": _order_position_counts(orders),
        "graph_preparation": (
            "all candidate graphs captured before interleaved timed replay"
        ),
        "units": {
            "call": "microseconds per synchronized Python call",
            "batched_event": (
                "microseconds per launch in a 32-launch CUDA-event batch"
            ),
            "graph": (
                "microseconds per launch in a captured 32-launch graph replay"
            ),
        },
    }
    return results, method


def _make_swage_vadd():
    """Define the canonical Swage vector-add kernel lazily."""
    import swage as sw
    import swage.language as sl

    @sw.jit
    def add_kernel(x_ptr, y_ptr, output_ptr, n, BLOCK: sl.constexpr):
        pid = sl.program_id(0)
        offsets = pid * BLOCK + sl.arange(0, BLOCK)
        mask = offsets < n
        x = sl.load(x_ptr + offsets, mask=mask, other=0.0)
        y = sl.load(y_ptr + offsets, mask=mask, other=0.0)
        sl.store(output_ptr + offsets, x + y, mask=mask)

    return add_kernel


def _make_triton_vadd():
    """Define a direct Triton vector-add baseline lazily."""
    import triton
    import triton.language as tl

    @triton.jit
    def add_kernel(x_ptr, y_ptr, output_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
        tl.store(output_ptr + offsets, x + y, mask=mask)

    return add_kernel


def _run_vadd(torch, warmups: int, samples: int) -> list[dict[str, object]]:
    """Benchmark fixed vector add across problem sizes."""
    swage_kernel = _make_swage_vadd()
    triton_kernel = _make_triton_vadd()
    results = []
    for exponent in (10, 12, 14, 16, 18, 20, 22):
        n = 1 << exponent
        swage_block = 256
        grid = ((n + swage_block - 1) // swage_block,)
        x = torch.randn(n, device="cuda", dtype=torch.float32)
        y = torch.randn(n, device="cuda", dtype=torch.float32)
        outputs = {
            "swage": torch.empty_like(x),
            "torch": torch.empty_like(x),
        }
        launches = {
            "swage": lambda: swage_kernel.launch(
                arguments={
                    "x_ptr": x,
                    "y_ptr": y,
                    "output_ptr": outputs["swage"],
                    "n": n,
                },
                constexprs={"BLOCK": swage_block},
                grid=grid,
            ),
            "torch": lambda: torch.add(x, y, out=outputs["torch"]),
        }
        for triton_block in (128, 256, 512, 1024):
            triton_grid = ((n + triton_block - 1) // triton_block,)
            output = torch.empty_like(x)
            name = f"triton_b{triton_block}"
            outputs[name] = output
            launches[name] = (
                lambda block=triton_block,
                grid=triton_grid,
                out=output: triton_kernel[grid](x, y, out, n, BLOCK=block)
            )
        for launch in launches.values():
            launch()
        torch.cuda.synchronize()
        expected = x + y
        for output_name, output in outputs.items():
            torch.testing.assert_close(
                output,
                expected,
                msg=lambda message, n=output_name: f"{n}: {message}",
            )
        timing_results, timing_method = _timings(
            torch, launches, warmups, samples
        )
        row = {
            "case": "vadd",
            "n": n,
            "swage_block": swage_block,
            "swage_grid": grid[0],
            "triton_sweep_blocks": [128, 256, 512, 1024],
            "launch_contract": {
                "swage": "BLOCK=256 for the vector-add campaign",
                "triton": "compile-time BLOCK, one program per block",
            },
            "timing_method": timing_method,
            "timings": timing_results,
        }
        results.append(row)
    return results


def _segmented_case_inputs(generate_lengths, load_real_trace):
    """Yield synthetic cases unchanged, followed by the real trace."""
    for name in _SYNTHETIC_DISTRIBUTIONS:
        yield {
            "distribution": name,
            "lengths": generate_lengths(name, _SEGMENT_COUNT, _SEED),
        }
    lengths, provenance = load_real_trace(_REAL_TRACE_NAME)
    yield {
        "distribution": _REAL_TRACE_NAME,
        "lengths": lengths,
        "trace_provenance": provenance,
    }


def _offsets_from_lengths(torch, lengths: list[int]):
    """Create host and device offsets from segment lengths."""
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    device_offsets = torch.tensor(offsets, device="cuda", dtype=torch.int32)
    return offsets, device_offsets


def _make_triton_segmented_sum():
    """Define a one-program-per-segment Triton sum baseline lazily."""
    import triton
    import triton.language as tl

    @triton.jit
    def sum_kernel(values, offsets, output, segment_count, BLOCK: tl.constexpr):
        sid = tl.program_id(0)
        begin = tl.load(offsets + sid)
        end = tl.load(offsets + sid + 1)
        idx = begin + tl.arange(0, BLOCK)
        mask = (idx < end) & (sid < segment_count)
        data = tl.load(values + idx, mask=mask, other=0.0)
        result = tl.sum(data, axis=0)
        tl.store(output + sid, result, mask=sid < segment_count)

    return sum_kernel


def _make_triton_matched_task_partition():
    """Define Triton kernels with the same host task partition as Swage."""
    import triton
    import triton.language as tl

    @triton.jit
    def packed_warp_kernel(
        values,
        offsets,
        output,
        task_ids,
        task_count,
        TASKS: tl.constexpr,
        WARP: tl.constexpr,
    ):
        lane = tl.arange(0, TASKS * WARP)
        slot = lane // WARP
        lane_in_slot = lane % WARP
        task_index = tl.program_id(0) * TASKS + slot
        active = task_index < task_count
        segment_id = tl.load(task_ids + task_index, mask=active, other=0)
        begin = tl.load(offsets + segment_id, mask=active, other=0)
        end = tl.load(offsets + segment_id + 1, mask=active, other=0)
        index = begin + lane_in_slot
        data = tl.load(values + index, mask=active & (index < end), other=0.0)
        matrix = tl.reshape(data, (TASKS, WARP))
        totals = tl.sum(matrix, axis=1)
        output_slot = tl.arange(0, TASKS)
        output_task = tl.program_id(0) * TASKS + output_slot
        output_active = output_task < task_count
        output_segment = tl.load(
            task_ids + output_task, mask=output_active, other=0
        )
        tl.store(output + output_segment, totals, mask=output_active)

    @triton.jit
    def cta_task_kernel(
        values, offsets, output, task_ids, task_count, BLOCK: tl.constexpr
    ):
        task_index = tl.program_id(0)
        segment_id = tl.load(task_ids + task_index)
        begin = tl.load(offsets + segment_id)
        end = tl.load(offsets + segment_id + 1)
        index = begin + tl.arange(0, BLOCK)
        data = tl.load(values + index, mask=index < end, other=0.0)
        tl.store(output + segment_id, tl.sum(data, axis=0))

    return packed_warp_kernel, cta_task_kernel


def _make_triton_fused_sum():
    """Define one Triton launch with Swage-like mixed task organization."""
    import triton
    import triton.language as tl

    @triton.jit
    def fused_sum_kernel(
        values,
        offsets,
        output,
        warp_task_ids,
        warp_task_count,
        cta_task_ids,
        cta_task_count,
        WARP_PROGRAMS: tl.constexpr,
        LOGICAL_LANES: tl.constexpr,
        SHORT_TASK_SLOTS: tl.constexpr,
        WARP_LANES: tl.constexpr,
        MAX_CTA_ELEMENTS: tl.constexpr,
    ):
        program_id = tl.program_id(0)
        lanes = tl.arange(0, LOGICAL_LANES)
        is_warp_program = program_id < WARP_PROGRAMS

        short_slot = lanes // WARP_LANES
        short_lane = lanes % WARP_LANES
        short_task = program_id * SHORT_TASK_SLOTS + short_slot
        short_active = (
            is_warp_program
            & (short_slot < SHORT_TASK_SLOTS)
            & (short_task < warp_task_count)
        )
        short_segment = tl.load(
            warp_task_ids + short_task, mask=short_active, other=0
        )
        short_begin = tl.load(
            offsets + short_segment, mask=short_active, other=0
        )
        short_end = tl.load(
            offsets + short_segment + 1, mask=short_active, other=0
        )
        short_index = short_begin + short_lane
        short_values = tl.load(
            values + short_index,
            mask=short_active & (short_index < short_end),
            other=0.0,
        )
        short_matrix = tl.reshape(
            short_values, (SHORT_TASK_SLOTS, WARP_LANES)
        )
        short_totals = tl.sum(short_matrix, axis=1)
        output_slots = tl.arange(0, SHORT_TASK_SLOTS)
        output_tasks = program_id * SHORT_TASK_SLOTS + output_slots
        output_active = (
            is_warp_program & (output_tasks < warp_task_count)
        )
        output_segments = tl.load(
            warp_task_ids + output_tasks, mask=output_active, other=0
        )
        tl.store(
            output + output_segments, short_totals, mask=output_active
        )

        cta_task = program_id - WARP_PROGRAMS
        cta_active = (~is_warp_program) & (cta_task < cta_task_count)
        cta_segment = tl.load(
            cta_task_ids + cta_task, mask=cta_active, other=0
        )
        cta_begin = tl.load(
            offsets + cta_segment, mask=cta_active, other=0
        )
        cta_end = tl.load(
            offsets + cta_segment + 1, mask=cta_active, other=0
        )
        cta_total = tl.zeros((LOGICAL_LANES,), dtype=tl.float32)
        for base in range(0, MAX_CTA_ELEMENTS, LOGICAL_LANES):
            cta_index = cta_begin + base + lanes
            cta_total += tl.load(
                values + cta_index,
                mask=cta_active & (cta_index < cta_end),
                other=0.0,
            )
        tl.store(
            output + cta_segment,
            tl.sum(cta_total, axis=0),
            mask=cta_active,
        )

    return fused_sum_kernel


def _triton_sum_configs(max_length: int) -> list[tuple[int, int]]:
    """Return legal Triton segmented-sum sweep configs."""
    configs = []
    for block in (32, 64, 128, 256, 512, 1024, 2048, 4096):
        if block < max_length:
            continue
        for warps in (1, 2, 4, 8):
            if warps <= block // 32:
                configs.append((block, warps))
    return configs


def _partition_lengths(lengths: Iterable[int]) -> tuple[list[int], list[int]]:
    """Partition segment IDs by the benchmark's fixed short-task boundary."""
    warp_ids = []
    cta_ids = []
    for segment_id, length in enumerate(lengths):
        if type(length) is not int or length < 0:
            raise ValueError("segment lengths must be nonnegative integers")
        if length <= _WARP_MAX_ELEMENTS:
            warp_ids.append(segment_id)
        elif length <= 4096:
            cta_ids.append(segment_id)
        else:
            raise ValueError("Triton comparison supports lengths up to 4096")
    return warp_ids, cta_ids


def _host_offsets(lengths: Iterable[int]) -> list[int]:
    """Materialize monotonically increasing host offsets."""
    offsets = [0]
    for length in lengths:
        if type(length) is not int or length < 0:
            raise ValueError("segment lengths must be nonnegative integers")
        offsets.append(offsets[-1] + length)
    return offsets


def _launch_matched_task_partition(
    packed_kernel,
    cta_kernel,
    values,
    offsets,
    output,
    warp_ids,
    cta_ids,
    *,
    cta_warps: int,
) -> None:
    """Launch the two-kernel matched task-partition Triton comparator."""
    if warp_ids.numel():
        packed_kernel[((warp_ids.numel() + 3) // 4,)](
            values,
            offsets,
            output,
            warp_ids,
            warp_ids.numel(),
            TASKS=4,
            WARP=32,
            num_warps=4,
        )
    if cta_ids.numel():
        cta_kernel[(cta_ids.numel(),)](
            values,
            offsets,
            output,
            cta_ids,
            cta_ids.numel(),
            BLOCK=4096,
            num_warps=cta_warps,
        )


def _wall_clock_interleaved_us(
    operations: dict[str, Callable[[int], object]],
    warmups: int,
    samples: int,
    *,
    unit: str = "microseconds per synchronized end-to-end operation",
) -> tuple[dict[str, object], dict[str, object]]:
    """Measure complete synchronized operations in rotating order."""
    for round_index, order in enumerate(
        _rotating_orders(operations, warmups)
    ):
        for name in order:
            operations[name](round_index)
    timings = {name: [] for name in operations}
    orders = _rotating_orders(operations, samples)
    for round_index, order in enumerate(orders):
        for name in order:
            start = time.perf_counter_ns()
            operations[name](round_index)
            end = time.perf_counter_ns()
            timings[name].append((end - start) / 1_000.0)
    results = {
        name: {"samples_us": values, "summary_us": _median_iqr(values)}
        for name, values in timings.items()
    }
    method = {
        "sampling": "deterministic_rotating_interleaved",
        "base_candidate_order": list(operations),
        "round_rotation": "left by round_index modulo candidate_count",
        "timed_rounds": samples,
        "order_position_counts": _order_position_counts(orders),
        "unit": unit,
    }
    return results, method


def _orchestration_measurements(
    torch,
    lengths: list[int],
    values,
    offsets,
    packed_kernel,
    cta_kernel,
    prepare_swage,
    warmups: int,
    samples: int,
) -> dict[str, object]:
    """Measure preparation and complete orchestration without compilation."""
    from swage._segmented_validation import (
        _validate_offsets,
        _validate_shapes,
    )

    def prepare_swage_only(device_offsets, output):
        return prepare_swage(
            values,
            device_offsets,
            output,
            warp_max_elements=_WARP_MAX_ELEMENTS,
        )

    def prepare_triton_only(device_offsets, output):
        _, _, validated_offsets = _validate_shapes(
            values, device_offsets, output, _validate_offsets
        )
        validated_lengths = [
            end - begin
            for begin, end in zip(
                validated_offsets, validated_offsets[1:]
            )
        ]
        warp_ids, cta_ids = _partition_lengths(validated_lengths)
        device_warp_ids = torch.tensor(
            warp_ids, device="cuda", dtype=torch.int32
        )
        device_cta_ids = torch.tensor(
            cta_ids, device="cuda", dtype=torch.int32
        )
        return device_warp_ids, device_cta_ids

    def run_swage(device_offsets):
        output = torch.empty(len(lengths), device="cuda")
        prepared = prepare_swage_only(device_offsets, output)
        prepared.launch_mixed()
        torch.cuda.synchronize()

    def run_triton(device_offsets):
        output = torch.empty(len(lengths), device="cuda")
        device_warp_ids, device_cta_ids = prepare_triton_only(
            device_offsets, output
        )
        _launch_matched_task_partition(
            packed_kernel,
            cta_kernel,
            values,
            device_offsets,
            output,
            device_warp_ids,
            device_cta_ids,
            cta_warps=1,
        )
        torch.cuda.synchronize()

    run_swage(offsets)
    run_triton(offsets)

    swage_preparation_output = torch.empty(len(lengths), device="cuda")
    triton_preparation_output = torch.empty(len(lengths), device="cuda")

    def prepare_swage_sample(_):
        prepared = prepare_swage_only(offsets, swage_preparation_output)
        torch.cuda.synchronize()
        return prepared

    def prepare_triton_sample(_):
        descriptors = prepare_triton_only(
            offsets, triton_preparation_output
        )
        torch.cuda.synchronize()
        return descriptors

    preparation_operations = {
        "swage_mixed": prepare_swage_sample,
        "triton_matched_task_partition": prepare_triton_sample,
    }
    preparation_timings, preparation_method = _wall_clock_interleaved_us(
        preparation_operations,
        warmups,
        samples,
        unit="microseconds per synchronized preparation",
    )

    fixed_operations = {
        "swage_mixed": lambda _: run_swage(offsets),
        "triton_matched_task_partition": lambda _: run_triton(offsets),
    }
    fixed_timings, fixed_method = _wall_clock_interleaved_us(
        fixed_operations, warmups, samples
    )

    def changing_offsets(round_index):
        shift = round_index % len(lengths)
        changed_lengths = lengths[shift:] + lengths[:shift]
        return torch.tensor(
            _host_offsets(changed_lengths),
            device="cuda",
            dtype=torch.int32,
        )

    changing_operations = {
        "swage_mixed": lambda round_index: run_swage(
            changing_offsets(round_index)
        ),
        "triton_matched_task_partition": lambda round_index: run_triton(
            changing_offsets(round_index)
        ),
    }
    changing_timings, changing_method = _wall_clock_interleaved_us(
        changing_operations, warmups, samples
    )
    complete_included = [
        "tensor validation",
        "host task classification",
        "descriptor tensor materialization",
        "output allocation",
        "kernel launch",
        "device synchronization",
    ]
    return {
        "preparation_only": {
            "geometry": "fixed offsets reused; output preallocated",
            "included": [
                "tensor validation",
                "host task classification",
                "descriptor tensor materialization",
                "device synchronization",
            ],
            "excluded": [
                "output allocation",
                "kernel launch",
                "artifact/JIT compilation",
            ],
            "compilation_excluded_after_explicit_warmup": True,
            "timing_method": preparation_method,
            "timings": preparation_timings,
        },
        "end_to_end": {
            "artifact_jit_warmup": (
                "one untimed complete operation per candidate before all "
                "preparation and end-to-end samples"
            ),
            "compilation_excluded": True,
            "graph_samples_combined": False,
            "warm_preparation": {
                "geometry": (
                    "fixed offsets reused; plan preparation repeated"
                ),
                "included": complete_included,
                "timing_method": fixed_method,
                "timings": fixed_timings,
            },
            "changing_geometry": {
                "geometry": (
                    "segment lengths deterministically rotated each round; "
                    "device offsets rematerialized inside the timed operation"
                ),
                "included": complete_included,
                "timing_method": changing_method,
                "timings": changing_timings,
            },
        },
    }


def _run_segmented_sum(
    torch, warmups: int, samples: int
) -> list[dict[str, object]]:
    """Benchmark private segmented sum against Triton and torch baselines."""
    from distributions import generate_lengths, summarize_lengths
    from real_traces import load_real_trace
    from swage._segmented_qualification import _prepare_planned_sum

    triton_kernel = _make_triton_segmented_sum()
    triton_packed_kernel, triton_cta_kernel = (
        _make_triton_matched_task_partition()
    )
    triton_fused_kernel = _make_triton_fused_sum()
    cases = _segmented_case_inputs(generate_lengths, load_real_trace)
    results = []
    for case in cases:
        name = case["distribution"]
        lengths = case["lengths"]
        statistics_summary = summarize_lengths(lengths)
        triton_configs = _triton_sum_configs(statistics_summary["max"])
        warp_ids, cta_ids = _partition_lengths(lengths)
        host_offsets, offsets = _offsets_from_lengths(torch, lengths)
        device_warp_ids = torch.tensor(
            warp_ids, device="cuda", dtype=torch.int32
        )
        device_cta_ids = torch.tensor(
            cta_ids, device="cuda", dtype=torch.int32
        )
        values = torch.ones(
            host_offsets[-1], device="cuda", dtype=torch.float32
        )
        expected = torch.tensor(lengths, device="cuda", dtype=torch.float32)
        outputs = {
            "swage_warp": torch.empty(_SEGMENT_COUNT, device="cuda"),
            "swage_cta": torch.empty(_SEGMENT_COUNT, device="cuda"),
            "swage_mixed": torch.empty(_SEGMENT_COUNT, device="cuda"),
            "triton_fused": torch.empty(_SEGMENT_COUNT, device="cuda"),
        }
        torch_output = {"value": None}
        prepared = _prepare_planned_sum(
            values,
            offsets,
            outputs["swage_mixed"],
            warp_max_elements=_WARP_MAX_ELEMENTS,
        )
        swage_warp = _prepare_planned_sum(
            values,
            offsets,
            outputs["swage_warp"],
            warp_max_elements=_WARP_MAX_ELEMENTS,
        ).launch_warp
        swage_cta = _prepare_planned_sum(
            values,
            offsets,
            outputs["swage_cta"],
            warp_max_elements=_WARP_MAX_ELEMENTS,
        ).launch_cta

        def launch_torch():
            torch_output["value"] = torch.segment_reduce(
                values, "sum", offsets=offsets
            )
            return torch_output["value"]

        warp_programs = (len(warp_ids) + 3) // 4
        fused_grid = (warp_programs + len(cta_ids),)

        def launch_fused():
            return triton_fused_kernel[fused_grid](
                values,
                offsets,
                outputs["triton_fused"],
                device_warp_ids,
                len(warp_ids),
                device_cta_ids,
                len(cta_ids),
                WARP_PROGRAMS=warp_programs,
                LOGICAL_LANES=128,
                SHORT_TASK_SLOTS=4,
                WARP_LANES=32,
                MAX_CTA_ELEMENTS=4096,
                num_warps=4,
            )

        launches = {
            "swage_warp": swage_warp,
            "swage_cta": swage_cta,
            "swage_mixed": prepared.launch_mixed,
            "torch": launch_torch,
            "triton_fused": launch_fused,
        }
        for block, warps in triton_configs:
            output = torch.empty(_SEGMENT_COUNT, device="cuda")
            launch_name = f"triton_b{block}_w{warps}"
            outputs[launch_name] = output
            launches[launch_name] = (
                lambda out=output, block=block, warps=warps: triton_kernel[
                    (_SEGMENT_COUNT,)
                ](
                    values,
                    offsets,
                    out,
                    _SEGMENT_COUNT,
                    BLOCK=block,
                    num_warps=warps,
                )
            )
        for warps in (1, 2, 4, 8):
            output = torch.empty(_SEGMENT_COUNT, device="cuda")
            launch_name = "triton_matched_task_partition"
            if warps != 1:
                launch_name = f"{launch_name}_w{warps}"
            outputs[launch_name] = output

            def launch_matched(out=output, cta_warps=warps):
                return _launch_matched_task_partition(
                    triton_packed_kernel,
                    triton_cta_kernel,
                    values,
                    offsets,
                    out,
                    device_warp_ids,
                    device_cta_ids,
                    cta_warps=cta_warps,
                )

            launches[launch_name] = launch_matched
        for launch in launches.values():
            launch()
        torch.cuda.synchronize()
        checked_outputs = {**outputs, "torch": torch_output["value"]}
        for output_name, output in checked_outputs.items():
            torch.testing.assert_close(
                output,
                expected,
                rtol=0,
                atol=0,
                msg=lambda message, n=output_name: f"{n}: {message}",
            )
        timing_results, timing_method = _timings(
            torch, launches, warmups, samples
        )
        orchestration = _orchestration_measurements(
            torch,
            lengths,
            values,
            offsets,
            triton_packed_kernel,
            triton_cta_kernel,
            _prepare_planned_sum,
            warmups,
            samples,
        )
        row = {
            "case": "segmented-sum",
            "distribution": name,
            "segment_count": _SEGMENT_COUNT,
            "statistics": statistics_summary,
            "triton_sweep_configs": [
                {"block": block, "num_warps": warps}
                for block, warps in triton_configs
            ],
            "matched_task_partition_triton": {
                "comparison": (
                    "same host task partition; not identical execution"
                ),
                "launches": (
                    "one packed short-task kernel plus one CTA-task kernel "
                    "when both partitions are nonempty"
                ),
                "warp_threshold_elements": _WARP_MAX_ELEMENTS,
                "warp_tasks": len(warp_ids),
                "cta_tasks": len(cta_ids),
                "short_tasks_per_program": 4,
                "cta_block_elements": 4096,
                "primary_cta_num_warps": 1,
                "cta_num_warps_sweep": [1, 2, 4, 8],
            },
            "triton_fused_contract": {
                "result_name": "triton_fused",
                "launch_count": 1,
                "logical_lanes_per_program": 128,
                "short_task_slots": 4,
                "lanes_per_short_task": 32,
                "program_order": (
                    "packed short-task programs first, then one CTA task "
                    "per later program"
                ),
                "cta_accumulation": (
                    "128-lane block-stride loads through 4096 elements"
                ),
                "maximum_segment_length": 4096,
                "physical_num_warps": 4,
                "warp_programs": warp_programs,
                "cta_programs": len(cta_ids),
                "grid_programs": fused_grid[0],
            },
            "timing_method": timing_method,
            "timings": timing_results,
            "preparation_only": orchestration["preparation_only"],
            "end_to_end": orchestration["end_to_end"],
        }
        if "trace_provenance" in case:
            row["trace_provenance"] = case["trace_provenance"]
        results.append(row)
    return results


def main():
    """Run the selected comparison benchmark and write JSON evidence."""
    arguments = _arguments()
    if arguments.samples <= 0 or arguments.warmups < 0:
        raise ValueError("samples must be positive and warmups nonnegative")

    import torch
    import triton
    from swage import _cuda_backend

    if not torch.cuda.is_available():
        raise RuntimeError("benchmark requires CUDA-enabled PyTorch")
    root = pathlib.Path(__file__).resolve().parents[1]
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    capability = torch.cuda.get_device_capability(device)
    result = {
        "benchmark": "swage-triton-comparison",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "source": _git_metadata(root),
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "pytorch": torch.__version__,
            "pytorch_cuda": torch.version.cuda,
            "triton": triton.__version__,
            "cuda_driver": _cuda_backend.driver_version(),
            "gpu": torch.cuda.get_device_name(device),
            "compute_capability": f"sm_{capability[0]}{capability[1]}",
            "multiprocessors": properties.multi_processor_count,
            "total_memory_bytes": properties.total_memory,
        },
        "methodology": {
            "warmups_per_candidate_per_measurement": arguments.warmups,
            "samples_per_candidate_per_measurement": arguments.samples,
            "candidate_sampling": (
                "deterministic rotating/interleaved order"
            ),
            "batched_launches": _BATCHED_LAUNCHES,
            "graph_replay_launches": _BATCHED_LAUNCHES,
            "graph_capture_before_interleaved_replay": True,
            "kernel_timing_compilation_excluded": True,
            "end_to_end_compilation_excluded_after_explicit_warmup": True,
            "preparation_only_output_preallocated": True,
            "preparation_only_kernel_launch_excluded": True,
            "end_to_end_not_combined_with_graph_samples": True,
            "correctness_checked_before_timing": True,
            "triton_dependency": "optional runtime import; not a project dep",
        },
        "results": [],
    }
    if arguments.suite in {"all", "vadd"}:
        result["results"].extend(
            _run_vadd(torch, arguments.warmups, arguments.samples)
        )
    if arguments.suite in {"all", "segmented-sum"}:
        result["results"].extend(
            _run_segmented_sum(torch, arguments.warmups, arguments.samples)
        )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(arguments.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
