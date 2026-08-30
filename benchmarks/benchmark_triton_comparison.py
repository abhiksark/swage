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
    """Return median and quartiles for one sample list."""
    ordered = sorted(values)
    return {
        "median": statistics.median(ordered),
        "q1": statistics.quantiles(ordered, n=4, method="inclusive")[0],
        "q3": statistics.quantiles(ordered, n=4, method="inclusive")[2],
    }


def _call_us(torch, launch: Callable[[], object], warmups: int,
             samples: int) -> dict[str, object]:
    """Measure synchronized Python-call latency in microseconds."""
    for _ in range(warmups):
        launch()
    torch.cuda.synchronize()
    timings = []
    for _ in range(samples):
        start = time.perf_counter_ns()
        launch()
        torch.cuda.synchronize()
        end = time.perf_counter_ns()
        timings.append((end - start) / 1_000.0)
    return {"samples_us": timings, "summary_us": _median_iqr(timings)}


def _batched_event_us(torch, launch: Callable[[], object], warmups: int,
                      samples: int) -> dict[str, object]:
    """Measure CUDA-event time per launch in a back-to-back batch."""
    for _ in range(warmups):
        launch()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    timings = []
    for _ in range(samples):
        start.record()
        for _ in range(_BATCHED_LAUNCHES):
            launch()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) * 1_000.0 / _BATCHED_LAUNCHES)
    return {"samples_us": timings, "summary_us": _median_iqr(timings)}


def _graph_us(torch, launch: Callable[[], object], warmups: int,
              samples: int) -> dict[str, object]:
    """Measure one launch through replay of a captured 32-launch graph."""
    for _ in range(warmups):
        launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            for _ in range(_BATCHED_LAUNCHES):
                launch()
    except RuntimeError as error:
        torch.cuda.synchronize()
        return {"available": False, "error": str(error)}
    for _ in range(warmups):
        graph.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    timings = []
    for _ in range(samples):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) * 1_000.0 / _BATCHED_LAUNCHES)
    return {
        "available": True,
        "samples_us": timings,
        "summary_us": _median_iqr(timings),
    }


def _timings(torch, launch: Callable[[], object], warmups: int,
             samples: int) -> dict[str, object]:
    """Collect host, batched-event, and graph-replay measurements."""
    return {
        "call": _call_us(torch, launch, warmups, samples),
        "batched_event": _batched_event_us(
            torch, launch, warmups, samples
        ),
        "graph": _graph_us(torch, launch, warmups, samples),
    }


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
                lambda block=triton_block, grid=triton_grid, out=output:
                triton_kernel[grid](x, y, out, n, BLOCK=block)
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
        row = {
            "case": "vadd",
            "n": n,
            "swage_block": swage_block,
            "swage_grid": grid[0],
            "triton_sweep_blocks": [128, 256, 512, 1024],
        }
        row["timings"] = {
            name: _timings(torch, launch, warmups, samples)
            for name, launch in launches.items()
        }
        results.append(row)
    return results


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
    def sum_kernel(values, offsets, output, segment_count,
                   BLOCK: tl.constexpr):
        sid = tl.program_id(0)
        begin = tl.load(offsets + sid)
        end = tl.load(offsets + sid + 1)
        idx = begin + tl.arange(0, BLOCK)
        mask = (idx < end) & (sid < segment_count)
        data = tl.load(values + idx, mask=mask, other=0.0)
        result = tl.sum(data, axis=0)
        tl.store(output + sid, result, mask=sid < segment_count)

    return sum_kernel


def _make_triton_planned_sum():
    """Define Triton kernels consuming host-classified task IDs."""
    import triton
    import triton.language as tl

    @triton.jit
    def packed_warp_kernel(values, offsets, output, task_ids, task_count,
                           TASKS: tl.constexpr, WARP: tl.constexpr):
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
    def cta_task_kernel(values, offsets, output, task_ids, task_count,
                        BLOCK: tl.constexpr):
        task_index = tl.program_id(0)
        segment_id = tl.load(task_ids + task_index)
        begin = tl.load(offsets + segment_id)
        end = tl.load(offsets + segment_id + 1)
        index = begin + tl.arange(0, BLOCK)
        data = tl.load(values + index, mask=index < end, other=0.0)
        tl.store(output + segment_id, tl.sum(data, axis=0))

    return packed_warp_kernel, cta_task_kernel


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


def _run_segmented_sum(torch, warmups: int,
                       samples: int) -> list[dict[str, object]]:
    """Benchmark private segmented sum against Triton and torch baselines."""
    from distributions import generate_lengths, summarize_lengths
    from swage._segmented_qualification import _prepare_planned_sum

    triton_kernel = _make_triton_segmented_sum()
    triton_packed_kernel, triton_cta_kernel = _make_triton_planned_sum()
    distributions = (
        "many-tiny",
        "uniform",
        "log-normal",
        "bimodal",
        "zipf-like",
        "few-huge",
        "one-outlier",
    )
    results = []
    for name in distributions:
        lengths = generate_lengths(name, _SEGMENT_COUNT, _SEED)
        statistics_summary = summarize_lengths(lengths)
        triton_configs = _triton_sum_configs(statistics_summary["max"])
        warp_ids = [
            index
            for index, length in enumerate(lengths)
            if length <= _WARP_MAX_ELEMENTS
        ]
        cta_ids = [
            index
            for index, length in enumerate(lengths)
            if length > _WARP_MAX_ELEMENTS
        ]
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
        ).warp
        swage_cta = _prepare_planned_sum(
            values,
            offsets,
            outputs["swage_cta"],
            warp_max_elements=_WARP_MAX_ELEMENTS,
        ).cta

        def launch_torch():
            torch_output["value"] = torch.segment_reduce(
                values, "sum", offsets=offsets
            )
            return torch_output["value"]

        launches = {
            "swage_warp": swage_warp,
            "swage_cta": swage_cta,
            "swage_mixed": prepared.mixed,
            "torch": launch_torch,
        }
        for block, warps in triton_configs:
            output = torch.empty(_SEGMENT_COUNT, device="cuda")
            launch_name = f"triton_b{block}_w{warps}"
            outputs[launch_name] = output
            launches[launch_name] = (
                lambda out=output, block=block, warps=warps:
                triton_kernel[(_SEGMENT_COUNT,)](
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
            launch_name = f"triton_planned_w{warps}"
            outputs[launch_name] = output

            def launch_planned(out=output, cta_warps=warps):
                if warp_ids:
                    triton_packed_kernel[((len(warp_ids) + 3) // 4,)](
                        values,
                        offsets,
                        out,
                        device_warp_ids,
                        len(warp_ids),
                        TASKS=4,
                        WARP=32,
                        num_warps=4,
                    )
                if cta_ids:
                    triton_cta_kernel[(len(cta_ids),)](
                        values,
                        offsets,
                        out,
                        device_cta_ids,
                        len(cta_ids),
                        BLOCK=4096,
                        num_warps=cta_warps,
                    )
                return None

            launches[launch_name] = launch_planned
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
        row = {
            "case": "segmented-sum",
            "distribution": name,
            "segment_count": _SEGMENT_COUNT,
            "statistics": statistics_summary,
            "triton_sweep_configs": [
                {"block": block, "num_warps": warps}
                for block, warps in triton_configs
            ],
            "triton_planned": {
                "warp_threshold": _WARP_MAX_ELEMENTS,
                "warp_tasks": len(warp_ids),
                "cta_tasks": len(cta_ids),
                "warp_tasks_per_program": 4,
                "cta_block": 4096,
                "cta_num_warps_sweep": [1, 2, 4, 8],
            },
            "timings": {
                launch_name: _timings(torch, launch, warmups, samples)
                for launch_name, launch in launches.items()
            },
        }
        results.append(row)
    return results


def main():
    """Run the selected comparison benchmark and write JSON evidence."""
    arguments = _arguments()
    if arguments.samples <= 0 or arguments.warmups < 0:
        raise ValueError("samples must be positive and warmups nonnegative")

    import torch
    import triton
    from swage import _runtime

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
            "cuda_driver": _runtime.driver_version(),
            "gpu": torch.cuda.get_device_name(device),
            "compute_capability": f"sm_{capability[0]}{capability[1]}",
            "multiprocessors": properties.multi_processor_count,
            "total_memory_bytes": properties.total_memory,
        },
        "methodology": {
            "warmups": arguments.warmups,
            "samples": arguments.samples,
            "batched_launches": _BATCHED_LAUNCHES,
            "graph_replay_launches": _BATCHED_LAUNCHES,
            "compilation_excluded": True,
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
