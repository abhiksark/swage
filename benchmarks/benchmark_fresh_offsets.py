# benchmarks/benchmark_fresh_offsets.py
"""Measure segmented sum when every call sees a new offsets layout.

The frozen benchmarks prepare one layout and time repeated launches of it.
This harness times the other regime: every iteration takes an offsets layout
that no earlier iteration used, and the timed region runs from offsets in to
result out. Preparation is therefore inside the timed region on purpose.

It is a research harness, not a CI gate. Triton is imported only when the
benchmark runs; the project does not depend on Triton, and the looped Triton
candidates are skipped when it is not installed.

Run with PYTHONPATH=python:build/python_packages and --output result.json.
A full run requires a clean worktree. Use --smoke to check the harness on a
few tiny layouts; a smoke record is labelled as not evidence.
"""

import argparse
import itertools
import json
import pathlib
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import NamedTuple

from benchmark_triton_comparison import (
    _exact_values,
    _launch_triton_looped,
    _make_triton_looped_sum,
    _median_iqr,
    _triton_looped_configs,
)
from distributions import generate_lengths, summarize_lengths

_SEGMENT_COUNT = 32_768
_SEED = 7
_WARMUPS = 5
_SAMPLES = 100
_SMOKE_SEGMENT_COUNT = 2_048
_SMOKE_WARMUPS = 1
_SMOKE_SAMPLES = 3
_WARP_MAX_ELEMENTS = 32
_DISTRIBUTIONS = (
    "uniform",
    "log-normal",
    "bimodal",
    "zipf-like",
    "many-tiny",
    "few-huge",
    "one-outlier",
    "alternating-empty",
    "power-law",
)
_REQUIRED_ENVIRONMENT = ("gpu", "compute_capability", "cuda_driver", "pytorch")


class _Layout(NamedTuple):
    """One host offsets layout and the seed that generated it."""

    seed: int
    lengths: list[int]
    offsets: list[int]


class _DeviceLayout(NamedTuple):
    """One uploaded layout with its inputs and its PyTorch reference."""

    values: object
    offsets: object
    expected: object


def _arguments(argv=None):
    """Parse the output path, the smoke switch, and the sample counts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Run a few tiny layouts to check the harness. Ignores --samples "
            "and --warmups, allows a dirty worktree, and labels the record "
            "as not evidence."
        ),
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=_SAMPLES,
        help="Timed fresh layouts per distribution.",
    )
    parser.add_argument(
        "--warmups",
        type=int,
        default=_WARMUPS,
        help="Untimed fresh layouts per distribution.",
    )
    arguments = parser.parse_args(argv)
    if arguments.samples < 2 or arguments.warmups < 1:
        parser.error("samples must be >= 2 and warmups must be >= 1")
    return arguments


def _sizes(arguments):
    """Return the segment count, warmups, and samples for this run."""
    if arguments.smoke:
        return _SMOKE_SEGMENT_COUNT, _SMOKE_WARMUPS, _SMOKE_SAMPLES
    return _SEGMENT_COUNT, arguments.warmups, arguments.samples


def _git_metadata(root, *, allow_dirty):
    """Return the source revision, rejecting dirty evidence.

    Args:
        root: Repository root.
        allow_dirty: Whether a modified or untracked worktree is accepted.
            Only smoke runs pass True, and their record says so.

    Returns:
        The revision, the clean flag, and the porcelain status lines.

    Raises:
        RuntimeError: If the worktree is dirty and that is not allowed.
    """
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
    if dirty and not allow_dirty:
        raise RuntimeError(
            "fresh-offsets benchmark requires a clean source worktree"
        )
    return {"revision": revision, "worktree_clean": not dirty, "dirty": dirty}


def _check_output(root, output, *, smoke):
    """Keep smoke records out of the committed evidence directory."""
    evidence = (root / "benchmarks" / "results").resolve()
    if smoke and output.resolve().is_relative_to(evidence):
        raise ValueError(
            "smoke records must not be written under benchmarks/results"
        )


def _optional_triton():
    """Return the Triton module, or None when it is not installed."""
    try:
        import triton
    except ImportError:
        return None
    return triton


def _nvidia_driver():
    """Return the NVIDIA driver version, or None when it is unavailable."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    versions = result.stdout.split()
    if result.returncode or not versions:
        return None
    return versions[0]


def _candidate_names(*, triton_available):
    """Return the names of the candidates this run times."""
    names = ["swage_mixed", "torch"]
    if triton_available:
        names.extend(
            f"triton_looped_b{block}_w{warps}"
            for block, warps in _triton_looped_configs()
        )
    return tuple(names)


def _layout_pool(name, count, size, seed):
    """Generate distinct offsets layouts, one per benchmark iteration.

    Args:
        name: Distribution name accepted by ``generate_lengths``.
        count: Segments per layout.
        size: Number of layouts, warmup and timed iterations together.
        seed: Seed of the first layout; layout ``i`` uses ``seed + i``.

    Returns:
        The layouts in iteration order.

    Raises:
        ValueError: If two layouts have the same offsets.
    """
    pool = []
    for layout_seed in range(seed, seed + size):
        lengths = generate_lengths(name, count, layout_seed)
        offsets = [0, *itertools.accumulate(lengths)]
        pool.append(_Layout(layout_seed, lengths, offsets))
    if len({tuple(layout.offsets) for layout in pool}) != size:
        raise ValueError(
            f"{name} pool of {size} layouts with {count} segments is not "
            "distinct; every iteration needs its own layout"
        )
    return pool


def _candidate_order(names, seed, distribution, iteration):
    """Return one iteration's seeded random candidate order.

    A rotation would give every candidate the same predecessor in almost
    every iteration, so whatever one candidate leaves behind would always
    fall on the same neighbour. A fresh permutation spreads it.

    Args:
        names: Candidate names.
        seed: The run seed.
        distribution: Distribution name of the row.
        iteration: Iteration index within the row, warmups included.

    Returns:
        The seed string given to ``random.Random`` and the permuted names.
    """
    order_seed = f"{seed}:{distribution}:{iteration}"
    order = list(names)
    random.Random(order_seed).shuffle(order)
    return order_seed, order


def _measure(
    layouts,
    orders,
    candidates,
    check,
    *,
    warmups,
    synchronize,
    clock=time.perf_counter_ns,
):
    """Time every candidate once on each layout, from offsets in to result.

    Args:
        layouts: One layout per iteration; none is used twice.
        orders: The candidate order for each iteration.
        candidates: Callables by name, each taking a layout and returning
            its result.
        check: Callable taking the candidate name, its result, and the
            layout; it raises when the result is wrong.
        warmups: Leading iterations that are flagged as not timed.
        synchronize: Callable that waits for all device work.
        clock: Monotonic nanosecond clock.

    Returns:
        One entry per iteration with its timed flag, the candidate order it
        ran, and the microsecond sample of each candidate in that order.
    """
    iterations = []
    for index, (layout, order) in enumerate(zip(layouts, orders, strict=True)):
        samples = {}
        for name in order:
            synchronize()
            start = clock()
            result = candidates[name](layout)
            synchronize()
            elapsed = (clock() - start) / 1_000.0
            check(name, result, layout)
            samples[name] = elapsed
        iterations.append(
            {
                "timed": index >= warmups,
                "candidate_order": list(order),
                "samples_us": samples,
            }
        )
    return iterations


def _timed_samples(iterations, names):
    """Return the timed samples of each candidate in iteration order."""
    return {
        name: [
            iteration["samples_us"][name]
            for iteration in iterations
            if iteration["timed"]
        ]
        for name in names
    }


def _environment(torch, *, cuda_driver, nvidia_driver, triton_version):
    """Return the machine and library identity for the record."""
    device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device)
    properties = torch.cuda.get_device_properties(device)
    return {
        "platform": platform.platform(),
        "python": sys.version,
        "pytorch": torch.__version__,
        "pytorch_cuda": torch.version.cuda,
        "triton": triton_version,
        "cuda_driver": cuda_driver,
        "nvidia_driver": nvidia_driver,
        "gpu": torch.cuda.get_device_name(device),
        "compute_capability": f"sm_{capability[0]}{capability[1]}",
        "multiprocessors": properties.multi_processor_count,
        "total_memory_bytes": properties.total_memory,
    }


def _configuration(*, segment_count, warmups, samples, triton_available):
    """Return the benchmark contract, including what the timer covers."""
    timed_region = {
        "swage_mixed": (
            "_prepare_planned_sum (offset validation and transfer to the "
            "host, classification, whatever kernel compilation and module "
            "loading the preparation path performs at this revision, task "
            "upload), then the mixed launch and synchronize"
        ),
        "torch": (
            "torch.segment_reduce on the device offsets with its output "
            "allocation, then synchronize"
        ),
    }
    if triton_available:
        timed_region["triton_looped"] = (
            "one launch of the looped kernel, then synchronize; it needs "
            "no host classification"
        )
    return {
        "distributions": list(_DISTRIBUTIONS),
        "segment_count": segment_count,
        "seed": _SEED,
        "layout_seeds": "seed plus the iteration index, warmups first",
        "iterations": (
            "every iteration in run order, warmups included and flagged as "
            "not timed, with its layout, candidate order, and samples; "
            "raw_samples_us repeats the timed samples by candidate"
        ),
        "warmups": warmups,
        "samples": samples,
        "layouts_per_distribution": warmups + samples,
        "layout_reuse": "none; every iteration takes the next unused layout",
        "candidates": list(_candidate_names(triton_available=triton_available)),
        "candidate_order": (
            "a new random permutation every iteration, shuffled by "
            "random.Random(order_seed) with order_seed "
            "'<seed>:<distribution>:<iteration>'; the seed and the order "
            "are recorded with each iteration"
        ),
        "warp_max_elements": _WARP_MAX_ELEMENTS,
        "swage_policy": (
            "_prepare_planned_sum with select_schedule=False; the "
            "preparation builds the warp, CTA, and mixed policies and only "
            "mixed is launched"
        ),
        "triton_looped": (
            "every block and warp configuration is timed; none is selected"
            if triton_available
            else "skipped: Triton is not installed"
        ),
        "values": (
            "CPU seeded randint(1, 8) / 4, float32, never zero; one buffer "
            "per distribution, each layout reads its prefix"
        ),
        "correctness": (
            "every candidate in every iteration, warmups included, exact "
            "against CPU torch.segment_reduce"
        ),
        "clock": "time.perf_counter_ns between two device synchronizations",
        "timed_region": timed_region,
        "swage_mixed_prepare_samples_us": (
            "the part of each swage_mixed sample spent inside "
            "_prepare_planned_sum"
        ),
        "excluded": [
            "layout generation",
            "offsets and values upload",
            "reference computation and the correctness check",
            "output allocation for swage_mixed and triton_looped",
            "CUDA, native compiler, and Triton initialization (warmups)",
        ],
    }


def _record(*, source, environment, configuration, results, smoke):
    """Assemble the JSON record, refusing one that cannot be attributed.

    Args:
        source: Revision and worktree state from ``_git_metadata``.
        environment: Machine identity from ``_environment``.
        configuration: Benchmark contract from ``_configuration``.
        results: One row per distribution.
        smoke: Whether this was a smoke run.

    Returns:
        The complete record.

    Raises:
        ValueError: If provenance is missing, or the worktree is dirty
            outside a smoke run.
    """
    missing = [
        field
        for field in _REQUIRED_ENVIRONMENT
        if environment.get(field) is None
    ]
    if "triton" not in environment:
        missing.append("triton")
    if not source.get("revision"):
        missing.append("revision")
    if "worktree_clean" not in source:
        missing.append("worktree_clean")
    if missing:
        raise ValueError(
            f"record is missing provenance fields: {', '.join(missing)}"
        )
    if not smoke and not source["worktree_clean"]:
        raise ValueError("a full record requires a clean source worktree")
    return {
        "benchmark": "fresh-offsets-segmented-sum",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "status": (
            "smoke run; not evidence"
            if smoke
            else "research run; no performance gate"
        ),
        "smoke": smoke,
        "source": source,
        "environment": environment,
        "configuration": configuration,
        "results": results,
    }


def _upload(torch, pool, device):
    """Upload one pool and compute each layout's PyTorch reference."""
    largest_total = max(layout.offsets[-1] for layout in pool)
    host_values = _exact_values(torch, largest_total)
    values = host_values.to(device)
    uploaded = []
    for layout in pool:
        total = layout.offsets[-1]
        host_offsets = torch.tensor(layout.offsets, dtype=torch.int32)
        expected = torch.segment_reduce(
            host_values[:total], "sum", offsets=host_offsets
        )
        uploaded.append(
            _DeviceLayout(
                values[:total], host_offsets.to(device), expected.to(device)
            )
        )
    return uploaded


def _looped_candidate(kernel, output, segment_count, block, warps):
    """Return one looped Triton configuration as a layout candidate."""

    def launch(layout):
        return _launch_triton_looped(
            kernel,
            layout.values,
            layout.offsets,
            output,
            segment_count,
            block,
            warps,
        )

    return launch


def _run_distribution(
    torch,
    prepare,
    looped_kernel,
    name,
    segment_count,
    warmups,
    samples,
    *,
    device,
    synchronize,
):
    """Measure every candidate on one distribution's fresh layouts.

    Args:
        torch: The PyTorch module.
        prepare: The private planned-sum preparation function.
        looped_kernel: The looped Triton kernel, or None without Triton.
        name: Distribution name.
        segment_count: Segments per layout.
        warmups: Leading iterations that are checked but not recorded.
        samples: Timed iterations.
        device: Device that holds the inputs and outputs.
        synchronize: Callable that waits for all work on that device.

    Returns:
        The record row for this distribution.
    """
    pool = _layout_pool(name, segment_count, warmups + samples, _SEED)
    layouts = _upload(torch, pool, device)
    names = _candidate_names(triton_available=looped_kernel is not None)
    # A result that was never written must not pass the correctness check.
    outputs = {
        candidate: torch.full((segment_count,), float("nan"), device=device)
        for candidate in names
        if candidate != "torch"
    }
    prepare_us = []

    def swage_mixed(layout):
        start = time.perf_counter_ns()
        prepared = prepare(
            layout.values,
            layout.offsets,
            outputs["swage_mixed"],
            warp_max_elements=_WARP_MAX_ELEMENTS,
        )
        prepare_us.append((time.perf_counter_ns() - start) / 1_000.0)
        prepared.mixed()
        return outputs["swage_mixed"]

    def torch_reduce(layout):
        return torch.segment_reduce(
            layout.values, "sum", offsets=layout.offsets
        )

    candidates = {"swage_mixed": swage_mixed, "torch": torch_reduce}
    if looped_kernel is not None:
        for block, warps in _triton_looped_configs():
            candidate = f"triton_looped_b{block}_w{warps}"
            candidates[candidate] = _looped_candidate(
                looped_kernel, outputs[candidate], segment_count, block, warps
            )

    def check(candidate, result, layout):
        torch.testing.assert_close(
            result,
            layout.expected,
            rtol=0,
            atol=0,
            msg=lambda message: f"{candidate} on {name}: {message}",
        )
        if candidate in outputs:
            outputs[candidate].fill_(float("nan"))

    orders = [
        _candidate_order(names, _SEED, name, index)
        for index in range(len(pool))
    ]
    iterations = _measure(
        layouts,
        [order for _, order in orders],
        candidates,
        check,
        warmups=warmups,
        synchronize=synchronize,
    )
    raw = _timed_samples(iterations, names)
    return {
        "distribution": name,
        "segment_count": segment_count,
        "iterations": [
            {
                "layout_seed": layout.seed,
                "layout_statistics": summarize_lengths(layout.lengths),
                "order_seed": order_seed,
                **iteration,
            }
            for layout, (order_seed, _), iteration in zip(
                pool, orders, iterations, strict=True
            )
        ],
        "raw_samples_us": raw,
        "summary_us": {
            candidate: _median_iqr(timings)
            for candidate, timings in raw.items()
        },
        "swage_mixed_prepare_samples_us": prepare_us[warmups:],
        "correctness_passed": True,
    }


def _headline(summary):
    """Return the medians worth printing; the record keeps every sample."""
    medians = {
        candidate: round(timing["median"], 1)
        for candidate, timing in summary.items()
    }
    looped = {
        candidate: median
        for candidate, median in medians.items()
        if candidate.startswith("triton_looped")
    }
    headline = {name: medians[name] for name in ("swage_mixed", "torch")}
    if looped:
        fastest = min(looped, key=looped.get)
        headline[f"fastest {fastest}"] = looped[fastest]
    return headline


def main():
    """Run the fresh-offsets benchmark and write its JSON record."""
    arguments = _arguments()
    root = pathlib.Path(__file__).resolve().parents[1]
    _check_output(root, arguments.output, smoke=arguments.smoke)
    source = _git_metadata(root, allow_dirty=arguments.smoke)
    segment_count, warmups, samples = _sizes(arguments)

    import torch
    from swage import _runtime
    from swage._segmented_qualification import _prepare_planned_sum

    if not torch.cuda.is_available():
        raise RuntimeError(
            "fresh-offsets benchmark requires CUDA-enabled PyTorch"
        )
    torch.ones(1, device="cuda").sum().item()
    triton = _optional_triton()
    environment = _environment(
        torch,
        cuda_driver=_runtime.driver_version(),
        nvidia_driver=_nvidia_driver(),
        triton_version=triton.__version__ if triton else None,
    )
    looped_kernel = _make_triton_looped_sum() if triton else None
    if triton is None:
        print("Triton is not installed; skipping triton_looped", flush=True)

    results = []
    for name in _DISTRIBUTIONS:
        row = _run_distribution(
            torch,
            _prepare_planned_sum,
            looped_kernel,
            name,
            segment_count,
            warmups,
            samples,
            device="cuda",
            synchronize=torch.cuda.synchronize,
        )
        results.append(row)
        print(f"{name}: median_us={_headline(row['summary_us'])}", flush=True)

    record = _record(
        source=source,
        environment=environment,
        configuration=_configuration(
            segment_count=segment_count,
            warmups=warmups,
            samples=samples,
            triton_available=triton is not None,
        ),
        results=results,
        smoke=arguments.smoke,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {"output": str(arguments.output), "status": record["status"]},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
