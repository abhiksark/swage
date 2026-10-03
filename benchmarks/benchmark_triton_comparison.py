# benchmarks/benchmark_triton_comparison.py
"""Compare Swage GPU paths with Triton and PyTorch baselines.

This is a research benchmark harness, not a CI gate. Triton is imported only
when the benchmark is executed; the project does not depend on Triton.

Run with PYTHONPATH=python:build/python_packages and --output result.json.
Without other options the segmented suite runs the seven distributions,
32,768 segments, seed 7, and all-one values of the recorded campaign. The
device is the current CUDA device; select another with CUDA_VISIBLE_DEVICES.
"""

import argparse
import itertools
import json
import pathlib
import platform
import statistics
import subprocess
import sys
import time
import types
from collections.abc import Callable, Iterable
from datetime import datetime, timezone

import benchmark_provenance
from distributions import generate_lengths, summarize_lengths, worst_case_total

_WARMUPS = 25
_SAMPLES = 100
_BATCHED_LAUNCHES = 32
_MAX_BATCHED_LAUNCHES = 1 << 20
_TICK_FRACTION = 0.01
_SEGMENT_COUNT = 32_768
_SEED = 7
_WARP_MAX_ELEMENTS = 32
_LOOPED_BLOCKS = (128, 256, 512, 1024)
_PLANNED_CTA_BLOCK = 4096
_PLANNED_WARPS = (1, 2, 4, 8)
# A candidate filter names a candidate or its family: the name without the
# block and warp suffix of a sweep. The longest prefix is listed first.
_FAMILIES = (
    ("triton_planned_looped_b", "triton_planned_looped"),
    ("triton_planned_w", "triton_planned"),
    ("triton_looped_b", "triton_looped"),
    ("triton_rows_looped_r", "triton_rows_looped"),
    ("triton_b", "triton_fixed"),
)
# Rows per loop step and warps of the rank-two looped Triton sweep.
_ROWS_LOOPED_CONFIGS = ((4, 1), (4, 4), (16, 1), (16, 4), (64, 4))
# The column block of the rank-two looped kernel is a power of two between
# these bounds; the floor keeps a block of 1 or 2 lanes out of the kernel.
_ROWS_COLUMNS_FLOOR = 4
_ROWS_COLUMNS_CEILING = 64
_KIND_CODES = {"sum": 0, "max": 1, "min": 2, "mean": 3}
_I32_MAX = (1 << 31) - 1
# The seven distributions of the recorded campaign, in its run order.
_DISTRIBUTIONS = (
    "many-tiny",
    "uniform",
    "log-normal",
    "bimodal",
    "zipf-like",
    "few-huge",
    "one-outlier",
)
_OPTIONAL_DISTRIBUTIONS = ("alternating-empty", "power-law")
# The grid every value of a kind lies on, or None for values off any grid.
_QUANTUM = {"ones": 1.0, "quarters": 0.25, "normal": None}
_F32_UNIT_ROUNDOFF = 2.0**-24
_F32_EXACT_INTEGERS = 1 << 24
_UNIT_ROUNDOFF = {"float32": _F32_UNIT_ROUNDOFF, "float64": 2.0**-53}
_EXACT_INTEGERS = {"float32": _F32_EXACT_INTEGERS, "float64": 1 << 53}
# The value an output of a mean holds before a candidate writes it. A mean of
# an empty segment is NaN, so NaN cannot mark an unwritten mean.
_MEAN_MARKER = 1.0e30


def _padded_bytes_per_element(itemsize: int) -> int:
    """Return an upper bound of the bytes per padded element alive at once.

    While padding and reducing: an i32 index, a bool mask, and the
    gathered, the padded, and the masked values, each of ``itemsize``.
    """
    return 4 + 1 + 3 * itemsize


_PADDED_BYTES_PER_ELEMENT = _padded_bytes_per_element(4)


def _arguments(argv=None):
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
    parser.add_argument(
        "--distributions",
        nargs="+",
        choices=(*_DISTRIBUTIONS, *_OPTIONAL_DISTRIBUTIONS),
        default=list(_DISTRIBUTIONS),
        metavar="NAME",
        help=(
            "Segmented-sum distributions. The default is the seven of the "
            "recorded campaign; alternating-empty and power-law are run "
            "only when named."
        ),
    )
    parser.add_argument(
        "--segment-count",
        type=int,
        default=_SEGMENT_COUNT,
        help="Segments per segmented-sum row.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[_SEED],
        help="One segmented-sum row per distribution and seed.",
    )
    parser.add_argument(
        "--values",
        choices=tuple(_QUANTUM),
        default="ones",
        help=(
            "Timed segmented-sum values: all ones, seeded nonzero quarter "
            "multiples, or seeded standard normal values."
        ),
    )
    _add_candidate_filter(parser)
    arguments = parser.parse_args(argv)
    if arguments.samples <= 0 or arguments.warmups < 0:
        parser.error("samples must be positive and warmups nonnegative")
    filtered = arguments.candidates is not None or arguments.exclude_candidates
    if filtered and arguments.suite != "segmented-sum":
        parser.error(
            "--candidates and --exclude-candidates filter the segmented-sum "
            "suite; pass --suite segmented-sum"
        )
    try:
        _check_selectors(
            [*(arguments.candidates or ()), *arguments.exclude_candidates],
            _segmented_candidates(),
        )
    except ValueError as error:
        parser.error(str(error))
    if arguments.segment_count <= 0:
        parser.error("segment-count must be positive")
    for name in arguments.distributions:
        total = worst_case_total(name, arguments.segment_count)
        if total > _I32_MAX:
            parser.error(
                f"{name} with {arguments.segment_count} segments can reach "
                f"{total} elements, which does not fit i32 offsets"
            )
    return arguments


def _add_candidate_filter(parser):
    """Add the two candidate filter options to a harness parser."""
    parser.add_argument(
        "--candidates",
        nargs="+",
        metavar="NAME",
        help=(
            "Time only these candidates. A name is a candidate, such as "
            "triton_looped_b256_w4, or a family, such as triton_looped. The "
            "default is every candidate."
        ),
    )
    parser.add_argument(
        "--exclude-candidates",
        nargs="+",
        default=[],
        metavar="NAME",
        help="Leave these candidates or families out.",
    )


def _family(name: str) -> str:
    """Return the family of a candidate: its name without a sweep suffix."""
    for prefix, family in _FAMILIES:
        if name.startswith(prefix):
            return family
    return name


def _select(names, only, exclude) -> list[str]:
    """Return the candidates a filter keeps, in their given order.

    Args:
        names: Candidate names.
        only: Selectors to keep, or None to keep every candidate.
        exclude: Selectors to leave out; they win over ``only``.

    Returns:
        The kept names. A selector matches a candidate by its name or by
        its family.
    """

    def named(selectors, name):
        return name in selectors or _family(name) in selectors

    return [
        name
        for name in names
        if (only is None or named(only, name)) and not named(exclude, name)
    ]


def _wanted_skips(unable, wanted):
    """Return the reasons of the families a run asked for and cannot time.

    Args:
        unable: Reason by family for every family the row cannot run.
        wanted: The candidates the filter keeps, of every family.

    Returns:
        The entries of ``unable`` whose family the filter keeps. A family
        that the filter leaves out is excluded by option, whether or not
        the row could have run it.
    """
    families = {_family(name) for name in wanted}
    return {
        family: reason
        for family, reason in unable.items()
        if family in families
    }


def _check_selectors(selectors, names):
    """Require every selector to name a candidate or a family.

    Raises:
        ValueError: If a selector matches nothing. A misspelled selector
            would otherwise run, or leave out, something else than asked.
    """
    known = {*names, *map(_family, names)}
    unknown = [selector for selector in selectors if selector not in known]
    if unknown:
        families = sorted({_family(name) for name in names})
        raise ValueError(
            f"unknown candidates {', '.join(unknown)}; the families are "
            f"{', '.join(families)}, and a single configuration is named "
            "like triton_looped_b256_w4"
        )


def _segmented_candidates() -> list[str]:
    """Return every segmented-sum candidate name, in run order.

    A row runs the ones it can: the fixed sweep keeps the blocks that cover
    its longest segment, and a baseline that cannot produce a correct sum
    on the row is skipped.
    """
    looped = _triton_looped_configs()
    return [
        "swage_warp",
        "swage_cta",
        "swage_mixed",
        "torch",
        *(f"triton_b{b}_w{w}" for b, w in _triton_sum_configs(0)),
        *(f"triton_planned_w{w}" for w in _PLANNED_WARPS),
        *(f"triton_looped_b{b}_w{w}" for b, w in looped),
        *(f"triton_planned_looped_b{b}_w{w}" for b, w in looped),
        "torch_padded",
    ]


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


def _useful_bytes(value_count: int, segment_count: int) -> int:
    """Return the bytes any correct segmented sum has to move.

    These are the f32 values and the i32 offsets it reads and the f32 sums
    it writes. A baseline that moves more, such as a padded matrix, is
    still rated by these bytes, which is what makes the rate effective.
    """
    return 4 * (value_count + (segment_count + 1) + segment_count)


def _gb_per_s(useful_bytes: int, microseconds: float) -> float | None:
    """Return useful bytes per second in GB/s, or None for a zero time."""
    if microseconds <= 0:
        return None
    return useful_bytes / (microseconds * 1_000.0)


def _resolution(tick_us, sample_us) -> dict[str, object]:
    """Return a timer tick and its fraction of one timed sample."""
    return {
        "timer_tick_us": tick_us,
        "tick_fraction_of_sample": (
            None if tick_us is None or sample_us <= 0 else tick_us / sample_us
        ),
    }


def _resolved_samples(elapsed_us: Callable[[int], float], samples: int,
                      tick_us) -> tuple[list[float], int]:
    """Take per-launch samples from batches that outgrow the timer tick.

    A sample of a few timer ticks cannot resolve a difference of a few
    percent. A sample starts as a batch of 32 launches. When one tick is
    not below one percent of the median sample, the batch doubles and the
    samples are taken again, so the samples that are kept satisfy the limit
    themselves.

    Args:
        elapsed_us: Callable timing that many back-to-back launches once.
        samples: Number of samples.
        tick_us: Measured timer tick, or None to keep the 32 launches.

    Returns:
        The time per launch of each sample, and the launches per sample.

    Raises:
        RuntimeError: If no batch up to the limit outgrows the tick.
    """
    launches = _BATCHED_LAUNCHES
    while True:
        timings = [elapsed_us(launches) / launches for _ in range(samples)]
        sample_us = statistics.median(timings) * launches
        if tick_us is None or tick_us < _TICK_FRACTION * sample_us:
            return timings, launches
        if launches >= _MAX_BATCHED_LAUNCHES:
            raise RuntimeError(
                f"a batch of {launches} launches does not bring the "
                f"{tick_us} us timer tick below one percent of a sample"
            )
        launches *= 2


def _event_tick_us(torch, device, pairs: int = 256):
    """Estimate the CUDA event timer tick on the device.

    Events are recorded around a tiny operation many times, and the tick is
    the step those readings favour; see ``benchmark_provenance.timer_tick``.
    """
    scratch = torch.zeros(1, device=device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    elapsed = []
    for _ in range(pairs):
        start.record()
        scratch.add_(1)
        end.record()
        end.synchronize()
        elapsed.append(start.elapsed_time(end) * 1_000.0)
    return benchmark_provenance.timer_tick(elapsed)


def _call_us(torch, launch: Callable[[], object], warmups: int,
             samples: int, tick_us=None) -> dict[str, object]:
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
    summary = _median_iqr(timings)
    return {
        "samples_us": timings,
        "summary_us": summary,
        "launches_per_sample": 1,
        **_resolution(tick_us, summary["median"]),
    }


def _batched_event_us(torch, launch: Callable[[], object], warmups: int,
                      samples: int, tick_us=None) -> dict[str, object]:
    """Measure CUDA-event time per launch in a back-to-back batch."""
    for _ in range(warmups):
        launch()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    def elapsed_us(launches):
        start.record()
        for _ in range(launches):
            launch()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) * 1_000.0

    timings, launches = _resolved_samples(elapsed_us, samples, tick_us)
    summary = _median_iqr(timings)
    return {
        "samples_us": timings,
        "summary_us": summary,
        "launches_per_sample": launches,
        **_resolution(tick_us, summary["median"] * launches),
    }


class _CaptureFailed(Exception):
    """A launch could not be captured into a CUDA graph."""


def _graph_us(torch, launch: Callable[[], object], warmups: int,
              samples: int, tick_us=None) -> dict[str, object]:
    """Measure one launch through replay of a captured graph of launches."""
    for _ in range(warmups):
        launch()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    graphs = {}

    def replay_us(launches):
        if launches not in graphs:
            graph = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(graph):
                    for _ in range(launches):
                        launch()
            except RuntimeError as error:
                torch.cuda.synchronize()
                raise _CaptureFailed(str(error)) from error
            for _ in range(warmups):
                graph.replay()
            torch.cuda.synchronize()
            graphs.clear()
            graphs[launches] = graph
        start.record()
        graphs[launches].replay()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) * 1_000.0

    try:
        timings, launches = _resolved_samples(replay_us, samples, tick_us)
    except _CaptureFailed as error:
        return {"available": False, "error": str(error)}
    summary = _median_iqr(timings)
    return {
        "available": True,
        "samples_us": timings,
        "summary_us": summary,
        "launches_per_sample": launches,
        **_resolution(tick_us, summary["median"] * launches),
    }


def _timings(torch, launch: Callable[[], object], warmups: int,
             samples: int, *, ticks=None,
             useful_bytes=None) -> dict[str, object]:
    """Collect host, batched-event, and graph-replay measurements.

    Args:
        torch: The PyTorch module.
        launch: The launch to time.
        warmups: Untimed launches before each method.
        samples: Timed samples per method.
        ticks: Measured ``clock`` and ``event`` timer ticks in microseconds.
            Without them the event methods batch 32 launches and no
            resolution is recorded.
        useful_bytes: Bytes one launch has to move. With them every method
            that produced samples also reports ``effective_gb_per_s``.

    Returns:
        The ``call``, ``batched_event``, and ``graph`` entries.
    """
    ticks = ticks or {}
    methods = {
        "call": _call_us(
            torch, launch, warmups, samples, tick_us=ticks.get("clock")
        ),
        "batched_event": _batched_event_us(
            torch, launch, warmups, samples, tick_us=ticks.get("event")
        ),
        "graph": _graph_us(
            torch, launch, warmups, samples, tick_us=ticks.get("event")
        ),
    }
    if useful_bytes is not None:
        for method in methods.values():
            if "summary_us" in method:
                method["effective_gb_per_s"] = _gb_per_s(
                    useful_bytes, method["summary_us"]["median"]
                )
    return methods


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


def _run_vadd(torch, measure, seed: int) -> list[dict[str, object]]:
    """Benchmark fixed vector add across problem sizes.

    Args:
        torch: The PyTorch module.
        measure: Callable taking a launch and the bytes it has to move and
            returning its timings.
        seed: Seed of the input values.

    Returns:
        One row per problem size.
    """
    swage_kernel = _make_swage_vadd()
    triton_kernel = _make_triton_vadd()
    results = []
    for exponent in (10, 12, 14, 16, 18, 20, 22):
        n = 1 << exponent
        swage_block = 256
        grid = ((n + swage_block - 1) // swage_block,)
        generator = torch.Generator().manual_seed(seed)
        x = torch.randn(n, generator=generator).cuda()
        y = torch.randn(n, generator=generator).cuda()
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
            "seed": seed,
            # Two f32 inputs read and one f32 output written.
            "useful_bytes": 12 * n,
            "swage_block": swage_block,
            "swage_grid": grid[0],
            "triton_sweep_blocks": [128, 256, 512, 1024],
        }
        row["timings"] = {
            name: measure(launch, row["useful_bytes"])
            for name, launch in launches.items()
        }
        results.append(row)
    return results


def _exact_values(torch, count: int, seed: int = _SEED):
    """Return seeded host values that expose any misplaced element read.

    Every value is one of 0.25, 0.5, ..., 1.75. None is zero, so reading one
    element too many or too few always changes a sum, and neighbouring
    elements differ, so a shifted window changes it too. Sums of these
    quarter multiples are exact in f32 in any order for segments shorter than
    two million elements, which allows an exact comparison.
    """
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(1, 8, (count,), generator=generator).float() / 4


def _values(torch, kind: str, count: int, seed: int):
    """Return seeded host f32 values of one kind.

    Args:
        torch: The PyTorch module.
        kind: ``ones``; ``quarters``, the values of ``_exact_values``; or
            ``normal``, standard normal values that no grid makes exact.
        count: Number of values.
        seed: Seed of the random kinds.

    Returns:
        The values on the host.

    Raises:
        ValueError: If the kind is unknown.
    """
    if kind == "ones":
        return torch.ones(count)
    if kind == "quarters":
        return _exact_values(torch, count, seed)
    if kind == "normal":
        return torch.randn(count, generator=torch.Generator().manual_seed(seed))
    raise ValueError(f"unknown values kind {kind!r}")


def _sum_tolerance(torch, lengths, magnitude, quantum, *,
                   unit=_F32_UNIT_ROUNDOFF,
                   exact_integers=_F32_EXACT_INTEGERS):
    """Return how far a correct segment sum may be from the exact sum.

    Args:
        torch: The PyTorch module.
        lengths: Segment lengths as float64; they broadcast against
            ``magnitude``, so a column of lengths serves rank-two values.
        magnitude: Float64 sum of the absolute values of each segment.
        quantum: Grid that every value is a multiple of, or None.
        unit: Unit roundoff of the summed type, ``2 ** -24`` for f32.
        exact_integers: Largest count of grid steps the summed type holds
            exactly, ``2 ** 24`` for f32.

    Returns:
        One float64 tolerance per segment. It is zero where every partial
        sum in any order is exactly representable: the values lie on the
        grid and their magnitudes add up to at most ``exact_integers`` grid
        steps. Elsewhere it is ``gamma(n - 1) * magnitude`` with
        ``gamma(k) = k * u / (1 - k * u)``, the bound that holds for n
        values added in any order. Past ``k * u = 1 / 2`` the bound says
        nothing and the tolerance is infinite: the segment is not checked
        beyond having been written.
    """
    steps = (lengths - 1).clamp(min=0) * unit
    tolerance = torch.where(
        steps < 0.5,
        steps / (1 - steps) * magnitude,
        torch.full_like(magnitude, float("inf")),
    )
    if quantum is not None:
        exact = magnitude <= exact_integers * quantum
        tolerance = torch.where(exact, torch.zeros_like(tolerance), tolerance)
    return tolerance


def _reduction_reference(torch, values, offsets, kind, *, quantum, dtype):
    """Return the float64 results of one reduction and their tolerances.

    Args:
        torch: The PyTorch module.
        values: Host float64 values, ``[N]`` or ``[N, D]``.
        offsets: Host offsets of the segments, over the rows.
        kind: ``sum``, ``max``, ``min``, or ``mean``.
        quantum: Grid that every value is a multiple of, or None.
        dtype: ``float32`` or ``float64``, the type the candidates compute
            in.

    Returns:
        The reference, ``[S]`` or ``[S, D]``, with
        ``torch.segment_reduce(axis=0)`` semantics, and one tolerance per
        result. A maximum and a minimum are exact. A sum has the bound of
        ``_sum_tolerance`` in the unit of ``dtype``; in float64 it is
        doubled, because the float64 reference rounds as well. A mean has
        that bound divided by the segment length, plus two units of its own
        magnitude for the division and its reference. An empty segment has
        tolerance zero; its result must equal the reference, infinite for a
        maximum or a minimum and NaN for a mean.
    """
    reference = torch.segment_reduce(values, kind, offsets=offsets, axis=0)
    if kind in ("max", "min"):
        return reference, torch.zeros_like(reference)
    magnitude = torch.segment_reduce(
        values.abs(), "sum", offsets=offsets, axis=0
    )
    lengths = (offsets[1:] - offsets[:-1]).double()
    lengths = lengths.reshape(-1, *[1] * (values.dim() - 1))
    unit = _UNIT_ROUNDOFF[dtype]
    tolerance = _sum_tolerance(
        torch,
        lengths,
        magnitude,
        quantum,
        unit=unit,
        exact_integers=_EXACT_INTEGERS[dtype],
    )
    if dtype == "float64":
        tolerance = 2 * tolerance
    if kind == "mean":
        tolerance = torch.where(
            lengths > 0,
            tolerance / lengths.clamp(min=1) + 2 * unit * reference.abs(),
            torch.zeros_like(tolerance),
        )
    return reference, tolerance


def _sum_reference(torch, host_values, host_offsets, quantum):
    """Return the float64 segment sums and the tolerance of each."""
    values = host_values.double()
    reference = torch.segment_reduce(values, "sum", offsets=host_offsets)
    magnitude = torch.segment_reduce(
        values.abs(), "sum", offsets=host_offsets
    )
    lengths = (host_offsets[1:] - host_offsets[:-1]).double()
    return reference, _sum_tolerance(torch, lengths, magnitude, quantum)


def _check_modes(tolerance) -> dict[str, int]:
    """Count the segments checked exactly, within a bound, and not at all."""
    unchecked = int(tolerance.isinf().sum())
    exact = int((tolerance == 0).sum())
    return {
        "exact_segments": exact,
        "bounded_segments": tolerance.numel() - exact - unchecked,
        "unchecked_segments": unchecked,
    }


def _check_sums(torch, label: str, result, reference, tolerance):
    """Require every segment sum to be within its tolerance.

    Raises:
        AssertionError: If a sum is outside its tolerance or was never
            written, which also fails a segment with an infinite tolerance.
    """
    error = (result.double() - reference).abs()
    wrong = ~(error <= tolerance)
    if wrong.any():
        index = int(wrong.nonzero()[0])
        raise AssertionError(
            f"{label}: {int(wrong.sum())} of {wrong.numel()} segment sums "
            f"are outside their tolerance; segment {index} is "
            f"{result[index].item()!r}, expected {reference[index].item()!r} "
            f"within {tolerance[index].item()!r}"
        )


def _check_reduction(torch, label: str, result, reference, tolerance,
                     kind: str):
    """Require every result of a reduction to be within its tolerance.

    Args:
        torch: The PyTorch module.
        label: Candidate and row, for the failure message.
        result: The candidate's result, ``[S]`` or ``[S, D]``.
        reference: The float64 reference from ``_reduction_reference``. The
            result is compared where the reference is, so a reference kept
            on the host costs a copy of the result outside the timer.
        tolerance: Its tolerances.
        kind: The reduction kind.

    Raises:
        AssertionError: If a result is outside its tolerance, differs from
            an infinite or NaN reference, or was never written: NaN marks
            an unwritten sum, maximum, and minimum, and ``_MEAN_MARKER`` an
            unwritten mean, also where the tolerance is infinite.
    """
    unwritten = (result == _MEAN_MARKER) if kind == "mean" else None
    result = result.to(device=reference.device, dtype=torch.float64)
    ok = ((result - reference).abs() <= tolerance) | (result == reference)
    ok |= result.isnan() & reference.isnan()
    if unwritten is not None:
        ok &= ~unwritten.to(reference.device)
    wrong = ~ok
    if not wrong.any():
        return
    index = int(wrong.flatten().nonzero()[0])
    if result.dim() == 2:
        features = result.shape[1]
        segment, column = divmod(index, features)
        where = f"segment {segment}, column {column}"
    else:
        where = f"segment {index}"
    flat = (result.flatten(), reference.flatten(), tolerance.flatten())
    raise AssertionError(
        f"{label}: {int(wrong.sum())} of {wrong.numel()} results are outside "
        f"their tolerance; {where} is {flat[0][index].item()!r}, expected "
        f"{flat[1][index].item()!r} within {flat[2][index].item()!r}"
    )


def _padded_inputs(torch, values, offsets):
    """Pad every segment with zeros to the longest one.

    Returns:
        The f32 matrix with one row per segment and its bool mask.
    """
    width = int((offsets[1:] - offsets[:-1]).max())
    index = offsets[:-1, None] + torch.arange(
        width, dtype=torch.int32, device=offsets.device
    )
    mask = index < offsets[1:, None]
    gathered = values[index.clamp_(max=max(values.numel() - 1, 0))]
    return torch.where(mask, gathered, torch.zeros_like(gathered)), mask


def _padded_sum(padded, mask):
    """Reduce a padded matrix under its mask, in pure PyTorch."""
    return (padded * mask).sum(dim=1)


def _free_device_bytes(torch) -> int:
    """Return the device memory a new tensor could use right now."""
    free, _ = torch.cuda.mem_get_info()
    return free + torch.cuda.memory_reserved() - torch.cuda.memory_allocated()


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


def _make_triton_looped_task_sum():
    """Define a looping Triton sum over a task list lazily.

    One program per task walks its segment in fixed blocks, as the looped
    kernel does, but reads its segment id from a task list. Together with
    the packed warp kernel it is the planned Triton scheduler whose long
    tasks loop instead of provisioning one block for the longest segment.
    """
    import triton
    import triton.language as tl

    @triton.jit
    def looped_task_kernel(values, offsets, output, task_ids,
                           BLOCK: tl.constexpr):
        segment_id = tl.load(task_ids + tl.program_id(0))
        begin = tl.load(offsets + segment_id)
        end = tl.load(offsets + segment_id + 1)
        total = tl.zeros((BLOCK,), dtype=tl.float32)
        for start in range(begin, end, BLOCK):
            index = start + tl.arange(0, BLOCK)
            total += tl.load(values + index, mask=index < end, other=0.0)
        tl.store(output + segment_id, tl.sum(total, axis=0))

    return looped_task_kernel


def _make_triton_rows_looped():
    """Define a looping Triton reduction of rank-two values lazily.

    One program reduces one block of columns of one segment: it walks the
    rows of the segment in fixed blocks and keeps one partial result per
    row slot and column, then combines the row slots. The kind is a
    compile-time constant, and the partial results have the type of the
    values. A masked load reads the identity of the kind. The maximum and
    the minimum do not propagate NaN; the timed values hold none.
    """
    import triton
    import triton.language as tl

    @triton.jit
    def rows_looped_kernel(values, offsets, output, features,
                           KIND: tl.constexpr, BLOCK_ROWS: tl.constexpr,
                           BLOCK_COLUMNS: tl.constexpr):
        segment = tl.program_id(0)
        begin = tl.load(offsets + segment).to(tl.int64)
        end = tl.load(offsets + segment + 1).to(tl.int64)
        columns = tl.program_id(1) * BLOCK_COLUMNS + tl.arange(
            0, BLOCK_COLUMNS
        )
        in_columns = columns < features
        rows = tl.arange(0, BLOCK_ROWS)
        element = values.dtype.element_ty
        if KIND == 1:
            total = tl.full(
                (BLOCK_ROWS, BLOCK_COLUMNS), float("-inf"), element
            )
        elif KIND == 2:
            total = tl.full(
                (BLOCK_ROWS, BLOCK_COLUMNS), float("inf"), element
            )
        else:
            total = tl.zeros((BLOCK_ROWS, BLOCK_COLUMNS), element)
        for start in range(begin, end, BLOCK_ROWS):
            row = start + rows
            mask = (row < end)[:, None] & in_columns[None, :]
            index = row[:, None] * features + columns[None, :]
            if KIND == 1:
                tile = tl.load(values + index, mask=mask, other=float("-inf"))
                total = tl.maximum(total, tile)
            elif KIND == 2:
                tile = tl.load(values + index, mask=mask, other=float("inf"))
                total = tl.minimum(total, tile)
            else:
                total += tl.load(values + index, mask=mask, other=0.0)
        if KIND == 1:
            result = tl.max(total, axis=0)
        elif KIND == 2:
            result = tl.min(total, axis=0)
        else:
            result = tl.sum(total, axis=0)
            if KIND == 3:
                result = result / (end - begin).to(element)
        tl.store(
            output + segment * features + columns, result, mask=in_columns
        )

    return rows_looped_kernel


def _rows_columns(features: int) -> tuple[int, int]:
    """Return the column block of the rank-two kernel and the block count."""
    block = _ROWS_COLUMNS_FLOOR
    while block < min(features, _ROWS_COLUMNS_CEILING):
        block *= 2
    return block, -(-features // block)


def _launch_triton_rows(kernel, values, offsets, output, segment_count,
                        features, kind, block_rows, warps):
    """Launch the looped rank-two reduction over every segment."""
    block_columns, column_blocks = _rows_columns(features)
    kernel[(segment_count, column_blocks)](
        values,
        offsets,
        output,
        features,
        KIND=_KIND_CODES[kind],
        BLOCK_ROWS=block_rows,
        BLOCK_COLUMNS=block_columns,
        num_warps=warps,
    )
    return output


def _make_triton_kernels():
    """Define every Triton segmented kernel of the harnesses lazily."""
    packed, cta = _make_triton_planned_sum()
    return types.SimpleNamespace(
        fixed=_make_triton_segmented_sum(),
        looped=_make_triton_looped_sum(),
        packed=packed,
        cta=cta,
        cta_looped=_make_triton_looped_task_sum(),
        rows=_make_triton_rows_looped(),
    )


def _partition_tasks(torch, offsets):
    """Split the segment ids into warp and CTA task lists.

    This is the planning step of the planned Triton baselines, written as a
    Triton user would: on the device that holds the offsets, with two
    ``nonzero`` calls that each wait for the device to learn their size.

    Returns:
        The int32 ids of the segments of at most 32 elements and of the
        longer ones, each in segment order.
    """
    lengths = offsets[1:] - offsets[:-1]
    short = lengths <= _WARP_MAX_ELEMENTS
    return (
        torch.nonzero(short).flatten().to(torch.int32),
        torch.nonzero(~short).flatten().to(torch.int32),
    )


def _launch_triton_planned(kernels, values, offsets, output, warp_ids,
                           cta_ids, *, warps: int, block: int | None = None):
    """Launch the planned Triton sum over two task lists.

    Args:
        kernels: The Triton kernels ``packed``, ``cta``, and ``cta_looped``.
        values: Device values.
        offsets: Device offsets.
        output: Device output, one sum per segment.
        warp_ids: Ids of the short segments, four packed per program.
        cta_ids: Ids of the longer segments, one program each.
        warps: Warps of a CTA program.
        block: Block of the looping task kernel. None launches the one-block
            task kernel, which reads at most 4096 elements of a segment.

    Returns:
        The output.
    """
    warp_count = warp_ids.numel()
    if warp_count:
        kernels.packed[((warp_count + 3) // 4,)](
            values,
            offsets,
            output,
            warp_ids,
            warp_count,
            TASKS=4,
            WARP=32,
            num_warps=4,
        )
    cta_count = cta_ids.numel()
    if cta_count and block is None:
        kernels.cta[(cta_count,)](
            values,
            offsets,
            output,
            cta_ids,
            cta_count,
            BLOCK=_PLANNED_CTA_BLOCK,
            num_warps=warps,
        )
    elif cta_count:
        kernels.cta_looped[(cta_count,)](
            values, offsets, output, cta_ids, BLOCK=block, num_warps=warps
        )
    return output


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


def _make_triton_looped_sum():
    """Define a looping one-program-per-segment Triton sum lazily.

    Each program walks its own segment in fixed blocks and accumulates, so
    the block does not have to cover the longest segment and the baseline
    needs no host classification.
    """
    import triton
    import triton.language as tl

    @triton.jit
    def looped_sum_kernel(values, offsets, output, BLOCK: tl.constexpr):
        sid = tl.program_id(0)
        begin = tl.load(offsets + sid)
        end = tl.load(offsets + sid + 1)
        total = tl.zeros((BLOCK,), dtype=tl.float32)
        for start in range(begin, end, BLOCK):
            index = start + tl.arange(0, BLOCK)
            total += tl.load(values + index, mask=index < end, other=0.0)
        tl.store(output + sid, tl.sum(total, axis=0))

    return looped_sum_kernel


def _triton_looped_configs() -> list[tuple[int, int]]:
    """Return the looped Triton sweep, independent of segment lengths."""
    return [
        (block, warps)
        for block in _LOOPED_BLOCKS
        for warps in (1, 2, 4, 8)
        if warps <= block // 32
    ]


def _launch_triton_looped(kernel, values, offsets, output, segment_count,
                          block: int, warps: int):
    """Launch the looped Triton sum with one program per segment."""
    kernel[(segment_count,)](
        values, offsets, output, BLOCK=block, num_warps=warps
    )
    return output


def _check_on_exact_values(torch, family, launch, configs, offsets,
                           segment_count):
    """Require exact sums of a swept baseline on position-dependent values.

    On all-one values any in-bounds window of the right length gives the
    right sum. This check runs every configuration on values that make a
    shifted, short, or long read visible. It uses its own values and output
    and leaves the timed inputs alone.

    Args:
        torch: The PyTorch module.
        family: Family name of the baseline, for the failure message.
        launch: Callable taking values, an output, a block, and warps.
        configs: The block and warp configurations to check.
        offsets: Device offsets of the distribution being measured.
        segment_count: Number of segments.

    Raises:
        AssertionError: If a configuration differs from the CPU PyTorch
            reference in any segment.
    """
    host_offsets = offsets.cpu()
    host_values = _exact_values(torch, int(host_offsets[-1]))
    expected = torch.segment_reduce(
        host_values, "sum", offsets=host_offsets
    ).to(offsets.device)
    values = host_values.to(offsets.device)
    output = torch.empty(segment_count, device=offsets.device)
    for block, warps in configs:
        output.fill_(float("nan"))
        launch(values, output, block, warps)
        torch.testing.assert_close(
            output,
            expected,
            rtol=0,
            atol=0,
            msg=lambda message, block=block, warps=warps: (
                f"{family}_b{block}_w{warps} on position-dependent "
                f"values: {message}"
            ),
        )


def _check_triton_looped(torch, kernel, configs, offsets, segment_count):
    """Check the looped Triton sweep on position-dependent values."""

    def launch(values, output, block, warps):
        _launch_triton_looped(
            kernel, values, offsets, output, segment_count, block, warps
        )

    _check_on_exact_values(
        torch, "triton_looped", launch, configs, offsets, segment_count
    )


def _segmented_row(
    torch,
    kernels,
    prepare,
    measure,
    name: str,
    *,
    segment_count: int,
    seed: int,
    values_kind: str,
    device,
    free_bytes: Callable[[], int],
    synchronize,
    only=None,
    exclude=(),
) -> dict[str, object]:
    """Check and time the selected candidates on one distribution and seed.

    Args:
        torch: The PyTorch module.
        kernels: The Triton kernels ``fixed``, ``looped``, ``packed``,
            ``cta``, and ``cta_looped``.
        prepare: The private planned-sum preparation function.
        measure: Callable taking a launch and the bytes it has to move and
            returning its timings.
        name: Distribution name.
        segment_count: Segments in the row.
        seed: Seed of the lengths and of the random value kinds.
        values_kind: Key of ``_QUANTUM``.
        device: Device that holds the inputs and outputs.
        free_bytes: Callable returning the bytes the padded baseline may
            use; it is called once the row's inputs are on the device.
        synchronize: Callable that waits for all work on that device.
        only: Candidate filter selectors to keep, or None for all.
        exclude: Candidate filter selectors to leave out.

    Returns:
        The record row. A baseline that cannot produce a correct sum on the
        row is left out of ``timings`` and listed in ``skipped`` with the
        reason; it is never timed on a wrong result. A candidate the filter
        left out is listed in ``excluded``; it is not prepared, launched,
        or checked. ``candidate_order`` is the order the candidates ran in.

    Raises:
        AssertionError: If a candidate that ran is wrong in any segment.
        ValueError: If the filter leaves no candidate the row can run.
    """
    lengths = generate_lengths(name, segment_count, seed)
    statistics_summary = summarize_lengths(lengths)
    max_length = statistics_summary["max"]
    host_offsets = torch.tensor(
        [0, *itertools.accumulate(lengths)], dtype=torch.int32
    )
    host_values = _values(
        torch, values_kind, statistics_summary["total"], seed
    )
    reference, tolerance = _sum_reference(
        torch, host_values, host_offsets, _QUANTUM[values_kind]
    )
    check = _check_modes(tolerance)
    values = host_values.to(device)
    offsets = host_offsets.to(device)
    reference = reference.to(device)
    tolerance = tolerance.to(device)
    useful_bytes = _useful_bytes(statistics_summary["total"], segment_count)

    unable = {}
    triton_configs = _triton_sum_configs(max_length)
    if not triton_configs:
        unable["triton_fixed"] = (
            "no swept block covers the longest segment of "
            f"{max_length} elements"
        )
    if max_length > _PLANNED_CTA_BLOCK:
        unable["triton_planned"] = (
            f"its CTA kernel reads one block of {_PLANNED_CTA_BLOCK} "
            f"elements and the longest segment has {max_length}"
        )
    padded_bytes = segment_count * max_length * _PADDED_BYTES_PER_ELEMENT
    memory_budget = free_bytes()
    if padded_bytes > memory_budget:
        unable["torch_padded"] = (
            f"padding {segment_count} segments to {max_length} elements "
            f"needs {padded_bytes} bytes and {memory_budget} are free"
        )
    triton_looped_configs = _triton_looped_configs()
    warp_ids, cta_ids = _partition_tasks(torch, offsets)
    outputs = {}
    allocated = {}

    def unwritten(candidate):
        # A result that was never written must not pass the check.
        outputs[candidate] = torch.full(
            (segment_count,), float("nan"), device=device
        )
        return outputs[candidate]

    def swage(policy):
        return getattr(
            prepare(
                values,
                offsets,
                unwritten(f"swage_{policy}"),
                warp_max_elements=_WARP_MAX_ELEMENTS,
            ),
            policy,
        )

    def torch_reduce():
        def launch():
            allocated["torch"] = torch.segment_reduce(
                values, "sum", offsets=offsets
            )
            return allocated["torch"]

        return launch

    def fixed(candidate, block, warps):
        output = unwritten(candidate)
        return lambda: kernels.fixed[(segment_count,)](
            values,
            offsets,
            output,
            segment_count,
            BLOCK=block,
            num_warps=warps,
        )

    def planned(candidate, warps, block=None):
        output = unwritten(candidate)
        return lambda: _launch_triton_planned(
            kernels,
            values,
            offsets,
            output,
            warp_ids,
            cta_ids,
            warps=warps,
            block=block,
        )

    def looped(candidate, block, warps):
        output = unwritten(candidate)
        return lambda: _launch_triton_looped(
            kernels.looped, values, offsets, output, segment_count, block, warps
        )

    def padded():
        # The padding is prepared outside the timed launch, as the Swage
        # plan and the Triton task lists are.
        matrix, mask = _padded_inputs(torch, values, offsets)

        def launch():
            allocated["torch_padded"] = _padded_sum(matrix, mask)
            return allocated["torch_padded"]

        return launch

    # Every candidate the row can run, in run order, with the setup that
    # makes its launch. Only a selected candidate is set up.
    setups = {
        "swage_warp": lambda: swage("warp"),
        "swage_cta": lambda: swage("cta"),
        "swage_mixed": lambda: swage("mixed"),
        "torch": torch_reduce,
    }
    for block, warps in triton_configs:
        candidate = f"triton_b{block}_w{warps}"
        setups[candidate] = (
            lambda candidate=candidate, block=block, warps=warps:
            fixed(candidate, block, warps)
        )
    for warps in _PLANNED_WARPS:
        candidate = f"triton_planned_w{warps}"
        setups[candidate] = (
            lambda candidate=candidate, warps=warps: planned(candidate, warps)
        )
    for block, warps in triton_looped_configs:
        candidate = f"triton_looped_b{block}_w{warps}"
        setups[candidate] = (
            lambda candidate=candidate, block=block, warps=warps:
            looped(candidate, block, warps)
        )
    for block, warps in triton_looped_configs:
        candidate = f"triton_planned_looped_b{block}_w{warps}"
        setups[candidate] = (
            lambda candidate=candidate, block=block, warps=warps:
            planned(candidate, warps, block)
        )
    setups["torch_padded"] = padded
    # Every candidate of the row is timed, excluded by the filter, or
    # skipped because the filter keeps it and the row cannot run it.
    skipped = _wanted_skips(
        unable, _select(_segmented_candidates(), only, exclude)
    )
    wanted = _select(setups, only, exclude)
    selected = [
        candidate for candidate in wanted if _family(candidate) not in unable
    ]
    if not selected:
        raise ValueError(
            f"the candidate filter leaves no candidate that can run on "
            f"{name} with {segment_count} segments"
        )
    launches = {candidate: setups[candidate]() for candidate in selected}
    for launch in launches.values():
        launch()
    synchronize()
    for output_name, output in {**outputs, **allocated}.items():
        _check_sums(torch, output_name, output, reference, tolerance)
    for family, launch_family in (
        (
            "triton_looped",
            lambda values, output, block, warps: _launch_triton_looped(
                kernels.looped,
                values,
                offsets,
                output,
                segment_count,
                block,
                warps,
            ),
        ),
        (
            "triton_planned_looped",
            lambda values, output, block, warps: _launch_triton_planned(
                kernels,
                values,
                offsets,
                output,
                warp_ids,
                cta_ids,
                warps=warps,
                block=block,
            ),
        ),
    ):
        _check_on_exact_values(
            torch,
            family,
            launch_family,
            [
                (block, warps)
                for block, warps in triton_looped_configs
                if f"{family}_b{block}_w{warps}" in launches
            ],
            offsets,
            segment_count,
        )
    return {
        "case": "segmented-sum",
        "distribution": name,
        "seed": seed,
        "values": values_kind,
        "segment_count": segment_count,
        "statistics": statistics_summary,
        "useful_bytes": useful_bytes,
        "check": check,
        "skipped": skipped,
        "excluded": [
            candidate for candidate in setups if candidate not in wanted
        ],
        "candidate_order": list(launches),
        "triton_sweep_configs": [
            {"block": block, "num_warps": warps}
            for block, warps in triton_configs
        ],
        "triton_looped_sweep_configs": [
            {"block": block, "num_warps": warps}
            for block, warps in triton_looped_configs
        ],
        "triton_planned": {
            "warp_threshold": _WARP_MAX_ELEMENTS,
            "warp_tasks": warp_ids.numel(),
            "cta_tasks": cta_ids.numel(),
            "warp_tasks_per_program": 4,
            "cta_block": _PLANNED_CTA_BLOCK,
            "cta_num_warps_sweep": list(_PLANNED_WARPS),
            "looped_cta_sweep_configs": [
                {"block": block, "num_warps": warps}
                for block, warps in triton_looped_configs
            ],
        },
        "timings": {
            launch_name: measure(launch, useful_bytes)
            for launch_name, launch in launches.items()
        },
    }


def _run_segmented_sum(torch, measure, arguments) -> list[dict[str, object]]:
    """Benchmark private segmented sum against Triton and torch baselines."""
    from swage._segmented_qualification import _prepare_planned_sum

    kernels = _make_triton_kernels()
    return [
        _segmented_row(
            torch,
            kernels,
            _prepare_planned_sum,
            measure,
            name,
            segment_count=arguments.segment_count,
            seed=seed,
            values_kind=arguments.values,
            device="cuda",
            free_bytes=lambda: _free_device_bytes(torch),
            synchronize=torch.cuda.synchronize,
            only=arguments.candidates,
            exclude=arguments.exclude_candidates,
        )
        for name in arguments.distributions
        for seed in arguments.seeds
    ]


def main():
    """Run the selected comparison benchmark and write JSON evidence."""
    arguments = _arguments()

    import torch
    import triton

    if not torch.cuda.is_available():
        raise RuntimeError("benchmark requires CUDA-enabled PyTorch")
    root = pathlib.Path(__file__).resolve().parents[1]
    device = torch.cuda.current_device()
    torch.ones(1, device="cuda").sum().item()
    provenance = benchmark_provenance.start(
        torch, benchmark_provenance.swage_build()
    )
    properties = torch.cuda.get_device_properties(device)
    capability = torch.cuda.get_device_capability(device)
    ticks = {
        "clock": benchmark_provenance.clock_tick_us(),
        "event": _event_tick_us(torch, "cuda"),
    }

    def measure(launch, useful_bytes):
        return _timings(
            torch,
            launch,
            arguments.warmups,
            arguments.samples,
            ticks=ticks,
            useful_bytes=useful_bytes,
        )

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
            "cuda_driver": provenance["cuda_driver"],
            "gpu": torch.cuda.get_device_name(device),
            "compute_capability": f"sm_{capability[0]}{capability[1]}",
            "multiprocessors": properties.multi_processor_count,
            "total_memory_bytes": properties.total_memory,
        },
        "provenance": provenance,
        "methodology": {
            "suite": arguments.suite,
            "warmups": arguments.warmups,
            "samples": arguments.samples,
            "distributions": arguments.distributions,
            "segment_count": arguments.segment_count,
            "seeds": arguments.seeds,
            "values": arguments.values,
            "batched_launches": _BATCHED_LAUNCHES,
            "graph_replay_launches": _BATCHED_LAUNCHES,
            "batching": (
                "event-timed samples batch at least batched_launches "
                "launches; while one event timer tick is not below one "
                "percent of the median sample, the batch doubles and the "
                "samples are taken again; every timing entry records its "
                "launches_per_sample, the tick, and the tick as a fraction "
                "of the sample"
            ),
            "timer_ticks_us": ticks,
            "timer_ticks": (
                "measured in this process: the smallest advance of "
                "back-to-back time.perf_counter_ns reads, and the step that "
                "CUDA event readings around a tiny operation favour, which "
                "can be coarser than the finest gap between two readings"
            ),
            "effective_gb_per_s": (
                "useful_bytes of the row divided by the median time; for "
                "segmented sum the f32 values and i32 offsets read and the "
                "f32 sums written, for vector add two f32 inputs read and "
                "one f32 output written"
            ),
            "candidate_order": (
                "each candidate is timed to completion in the order of the "
                "row's candidate_order; candidates are not interleaved"
            ),
            "candidate_filter": {
                "candidates": arguments.candidates,
                "exclude_candidates": arguments.exclude_candidates,
                "note": (
                    "a name selects one candidate or a family; a row lists "
                    "what the filter left out under excluded, apart from "
                    "what it could not run under skipped"
                ),
            },
            "compilation_excluded": True,
            "correctness_checked_before_timing": True,
            "correctness": (
                "every timed candidate is checked on the timed values "
                "against a float64 CPU reference: exactly where sums of the "
                "values are exact in f32 in any order, otherwise within "
                "gamma(n - 1) times the sum of magnitudes; each row counts "
                "its exact, bounded, and unchecked segments"
            ),
            "skipped": (
                "a baseline that cannot produce a correct sum on a row is "
                "listed in the row's skipped mapping with the reason and "
                "is not timed"
            ),
            "triton_dependency": "optional runtime import; not a project dep",
            "triton_fixed": (
                "one program per segment, one masked block; blocks smaller "
                "than the longest segment are excluded"
            ),
            "triton_looped": (
                "one program per segment, a loop over the segment in fixed "
                "blocks; no block is excluded for the longest segment"
            ),
            "triton_planned": (
                "the task lists come from a device nonzero over the segment "
                "lengths, outside the timed launch; short tasks are packed "
                "four per program and each longer task reads one block of "
                "4096 elements, so a longer segment skips the baseline"
            ),
            "triton_planned_looped": (
                "the same task lists and packed short tasks; each longer "
                "task loops over its segment in fixed blocks, with the "
                "block and warp sweep of triton_looped, so no segment "
                "length skips it"
            ),
            "triton_looped_check": (
                "besides the check shared by every candidate, each "
                "triton_looped and triton_planned_looped configuration is "
                "checked exactly before timing on seeded nonzero quarter "
                "multiples against CPU torch.segment_reduce"
            ),
            "torch_padded": (
                "pure PyTorch: every segment padded with zeros to the "
                "longest one outside the timed launch, then the masked "
                "matrix summed per row inside it"
            ),
        },
        "results": [],
    }
    if arguments.suite in {"all", "vadd"}:
        result["results"].extend(_run_vadd(torch, measure, arguments.seeds[0]))
    if arguments.suite in {"all", "segmented-sum"}:
        result["results"].extend(
            _run_segmented_sum(torch, measure, arguments)
        )
    benchmark_provenance.finish(provenance)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(arguments.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
