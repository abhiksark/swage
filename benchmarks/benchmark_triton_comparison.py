# benchmarks/benchmark_triton_comparison.py
"""Compare Swage GPU paths with Triton and PyTorch baselines.

This is a research benchmark harness, not a CI gate. Triton is imported only
when the benchmark is executed; the project does not depend on Triton.

Run with PYTHONPATH=python:build/python_packages and --output result.json.
The segmented-sum suite measures compilation first, so SWAGE_CACHE_DIR and
TRITON_CACHE_DIR must name two distinct empty directories; the campaign
driver run_triton_comparison_campaign.py creates them for every process.
Without other options the segmented suite runs the seven synthetic
distributions and the soc-epinions1-outdegree-v1 trace with 32,768
segments, seed 7, and all-one values. The device is the current CUDA
device; select another with CUDA_VISIBLE_DEVICES.
"""

import argparse
import functools
import itertools
import json
import os
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
from benchmark_campaign import SCHEMA_VERSION, summarize_us, validate_child
from distributions import generate_lengths, summarize_lengths, worst_case_total

_WARMUPS = 25
_SAMPLES = 100
# An event or graph sample starts as a batch of this many launches. While
# one timer tick is not below _TICK_FRACTION of a candidate's median sample,
# that candidate's batch doubles, up to _MAX_BATCHED_LAUNCHES.
_BATCHED_LAUNCHES = 32
_MAX_BATCHED_LAUNCHES = 1 << 20
_TICK_FRACTION = 0.01
_SEGMENT_COUNT = 32_768
_SEED = 7
_WARP_MAX_ELEMENTS = 32
_VADD_EXPONENTS = (10, 12, 14, 16, 18, 20, 22)
_VADD_SWAGE_BLOCK = 256
_VADD_TRITON_BLOCKS = (128, 256, 512, 1024)
_FIXED_BLOCKS = (32, 64, 128, 256, 512, 1024, 2048, 4096)
_LOOPED_BLOCKS = (128, 256, 512, 1024)
_PLANNED_CTA_BLOCK = 4096
_PLANNED_WARPS = (1, 2, 4, 8)
_PACKED_TASKS = 4
_FUSED_LANES = 128
_FUSED_MAX_ELEMENTS = 4096
_FUSED_NUM_WARPS = 4
_MATCHED = "triton_matched_task_partition"
# A candidate filter names a candidate or its family: the name without the
# block and warp suffix of a sweep. The longest prefix is listed first.
# triton_planned_w names the planned family of benchmark_fresh_offsets.py,
# which imports this mapping.
_FAMILIES = (
    ("triton_planned_looped_b", "triton_planned_looped"),
    (_MATCHED, _MATCHED),
    ("triton_planned_w", "triton_planned"),
    ("triton_looped_b", "triton_looped"),
    ("triton_b", "triton_fixed"),
)
_I32_MAX = (1 << 31) - 1
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
_REAL_TRACE_SEGMENTS = 32_768
_OPTIONAL_DISTRIBUTIONS = ("alternating-empty", "power-law")
# The grid every value of a kind lies on, or None for values off any grid.
_QUANTUM = {"ones": 1.0, "quarters": 0.25, "normal": None}
_F32_UNIT_ROUNDOFF = 2.0**-24
_F32_EXACT_INTEGERS = 1 << 24
# An upper bound of the bytes per padded element alive at once while padding:
# an i32 index, a bool mask, the gathered f32, the zeros f32, and the padded
# f32.
_PADDED_BYTES_PER_ELEMENT = 17
# The two candidates whose complete orchestration is timed.
_ORCHESTRATION = ("swage_mixed", _MATCHED)


def _arguments(argv=None):
    """Parse benchmark controls.

    Args:
        argv: Command-line arguments, or None for ``sys.argv``.

    Returns:
        The parsed arguments.
    """
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
        choices=(
            *_SYNTHETIC_DISTRIBUTIONS,
            _REAL_TRACE_NAME,
            *_OPTIONAL_DISTRIBUTIONS,
        ),
        default=[*_SYNTHETIC_DISTRIBUTIONS, _REAL_TRACE_NAME],
        metavar="NAME",
        help=(
            "Segmented-sum distributions, in run order. The default is the "
            "seven synthetic distributions and the "
            f"{_REAL_TRACE_NAME} trace; alternating-empty and power-law are "
            "run only when named."
        ),
    )
    parser.add_argument(
        "--segment-count",
        type=int,
        default=_SEGMENT_COUNT,
        help=(
            "Segments per synthetic segmented-sum row. The trace has "
            f"{_REAL_TRACE_SEGMENTS} segments."
        ),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[_SEED],
        help=(
            "One segmented-sum row per distribution and seed. The seed "
            "draws the lengths of a synthetic distribution and the random "
            "values of every row; the first seed also draws the vector-add "
            "inputs."
        ),
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
    for option, chosen in (
        ("--distributions", arguments.distributions),
        ("--seeds", arguments.seeds),
    ):
        if len(chosen) != len(set(chosen)):
            parser.error(f"{option} names a value twice")
    if arguments.segment_count <= 0:
        parser.error("segment-count must be positive")
    if (
        _REAL_TRACE_NAME in arguments.distributions
        and arguments.segment_count != _REAL_TRACE_SEGMENTS
    ):
        parser.error(
            f"{_REAL_TRACE_NAME} has {_REAL_TRACE_SEGMENTS} segments; name "
            "the distributions without it to change --segment-count"
        )
    for name in arguments.distributions:
        if name == _REAL_TRACE_NAME:
            continue
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


def _matched_name(warps: int) -> str:
    """Return the matched task-partition candidate of one CTA warp count."""
    return _MATCHED if warps == 1 else f"{_MATCHED}_w{warps}"


def _row_candidates(max_length: int) -> list[str]:
    """Return the candidates of a row whose longest segment is given.

    The fixed sweep keeps only the blocks that cover the longest segment;
    every other family is listed whether or not the row can run it.
    """
    looped = _triton_looped_configs()
    return [
        "swage_warp",
        "swage_cta",
        "swage_mixed",
        "torch_segment_reduce",
        "torch_padded",
        "triton_fused",
        *(f"triton_b{b}_w{w}" for b, w in _triton_sum_configs(max_length)),
        *(_matched_name(warps) for warps in _PLANNED_WARPS),
        *(f"triton_looped_b{b}_w{w}" for b, w in looped),
        *(f"triton_planned_looped_b{b}_w{w}" for b, w in looped),
    ]


def _segmented_candidates() -> list[str]:
    """Return every segmented-sum candidate name, in run order.

    A row runs the ones it can: the fixed sweep keeps the blocks that cover
    its longest segment, and a baseline that cannot produce a correct sum
    on the row is skipped.
    """
    return _row_candidates(0)


def _length_limits(max_length: int) -> dict[str, str]:
    """Return why each family cannot run on a row, by its longest segment."""
    unable = {}
    if not _triton_sum_configs(max_length):
        unable["triton_fixed"] = (
            "no swept block covers the longest segment of "
            f"{max_length} elements"
        )
    if max_length > _PLANNED_CTA_BLOCK:
        unable[_MATCHED] = (
            f"its CTA kernel reads one block of {_PLANNED_CTA_BLOCK} "
            f"elements and the longest segment has {max_length}"
        )
    if max_length > _FUSED_MAX_ELEMENTS:
        unable["triton_fused"] = (
            f"its CTA programs read at most {_FUSED_MAX_ELEMENTS} elements "
            f"and the longest segment has {max_length}"
        )
    return unable


def _case_candidates(max_length: int, only, exclude) -> list[str]:
    """Return the selected candidates a row of this length can run.

    Only the memory check of the padded baseline is left out; it needs the
    free device memory of the row.
    """
    unable = _length_limits(max_length)
    return [
        name
        for name in _select(_row_candidates(max_length), only, exclude)
        if _family(name) not in unable
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


def _resolves(tick_us, sample_us) -> bool:
    """Return whether a timer tick is below one percent of a sample."""
    return tick_us is None or tick_us < _TICK_FRACTION * sample_us


def _resolved_samples(
    elapsed_us: Callable[[int], float], samples: int, tick_us
) -> tuple[list[float], int]:
    """Take per-launch samples from batches that outgrow the timer tick.

    A sample of a few timer ticks cannot resolve a difference of a few
    percent. A sample starts as a batch of 32 launches. When one tick is
    not below one percent of the median sample, the batch doubles and the
    samples are taken again, so the samples that are kept satisfy the limit
    themselves. benchmark_composable_reductions.py times with this; the
    comparison applies the same rule to interleaved candidates in
    ``_interleaved_batches``.

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
        if _resolves(tick_us, statistics.median(timings) * launches):
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


def _call_us(
    torch,
    launch: Callable[[], object],
    warmups: int,
    samples: int,
    tick_us=None,
) -> dict[str, object]:
    """Measure the synchronized Python-call latency of one candidate.

    benchmark_composable_reductions.py times with this; the comparison
    interleaves its candidates with ``_interleaved_call_us``.
    """
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


class _CaptureFailed(Exception):
    """A launch could not be captured into a CUDA graph."""


def _graph_us(
    torch,
    launch: Callable[[], object],
    warmups: int,
    samples: int,
    tick_us=None,
) -> dict[str, object]:
    """Measure one candidate through replay of a captured graph of launches.

    benchmark_composable_reductions.py times with this; the comparison
    interleaves its candidates with ``_interleaved_graph_us``.
    """
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
    offsets = (round_index % len(names) for round_index in range(rounds))
    return [names[offset:] + names[:offset] for offset in offsets]


def _order_position_counts(
    orders: Iterable[tuple[str, ...]],
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


def _timing_entry(samples: list[float], launches: int, tick_us):
    """Return one timing entry: raw samples, summary, batch, and tick."""
    summary = summarize_us(samples)
    return {
        "samples_us": samples,
        "summary_us": summary,
        "launches_per_sample": launches,
        **_resolution(tick_us, summary["median"] * launches),
    }


def _interleaved_call_us(
    torch,
    launches: dict[str, Callable[[], object]],
    warmups: int,
    samples: int,
    tick_us=None,
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
        name: _timing_entry(values, 1, tick_us)
        for name, values in timings.items()
    }


def _interleaved_batches(
    names: list[str],
    samples: int,
    tick_us,
    elapsed_us: Callable[[str, int], float | None],
    begin_pass: Callable[[dict[str, int]], None] | None = None,
) -> tuple[dict[str, list[float]], dict[str, int]]:
    """Take interleaved batch samples until every batch outgrows the tick.

    Every candidate starts with a batch of 32 launches, and the candidates
    are sampled in rotating order. When one tick is not below one percent
    of a candidate's median sample, that candidate's batch doubles and
    every candidate is sampled again in new rotating rounds, so the kept
    samples of every candidate are interleaved with each other and satisfy
    the limit themselves.

    Args:
        names: Candidate names in base order.
        samples: Timed rounds.
        tick_us: Measured timer tick, or None to keep 32 launches.
        elapsed_us: Callable timing one batch of a candidate; it returns
            None for a candidate that has nothing to time, which is left
            out of the result.
        begin_pass: Callable given the batches of a pass before its first
            round, or None.

    Returns:
        The per-launch samples and the launches per sample of every timed
        candidate.

    Raises:
        RuntimeError: If a batch up to the limit does not outgrow the tick.
    """
    batches = dict.fromkeys(names, _BATCHED_LAUNCHES)
    while True:
        if begin_pass is not None:
            begin_pass(batches)
        timings = {}
        for order in _rotating_orders(names, samples):
            for name in order:
                elapsed = elapsed_us(name, batches[name])
                if elapsed is not None:
                    timings.setdefault(name, []).append(elapsed / batches[name])
        unresolved = [
            name
            for name, values in timings.items()
            if not _resolves(tick_us, statistics.median(values) * batches[name])
        ]
        if not unresolved:
            return timings, batches
        for name in unresolved:
            if batches[name] >= _MAX_BATCHED_LAUNCHES:
                raise RuntimeError(
                    f"a batch of {batches[name]} launches of {name} does not "
                    f"bring the {tick_us} us timer tick below one percent "
                    "of a sample"
                )
            batches[name] *= 2


def _interleaved_batched_event_us(
    torch,
    launches: dict[str, Callable[[], object]],
    warmups: int,
    samples: int,
    tick_us=None,
) -> dict[str, dict[str, object]]:
    """Measure back-to-back launch batches in rotating candidate order."""
    _warm_interleaved(torch, launches, warmups)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    def elapsed_us(name, count):
        start.record()
        for _ in range(count):
            launches[name]()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) * 1_000.0

    timings, batches = _interleaved_batches(
        list(launches), samples, tick_us, elapsed_us
    )
    return {
        name: _timing_entry(timings[name], batches[name], tick_us)
        for name in launches
    }


def _capture_graphs(
    torch,
    launches: dict[str, Callable[[], object]],
    batches: dict[str, int],
) -> tuple[dict[str, object], dict[str, str]]:
    """Capture every candidate's graph of its batch before any replay."""
    graphs = {}
    errors = {}
    for name, launch in launches.items():
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                for _ in range(batches[name]):
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
    tick_us=None,
) -> dict[str, dict[str, object]]:
    """Measure prepared graph replays in rotating candidate order.

    Every pass captures the graph of every candidate at its batch of the
    pass before any graph of the pass is replayed. A candidate that cannot
    be captured is reported unavailable and is not captured again.
    """
    _warm_interleaved(torch, launches, warmups)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    errors = {}
    graphs = {}

    def begin_pass(batches):
        graphs.clear()
        captured, failed = _capture_graphs(
            torch,
            {
                name: launch
                for name, launch in launches.items()
                if name not in errors
            },
            batches,
        )
        errors.update(failed)
        graphs.update(captured)
        for order in _rotating_orders(launches, warmups):
            for name in order:
                if name in graphs:
                    graphs[name].replay()
        torch.cuda.synchronize()

    def elapsed_us(name, count):
        if name not in graphs:
            return None
        start.record()
        graphs[name].replay()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) * 1_000.0

    timings, batches = _interleaved_batches(
        list(launches), samples, tick_us, elapsed_us, begin_pass
    )
    results = {}
    for name in launches:
        if name in errors:
            results[name] = {"available": False, "error": errors[name]}
        else:
            results[name] = {
                "available": True,
                **_timing_entry(timings[name], batches[name], tick_us),
            }
    return results


def _timings(
    torch,
    launches: dict[str, Callable[[], object]],
    warmups: int,
    samples: int,
    *,
    ticks=None,
    useful_bytes=None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Collect interleaved call, event, and graph measurements.

    Args:
        torch: The PyTorch module.
        launches: The launch of every candidate, in base order.
        warmups: Untimed rounds before each method.
        samples: Timed rounds per method.
        ticks: Measured ``clock`` and ``event`` timer ticks in microseconds.
            Without them the event methods batch 32 launches and no
            resolution is recorded.
        useful_bytes: Bytes one launch has to move. With them every method
            that produced samples also reports ``effective_gb_per_s``.

    Returns:
        The ``call``, ``batched_event``, and ``graph`` entries of every
        candidate, and the timing method.
    """
    ticks = ticks or {}
    call = _interleaved_call_us(
        torch, launches, warmups, samples, ticks.get("clock")
    )
    batched = _interleaved_batched_event_us(
        torch, launches, warmups, samples, ticks.get("event")
    )
    graph = _interleaved_graph_us(
        torch, launches, warmups, samples, ticks.get("event")
    )
    results = {
        name: {
            "call": call[name],
            "batched_event": batched[name],
            "graph": graph[name],
        }
        for name in launches
    }
    if useful_bytes is not None:
        for methods in results.values():
            for method in methods.values():
                if "summary_us" in method:
                    method["effective_gb_per_s"] = _gb_per_s(
                        useful_bytes, method["summary_us"]["median"]
                    )
    orders = _rotating_orders(launches, samples)
    method = {
        "sampling": "deterministic_rotating_interleaved",
        "base_candidate_order": list(launches),
        "round_rotation": "left by round_index modulo candidate_count",
        "timed_rounds": samples,
        "order_position_counts": _order_position_counts(orders),
        "graph_preparation": (
            "all candidate graphs of a pass captured before its "
            "interleaved timed replay"
        ),
        "units": {
            "call": "microseconds per synchronized Python call",
            "batched_event": (
                "microseconds per launch in a CUDA-event batch of "
                "launches_per_sample launches"
            ),
            "graph": (
                "microseconds per launch in a captured graph replay of "
                "launches_per_sample launches"
            ),
        },
    }
    return results, method


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
        measure: Callable taking the launch of every candidate and the bytes
            one launch has to move, and returning their timings and the
            timing method.
        seed: Seed of the input values.

    Returns:
        One row per problem size.
    """
    from swage._benchmark import _fixed_vector_add_kernel

    swage_kernel = _fixed_vector_add_kernel()
    triton_kernel = _make_triton_vadd()
    results = []
    for exponent in _VADD_EXPONENTS:
        n = 1 << exponent
        grid = ((n + _VADD_SWAGE_BLOCK - 1) // _VADD_SWAGE_BLOCK,)
        generator = torch.Generator().manual_seed(seed)
        x = torch.randn(n, generator=generator).cuda()
        y = torch.randn(n, generator=generator).cuda()
        outputs = {
            "swage": torch.full_like(x, float("nan")),
            "torch": torch.full_like(x, float("nan")),
        }
        launches = {
            "swage": lambda: swage_kernel.launch(
                arguments={
                    "x_ptr": x,
                    "y_ptr": y,
                    "output_ptr": outputs["swage"],
                    "n": n,
                },
                constexprs={"BLOCK": _VADD_SWAGE_BLOCK},
                grid=grid,
            ),
            "torch": lambda: torch.add(x, y, out=outputs["torch"]),
        }
        for triton_block in _VADD_TRITON_BLOCKS:
            triton_grid = ((n + triton_block - 1) // triton_block,)
            output = torch.full_like(x, float("nan"))
            name = f"triton_b{triton_block}"
            outputs[name] = output
            launches[name] = (
                lambda block=triton_block, grid=triton_grid, out=output: (
                    triton_kernel[grid](x, y, out, n, BLOCK=block)
                )
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
        # Two f32 inputs read and one f32 output written.
        useful_bytes = 12 * n
        timings, timing_method = measure(launches, useful_bytes)
        results.append(
            {
                "case": "vadd",
                "n": n,
                "seed": seed,
                "useful_bytes": useful_bytes,
                "swage_block": _VADD_SWAGE_BLOCK,
                "swage_grid": grid[0],
                "triton_sweep_blocks": list(_VADD_TRITON_BLOCKS),
                "launch_contract": {
                    "swage": "BLOCK=256 for the vector-add campaign",
                    "triton": "compile-time BLOCK, one program per block",
                },
                "timing_method": timing_method,
                "timings": timings,
            }
        )
    return results


def _segmented_cases(
    distributions, segment_count, seeds, generate, load_real_trace
):
    """Yield one segmented case per distribution and seed, in that order.

    Args:
        distributions: Distribution names; the real trace may be one.
        segment_count: Segments of every synthetic case.
        seeds: Seeds of the cases of each distribution.
        generate: ``distributions.generate_lengths`` or a stand-in.
        load_real_trace: ``real_traces.load_real_trace`` or a stand-in. It is
            called once, when the trace is named.

    Yields:
        The distribution, the seed, the lengths, and for the trace its
        provenance. The seed does not change the lengths of the trace; it
        draws the random values of its rows.
    """
    trace = None
    for name in distributions:
        for seed in seeds:
            if name != _REAL_TRACE_NAME:
                yield {
                    "distribution": name,
                    "seed": seed,
                    "lengths": generate(name, segment_count, seed),
                }
                continue
            if trace is None:
                trace = load_real_trace(name)
            lengths, provenance = trace
            yield {
                "distribution": name,
                "seed": seed,
                "lengths": lengths,
                "trace_provenance": provenance,
            }


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


def _sum_tolerance(torch, lengths, magnitude, quantum):
    """Return how far a correct f32 segment sum may be from the exact sum.

    Args:
        torch: The PyTorch module.
        lengths: Segment lengths as float64.
        magnitude: Float64 sum of the absolute values of each segment.
        quantum: Grid that every value is a multiple of, or None.

    Returns:
        One float64 tolerance per segment. It is zero where every partial
        sum in any order is exactly representable: the values lie on the
        grid and their magnitudes add up to at most ``2 ** 24`` grid steps.
        Elsewhere it is ``gamma(n - 1) * magnitude`` with ``gamma(k) =
        k * u / (1 - k * u)`` and ``u = 2 ** -24``, the bound that holds for
        n f32 values added in any order. Past ``k * u = 1 / 2`` the bound
        says nothing and the tolerance is infinite: the segment is not
        checked beyond having been written.
    """
    steps = (lengths - 1).clamp(min=0) * _F32_UNIT_ROUNDOFF
    tolerance = torch.where(
        steps < 0.5,
        steps / (1 - steps) * magnitude,
        torch.full_like(magnitude, float("inf")),
    )
    if quantum is not None:
        exact = magnitude <= _F32_EXACT_INTEGERS * quantum
        tolerance = torch.where(exact, torch.zeros_like(tolerance), tolerance)
    return tolerance


def _sum_reference(torch, host_values, host_offsets, quantum):
    """Return the float64 segment sums and the tolerance of each."""
    values = host_values.double()
    reference = torch.segment_reduce(values, "sum", offsets=host_offsets)
    magnitude = torch.segment_reduce(values.abs(), "sum", offsets=host_offsets)
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


def _check_on_position_dependent_values(
    torch, launches, offsets, segment_count
):
    """Check relaunchable candidates on values that expose a misread.

    On all-one values any in-bounds window of the right length gives the
    right sum. This check runs every candidate again on the seeded quarter
    multiples of ``_exact_values``, which make a shifted, short, or long read
    visible, against a float64 CPU reference: exactly where those sums are
    exact in f32 in any order, which is every segment shorter than two
    million elements, and within the any-order bound elsewhere. It uses its
    own values and output and leaves the timed inputs alone.

    Args:
        torch: The PyTorch module.
        launches: Callable taking values and an output, by candidate name.
        offsets: Device offsets of the row being measured.
        segment_count: Number of segments.

    Raises:
        AssertionError: If a candidate misses its tolerance in any segment.
    """
    if not launches:
        return
    host_offsets = offsets.cpu()
    host_values = _exact_values(torch, int(host_offsets[-1]))
    reference, tolerance = _sum_reference(
        torch, host_values, host_offsets, _QUANTUM["quarters"]
    )
    reference = reference.to(offsets.device)
    tolerance = tolerance.to(offsets.device)
    values = host_values.to(offsets.device)
    output = torch.empty(segment_count, device=offsets.device)
    for name, launch in launches.items():
        output.fill_(float("nan"))
        launch(values, output)
        _check_sums(
            torch,
            f"{name} on position-dependent values",
            output,
            reference,
            tolerance,
        )


def _check_on_exact_values(
    torch, family, launch, configs, offsets, segment_count
):
    """Check every configuration of a swept baseline on quarter values.

    Args:
        torch: The PyTorch module.
        family: Family name of the baseline; a configuration is named
            ``{family}_b{block}_w{warps}``.
        launch: Callable taking values, an output, a block, and warps.
        configs: The block and warp configurations to check.
        offsets: Device offsets of the distribution being measured.
        segment_count: Number of segments.

    Raises:
        AssertionError: If a configuration misses the reference in any
            segment; see ``_check_on_position_dependent_values``.
    """
    _check_on_position_dependent_values(
        torch,
        {
            f"{family}_b{block}_w{warps}": (
                lambda values, output, block=block, warps=warps: launch(
                    values, output, block, warps
                )
            )
            for block, warps in configs
        },
        offsets,
        segment_count,
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


def _padded_inputs(torch, values, offsets):
    """Pad every segment with zeros to the longest one.

    Returns:
        The f32 matrix with one row per segment and its bool mask. Without
        a nonempty segment the matrix has no columns.
    """
    lengths = offsets[1:] - offsets[:-1]
    width = int(lengths.max()) if lengths.numel() else 0
    index = offsets[:-1, None] + torch.arange(
        width, dtype=torch.int32, device=offsets.device
    )
    mask = index < offsets[1:, None]
    gathered = values[index.clamp_(max=max(values.numel() - 1, 0))]
    return torch.where(mask, gathered, torch.zeros_like(gathered)), mask


def _padded_sum(padded, mask):
    """Reduce a padded matrix under its mask, in pure PyTorch."""
    return (padded * mask).sum(dim=1)


def _padded_layout(lengths) -> dict[str, object]:
    """Describe the f32 matrix that pads every segment to the longest one."""
    rows = len(lengths)
    columns = max(lengths, default=0)
    packed_elements = sum(lengths)
    padded_elements = rows * columns
    padding_elements = padded_elements - packed_elements
    return {
        "rows": rows,
        "columns": columns,
        "packed_elements": packed_elements,
        "padded_elements": padded_elements,
        "padding_elements": padding_elements,
        "padding_fraction": (
            padding_elements / padded_elements if padded_elements else 0.0
        ),
        "storage_bytes": padded_elements * 4,
        "dtype": "float32",
        "input_materialization_timed": False,
        "output_preallocated": True,
    }


def _free_device_bytes(torch) -> int:
    """Return the device memory a new tensor could use right now."""
    free, _ = torch.cuda.mem_get_info()
    return free + torch.cuda.memory_reserved() - torch.cuda.memory_allocated()


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
        short_matrix = tl.reshape(short_values, (SHORT_TASK_SLOTS, WARP_LANES))
        short_totals = tl.sum(short_matrix, axis=1)
        output_slots = tl.arange(0, SHORT_TASK_SLOTS)
        output_tasks = program_id * SHORT_TASK_SLOTS + output_slots
        output_active = is_warp_program & (output_tasks < warp_task_count)
        output_segments = tl.load(
            warp_task_ids + output_tasks, mask=output_active, other=0
        )
        tl.store(output + output_segments, short_totals, mask=output_active)

        cta_task = program_id - WARP_PROGRAMS
        cta_active = (~is_warp_program) & (cta_task < cta_task_count)
        cta_segment = tl.load(cta_task_ids + cta_task, mask=cta_active, other=0)
        cta_begin = tl.load(offsets + cta_segment, mask=cta_active, other=0)
        cta_end = tl.load(offsets + cta_segment + 1, mask=cta_active, other=0)
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
    def looped_task_kernel(
        values, offsets, output, task_ids, BLOCK: tl.constexpr
    ):
        segment_id = tl.load(task_ids + tl.program_id(0))
        begin = tl.load(offsets + segment_id)
        end = tl.load(offsets + segment_id + 1)
        total = tl.zeros((BLOCK,), dtype=tl.float32)
        for start in range(begin, end, BLOCK):
            index = start + tl.arange(0, BLOCK)
            total += tl.load(values + index, mask=index < end, other=0.0)
        tl.store(output + segment_id, tl.sum(total, axis=0))

    return looped_task_kernel


def _make_triton_kernels():
    """Define every Triton segmented-sum kernel of the harnesses lazily."""
    packed, cta = _make_triton_matched_task_partition()
    return types.SimpleNamespace(
        fixed=_make_triton_segmented_sum(),
        looped=_make_triton_looped_sum(),
        packed=packed,
        cta=cta,
        cta_looped=_make_triton_looped_task_sum(),
        fused=_make_triton_fused_sum(),
    )


def _triton_sum_configs(max_length: int) -> list[tuple[int, int]]:
    """Return legal Triton segmented-sum sweep configs."""
    configs = []
    for block in _FIXED_BLOCKS:
        if block < max_length:
            continue
        for warps in (1, 2, 4, 8):
            if warps <= block // 32:
                configs.append((block, warps))
    return configs


def _triton_looped_configs() -> list[tuple[int, int]]:
    """Return the looped Triton sweep, independent of segment lengths."""
    return [
        (block, warps)
        for block in _LOOPED_BLOCKS
        for warps in (1, 2, 4, 8)
        if warps <= block // 32
    ]


def _partition_tasks(torch, offsets):
    """Split the segment ids into warp and CTA task lists on the device.

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


def _partition_lengths(lengths: Iterable[int]) -> tuple[list[int], list[int]]:
    """Partition segment IDs on the host at the fixed short-task boundary.

    This is the host classification of the matched task partition, which
    the orchestration measurements time.

    Raises:
        ValueError: If a length is not a nonnegative integer, or exceeds the
            4096 elements of one CTA task.
    """
    warp_ids = []
    cta_ids = []
    for segment_id, length in enumerate(lengths):
        if type(length) is not int or length < 0:
            raise ValueError("segment lengths must be nonnegative integers")
        if length <= _WARP_MAX_ELEMENTS:
            warp_ids.append(segment_id)
        elif length <= _PLANNED_CTA_BLOCK:
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


def _packed_programs(warp_tasks: int) -> int:
    """Return the programs that pack four short tasks each."""
    return (warp_tasks + _PACKED_TASKS - 1) // _PACKED_TASKS


def _launch_triton_planned(
    kernels,
    values,
    offsets,
    output,
    warp_ids,
    cta_ids,
    *,
    warps: int,
    block: int | None = None,
):
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
        kernels.packed[(_packed_programs(warp_count),)](
            values,
            offsets,
            output,
            warp_ids,
            warp_count,
            TASKS=_PACKED_TASKS,
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


def _launch_triton_fused(kernels, values, offsets, output, warp_ids, cta_ids):
    """Launch the one-launch fused Triton sum over two task lists."""
    warp_programs = _packed_programs(warp_ids.numel())
    kernels.fused[(warp_programs + cta_ids.numel(),)](
        values,
        offsets,
        output,
        warp_ids,
        warp_ids.numel(),
        cta_ids,
        cta_ids.numel(),
        WARP_PROGRAMS=warp_programs,
        LOGICAL_LANES=_FUSED_LANES,
        SHORT_TASK_SLOTS=_PACKED_TASKS,
        WARP_LANES=32,
        MAX_CTA_ELEMENTS=_FUSED_MAX_ELEMENTS,
        num_warps=_FUSED_NUM_WARPS,
    )
    return output


def _launch_triton_looped(
    kernel, values, offsets, output, segment_count, block: int, warps: int
):
    """Launch the looped Triton sum with one program per segment."""
    kernel[(segment_count,)](
        values, offsets, output, BLOCK=block, num_warps=warps
    )
    return output


def _triton_relaunches(kernels, offsets, segment_count, warp_ids, cta_ids):
    """Return every Triton candidate as a launch on given values and output.

    The correctness checks run a candidate on other values than the timed
    ones; the timed launch binds the timed values.
    """
    launches = {}
    for block, warps in _triton_sum_configs(0):
        launches[f"triton_b{block}_w{warps}"] = functools.partial(
            lambda values, output, block, warps: kernels.fixed[
                (segment_count,)
            ](
                values,
                offsets,
                output,
                segment_count,
                BLOCK=block,
                num_warps=warps,
            ),
            block=block,
            warps=warps,
        )
    for warps in _PLANNED_WARPS:
        launches[_matched_name(warps)] = functools.partial(
            lambda values, output, warps: _launch_triton_planned(
                kernels, values, offsets, output, warp_ids, cta_ids, warps=warps
            ),
            warps=warps,
        )
    launches["triton_fused"] = lambda values, output: _launch_triton_fused(
        kernels, values, offsets, output, warp_ids, cta_ids
    )
    for block, warps in _triton_looped_configs():
        launches[f"triton_looped_b{block}_w{warps}"] = functools.partial(
            lambda values, output, block, warps: _launch_triton_looped(
                kernels.looped,
                values,
                offsets,
                output,
                segment_count,
                block,
                warps,
            ),
            block=block,
            warps=warps,
        )
        launches[f"triton_planned_looped_b{block}_w{warps}"] = (
            functools.partial(
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
                block=block,
                warps=warps,
            )
        )
    return launches


def _require_empty_compilation_caches() -> dict[str, bool]:
    """Require fresh, separate compiler-cache directories."""
    directories = []
    for variable in ("SWAGE_CACHE_DIR", "TRITON_CACHE_DIR"):
        value = os.environ.get(variable)
        if not value:
            raise RuntimeError(
                f"{variable} must name an existing empty directory"
            )
        directory = pathlib.Path(value).resolve()
        if (
            not directory.is_dir()
            or next(directory.iterdir(), None) is not None
        ):
            raise RuntimeError(
                f"{variable} must name an existing empty directory"
            )
        directories.append(directory)
    if directories[0].samefile(directories[1]):
        raise RuntimeError(
            "SWAGE_CACHE_DIR and TRITON_CACHE_DIR must be distinct"
        )
    return {
        "fresh_process": True,
        "unique_directories": True,
        "swage_initially_empty": True,
        "triton_initially_empty": True,
    }


def _swage_compile_steps(torch, cases):
    """Return the Swage kernels that the prepared sums of the cases compile.

    The prepared sum compiles a warp and a CTA kernel, the fused mixed
    kernel when a case has a segment that one task covers, and the split
    partial and merge kernels when a case splits a segment. The steps use
    the program, kernel name, target, and options of
    ``_prepare_planned_sum``, so the later preparations find the kernels in
    the in-process memo of ``_segmented_runtime`` and compile nothing.

    Returns:
        ``(component, compile)`` pairs in compile order; each ``compile``
        compiles one kernel and takes no arguments.
    """
    import numpy
    from swage import _segmented_plan as plan
    from swage import _segmented_programs as programs
    from swage import _segmented_runtime as execution

    module_text = programs._semantic_module("sum")
    warp_max, cta_chunk = plan._planning_limits(_WARP_MAX_ELEMENTS, None)
    blocks = execution._target_description()
    native = execution._native_swage()
    target = execution._target(torch, torch.cuda.current_device())
    direct = split = False
    for case in cases:
        lengths = case["lengths"]
        host = numpy.array(_host_offsets(lengths), dtype=numpy.int32)
        _, warp_tasks, cta_tasks, partial_tasks, _ = native._classify_segments(
            host,
            value_count=int(host[-1]),
            segment_count=len(lengths),
            warp_max_elements=warp_max,
            cta_chunk_elements=cta_chunk,
        )
        direct = direct or bool(warp_tasks + cta_tasks)
        split = split or bool(partial_tasks)
    steps = [
        (
            "swage_warp",
            "_compile_segmented_reduction_ptx",
            {"block_size": blocks.subgroup_width, "use_task_ids": True},
        ),
        (
            "swage_cta",
            "_compile_segmented_reduction_ptx",
            {"block_size": blocks.cta_block_threads, "use_task_ids": True},
        ),
    ]
    if direct:
        steps.append(
            ("swage_mixed", "_compile_fused_segmented_reduction_ptx", {})
        )
    if split:
        steps.append(
            ("swage_partial", "_compile_split_partial_reduction_ptx", {})
        )
        steps.append(("swage_merge", "_compile_split_merge_reduction_ptx", {}))
    return [
        (
            component,
            functools.partial(
                execution._compile_once,
                getattr(native, compiler),
                module_text,
                kernel_name="segmented_sum",
                target=target,
                **options,
            ),
        )
        for component, compiler, options in steps
    ]


def _case_signature(case, candidates) -> dict[str, object]:
    """Return the Triton scalar signature and timed candidates of a case."""
    lengths = case["lengths"]
    warp_tasks = sum(length <= _WARP_MAX_ELEMENTS for length in lengths)
    return {
        "distribution": case["distribution"],
        "seed": case["seed"],
        "value_count": sum(lengths),
        "segment_count": len(lengths),
        "warp_task_count": warp_tasks,
        "cta_task_count": len(lengths) - warp_tasks,
        "warp_programs": _packed_programs(warp_tasks),
        "candidates": candidates,
    }


def _measure_segmented_compilation(
    torch, cases, kernels, *, only=None, exclude=()
) -> dict[str, object]:
    """Compile every specialization before correctness or warmup launches.

    Args:
        torch: The PyTorch module.
        cases: The segmented cases, from ``_segmented_cases``.
        kernels: ``_make_triton_kernels()``.
        only: Candidate filter selectors to keep, or None for all.
        exclude: Candidate filter selectors to leave out.

    Returns:
        The compilation phase of the record. Every Swage kernel and every
        Triton specialization that a timed candidate of some case launches
        is compiled once, in a fixed component order, and timed by itself.
    """
    cache_policy = _require_empty_compilation_caches()
    signatures = []
    for case in cases:
        candidates = _case_candidates(max(case["lengths"]), only, exclude)
        signatures.append(_case_signature(case, candidates))
    if not signatures:
        raise ValueError("segmented compilation requires at least one case")

    timings = {}
    component_order = []

    def measure(name, operation):
        begin = time.perf_counter_ns()
        operation()
        sample = (time.perf_counter_ns() - begin) / 1_000
        if sample <= 0:
            raise RuntimeError(f"{name} compilation timer did not advance")
        component_order.append(name)
        timings[name] = {
            "samples_us": [sample],
            "summary_us": summarize_us([sample]),
        }

    def timed(signature, name):
        return any(
            candidate == name or _family(candidate) == name
            for candidate in signature["candidates"]
        )

    swage_components = []
    if any(
        candidate.startswith("swage_")
        for signature in signatures
        for candidate in signature["candidates"]
    ):
        for component, compile_kernel in _swage_compile_steps(torch, cases):
            measure(component, compile_kernel)
            swage_components.append(component)

    # Dtypes become Triton MockTensor pointers in JITFunction.warmup. Actual
    # scalar arguments still participate in Triton's runtime specialization.
    triton_components = []
    fixed_configs = sorted(
        {
            (block, warps)
            for signature in signatures
            for block, warps in _triton_sum_configs(0)
            if timed(signature, f"triton_b{block}_w{warps}")
        }
    )

    def compile_fixed(block, warps):
        for signature in signatures:
            if timed(signature, f"triton_b{block}_w{warps}"):
                kernels.fixed.warmup(
                    torch.float32,
                    torch.int32,
                    torch.float32,
                    signature["segment_count"],
                    BLOCK=block,
                    num_warps=warps,
                    grid=(signature["segment_count"],),
                )

    for block, warps in fixed_configs:
        component = f"triton_b{block}_w{warps}"
        measure(component, functools.partial(compile_fixed, block, warps))
        triton_components.append(component)

    def packs(signature):
        return signature["warp_task_count"] and (
            timed(signature, _MATCHED)
            or timed(signature, "triton_planned_looped")
        )

    if any(packs(signature) for signature in signatures):

        def compile_packed():
            for signature in signatures:
                if packs(signature):
                    kernels.packed.warmup(
                        torch.float32,
                        torch.int32,
                        torch.float32,
                        torch.int32,
                        signature["warp_task_count"],
                        TASKS=_PACKED_TASKS,
                        WARP=32,
                        num_warps=4,
                        grid=(signature["warp_programs"],),
                    )

        measure("triton_matched_packed", compile_packed)
        triton_components.append("triton_matched_packed")

    matched_warps = [
        warps
        for warps in _PLANNED_WARPS
        if any(
            signature["cta_task_count"]
            and timed(signature, _matched_name(warps))
            for signature in signatures
        )
    ]

    def compile_cta(warps):
        for signature in signatures:
            count = signature["cta_task_count"]
            if count and timed(signature, _matched_name(warps)):
                kernels.cta.warmup(
                    torch.float32,
                    torch.int32,
                    torch.float32,
                    torch.int32,
                    count,
                    BLOCK=_PLANNED_CTA_BLOCK,
                    num_warps=warps,
                    grid=(count,),
                )

    for warps in matched_warps:
        component = f"triton_matched_cta_w{warps}"
        measure(component, functools.partial(compile_cta, warps))
        triton_components.append(component)

    fused_programs = sorted(
        {
            signature["warp_programs"]
            for signature in signatures
            if timed(signature, "triton_fused")
        }
    )
    if fused_programs:

        def compile_fused():
            for signature in signatures:
                if timed(signature, "triton_fused"):
                    kernels.fused.warmup(
                        torch.float32,
                        torch.int32,
                        torch.float32,
                        torch.int32,
                        signature["warp_task_count"],
                        torch.int32,
                        signature["cta_task_count"],
                        WARP_PROGRAMS=signature["warp_programs"],
                        LOGICAL_LANES=_FUSED_LANES,
                        SHORT_TASK_SLOTS=_PACKED_TASKS,
                        WARP_LANES=32,
                        MAX_CTA_ELEMENTS=_FUSED_MAX_ELEMENTS,
                        num_warps=_FUSED_NUM_WARPS,
                        grid=(
                            signature["warp_programs"]
                            + signature["cta_task_count"],
                        ),
                    )

        measure("triton_fused", compile_fused)
        triton_components.append("triton_fused")

    # The looping kernels take no scalar argument, so one specialization
    # per configuration serves every case.
    looped_configs = [
        (block, warps)
        for block, warps in _triton_looped_configs()
        if any(
            timed(signature, f"triton_looped_b{block}_w{warps}")
            for signature in signatures
        )
    ]
    for block, warps in looped_configs:
        component = f"triton_looped_b{block}_w{warps}"
        measure(
            component,
            functools.partial(
                kernels.looped.warmup,
                torch.float32,
                torch.int32,
                torch.float32,
                BLOCK=block,
                num_warps=warps,
                grid=(1,),
            ),
        )
        triton_components.append(component)
    planned_looped_configs = [
        (block, warps)
        for block, warps in _triton_looped_configs()
        if any(
            signature["cta_task_count"]
            and timed(signature, f"triton_planned_looped_b{block}_w{warps}")
            for signature in signatures
        )
    ]
    for block, warps in planned_looped_configs:
        component = f"triton_planned_looped_cta_b{block}_w{warps}"
        measure(
            component,
            functools.partial(
                kernels.cta_looped.warmup,
                torch.float32,
                torch.int32,
                torch.float32,
                torch.int32,
                BLOCK=block,
                num_warps=warps,
                grid=(1,),
            ),
        )
        triton_components.append(component)

    derived_totals = {
        "swage_total": swage_components,
        "triton_total": triton_components,
    }
    for name, components in derived_totals.items():
        if not components:
            continue
        sample = sum(
            timings[component]["samples_us"][0] for component in components
        )
        timings[name] = {
            "samples_us": [sample],
            "summary_us": summarize_us([sample]),
        }
    return {
        "status": "measured",
        "case": "segmented-sum",
        "timing_method": (
            "one time.perf_counter_ns wall-clock sample per ordered "
            "compile-only component; totals are exact component sums, "
            "not isolated cold starts"
        ),
        "compiler_order": ["swage", "triton"],
        "component_order": component_order,
        "configurations": {
            "swage_kernels": [
                component[len("swage_") :] for component in swage_components
            ],
            "triton_fixed": [
                {"block": block, "num_warps": warps}
                for block, warps in fixed_configs
            ],
            "triton_matched_cta_num_warps": matched_warps,
            "triton_fused_warp_programs": fused_programs,
            "triton_looped": [
                {"block": block, "num_warps": warps}
                for block, warps in looped_configs
            ],
            "triton_planned_looped_cta": [
                {"block": block, "num_warps": warps}
                for block, warps in planned_looped_configs
            ],
            "triton_case_signatures": signatures,
        },
        "derived_totals": {
            name: components
            for name, components in derived_totals.items()
            if components
        },
        "cache_policy": cache_policy,
        "scope": {
            "swage": {
                "included": [
                    "semantic-module parse inside the compile miss",
                    "MLIR/LLVM/NVPTX lowering and verification",
                    "PTX production",
                    "launch-contract check",
                    "in-process kernel memo store; the private segmented "
                    "path has no persistent cache",
                ],
                "excluded": [
                    "imports",
                    "host classification that selects the kernels",
                    "planning admission",
                    "contract binding",
                    "CUDA module lease/load",
                    "first launch",
                    "device synchronization",
                ],
            },
            "triton": {
                "included": [
                    "JITFunction.warmup compilation for actual "
                    "per-case scalars",
                    "runtime-scalar and constexpr specialization",
                    "persistent-cache write",
                    "duplicate-signature in-process cache hits "
                    "within each group",
                ],
                "excluded": [
                    "imports",
                    "JIT-function construction",
                    "kernel launch",
                    "device synchronization",
                ],
            },
        },
        "not_applicable": {
            "torch_segment_reduce": (
                "eager PyTorch operator; no measured JIT phase"
            ),
            "torch_padded": "eager PyTorch operator; no measured JIT phase",
        },
        "timings": timings,
    }


def _plan_swage_sum(torch, values, offsets, output):
    """Plan a prepared sum without compiling, binding, or loading.

    This is the planning part of ``_prepare_planned_sum``: validation with
    host classification, planning admission (a per-process memo hit after
    the first preparation), the upload of the task records, and the split
    scratch, followed by a device synchronization. The output is only
    validated.

    Returns:
        The uploaded records, the scratch, and the warp, CTA, partial, and
        merge task counts.
    """
    from swage import _segmented_plan as plan
    from swage import _segmented_programs as programs
    from swage import _segmented_validation as validation

    module_text = programs._semantic_module("sum")
    warp_max, cta_chunk = plan._planning_limits(_WARP_MAX_ELEMENTS, None)
    validate, found = plan._classifying_validator(warp_max, cta_chunk)
    value_count, segment_count, host_offsets = validation._validate_shapes(
        values,
        offsets,
        output,
        validate,
        element=programs._program_element(module_text),
    )
    plan._admit_program(module_text, "segmented_sum", warp_max, cta_chunk)
    records, warp_tasks, cta_tasks, partial_tasks, merge_tasks = (
        plan._classification(
            found,
            host_offsets,
            value_count=value_count,
            segment_count=segment_count,
            warp_max_elements=warp_max,
            cta_chunk_elements=cta_chunk,
        )
    )
    task_records = scratch = None
    if len(records):
        task_records = torch.tensor(
            records, dtype=torch.int32, device=offsets.device
        )
    if partial_tasks:
        scratch = torch.empty(
            partial_tasks, dtype=values.dtype, device=offsets.device
        )
    torch.cuda.synchronize()
    return types.SimpleNamespace(
        records=task_records,
        scratch=scratch,
        counts=(warp_tasks, cta_tasks, partial_tasks, merge_tasks),
    )


def _wall_clock_interleaved_us(
    operations: dict[str, Callable[[int], object]],
    warmups: int,
    samples: int,
    *,
    unit: str = "microseconds per synchronized end-to-end operation",
) -> tuple[dict[str, object], dict[str, object]]:
    """Measure complete synchronized operations in rotating order."""
    for round_index, order in enumerate(_rotating_orders(operations, warmups)):
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
        name: {"samples_us": values, "summary_us": summarize_us(values)}
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
    kernels,
    prepare,
    timed,
    lengths: list[int],
    values,
    offsets,
    warmups: int,
    samples: int,
) -> dict[str, object]:
    """Measure compilation-free planning and warm complete orchestration.

    Args:
        torch: The PyTorch module.
        kernels: ``_make_triton_kernels()``.
        prepare: The private planned-sum preparation function.
        timed: The candidates the row times; the orchestration covers
            ``swage_mixed`` and ``triton_matched_task_partition`` when the
            row times them.
        lengths: Segment lengths of the row.
        values: Device values of the row.
        offsets: Device offsets of the row.
        warmups: Untimed rounds before each phase.
        samples: Timed rounds per phase.

    Returns:
        The ``planning`` and ``end_to_end`` phases of the row. Without
        either candidate both are marked as not run, with the reason.
    """
    from swage._segmented_validation import (
        _validate_offsets,
        _validate_shapes,
    )

    candidates = [name for name in _ORCHESTRATION if name in timed]
    if not candidates:
        not_run = {
            "status": "not-run",
            "reason": (
                "the row times neither swage_mixed nor "
                "triton_matched_task_partition"
            ),
        }
        return {"planning": not_run, "end_to_end": dict(not_run)}

    def prepare_triton_only(device_offsets, output):
        _, _, validated_offsets = _validate_shapes(
            values, device_offsets, output, _validate_offsets
        )
        bounds = validated_offsets.tolist()
        warp_ids, cta_ids = _partition_lengths(
            [end - begin for begin, end in itertools.pairwise(bounds)]
        )
        device = device_offsets.device
        return (
            torch.tensor(warp_ids, device=device, dtype=torch.int32),
            torch.tensor(cta_ids, device=device, dtype=torch.int32),
        )

    def run_swage(device_offsets):
        output = torch.empty(len(lengths), device=offsets.device)
        prepare(
            values,
            device_offsets,
            output,
            warp_max_elements=_WARP_MAX_ELEMENTS,
        ).mixed()
        torch.cuda.synchronize()

    def run_triton(device_offsets):
        output = torch.empty(len(lengths), device=offsets.device)
        warp_ids, cta_ids = prepare_triton_only(device_offsets, output)
        _launch_triton_planned(
            kernels, values, device_offsets, output, warp_ids, cta_ids, warps=1
        )
        torch.cuda.synchronize()

    runs = {"swage_mixed": run_swage, _MATCHED: run_triton}
    for name in candidates:
        runs[name](offsets)

    planning_outputs = {
        name: torch.empty(len(lengths), device=offsets.device)
        for name in candidates
    }

    def plan_triton(_):
        descriptors = prepare_triton_only(offsets, planning_outputs[_MATCHED])
        torch.cuda.synchronize()
        return descriptors

    planning_operations = {
        "swage_mixed": lambda _: _plan_swage_sum(
            torch, values, offsets, planning_outputs["swage_mixed"]
        ),
        _MATCHED: plan_triton,
    }
    planning_timings, planning_method = _wall_clock_interleaved_us(
        {name: planning_operations[name] for name in candidates},
        warmups,
        samples,
        unit="microseconds per synchronized planning operation",
    )
    fixed_timings, fixed_method = _wall_clock_interleaved_us(
        {name: lambda _, run=runs[name]: run(offsets) for name in candidates},
        warmups,
        samples,
    )

    def changing_offsets(round_index):
        shift = round_index % len(lengths)
        changed_lengths = lengths[shift:] + lengths[:shift]
        return torch.tensor(
            _host_offsets(changed_lengths),
            device=offsets.device,
            dtype=torch.int32,
        )

    changing_timings, changing_method = _wall_clock_interleaved_us(
        {
            name: lambda round_index, run=runs[name]: run(
                changing_offsets(round_index)
            )
            for name in candidates
        },
        warmups,
        samples,
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
        "planning": {
            "geometry": "fixed offsets reused; output preallocated",
            "included": [
                "tensor/offset validation",
                "host task classification",
                "Swage planning admission, a per-process memo hit",
                "device descriptor/scratch materialization",
                "device synchronization",
            ],
            "excluded": [
                "output allocation",
                "kernel launch",
                "kernel compilation/memo lookup",
                "contract binding",
                "CUDA module lease/load",
            ],
            "timing_method": planning_method,
            "timings": planning_timings,
        },
        "end_to_end": {
            "artifact_jit_warmup": (
                "one untimed complete operation per candidate before all "
                "planning and end-to-end samples"
            ),
            "compilation_excluded": True,
            "graph_samples_combined": False,
            "warm_preparation": {
                "geometry": "fixed offsets reused; plan preparation repeated",
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


def _segmented_row(
    torch,
    kernels,
    prepare,
    measure,
    orchestrate,
    case,
    *,
    values_kind: str,
    device,
    free_bytes: Callable[[], int],
    synchronize,
    only=None,
    exclude=(),
) -> dict[str, object]:
    """Check and time the selected candidates on one segmented case.

    Args:
        torch: The PyTorch module.
        kernels: The Triton kernels of ``_make_triton_kernels``.
        prepare: The private planned-sum preparation function.
        measure: Callable taking the launch of every candidate and the bytes
            one launch has to move, and returning their timings and the
            timing method.
        orchestrate: Callable taking the timed launches, the lengths, the
            device values, and the device offsets, and returning the
            ``planning`` and ``end_to_end`` phases.
        case: The distribution, seed, lengths, and any trace provenance.
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
        or checked. ``candidate_order`` is the base order of the timed
        candidates.

    Raises:
        AssertionError: If a candidate that ran is wrong in any segment.
        ValueError: If the filter leaves no candidate the row can run.
    """
    name = case["distribution"]
    lengths = case["lengths"]
    seed = case["seed"]
    segment_count = len(lengths)
    statistics_summary = summarize_lengths(lengths)
    max_length = statistics_summary["max"]
    host_offsets = torch.tensor(
        [0, *itertools.accumulate(lengths)], dtype=torch.int32
    )
    host_values = _values(torch, values_kind, statistics_summary["total"], seed)
    reference, tolerance = _sum_reference(
        torch, host_values, host_offsets, _QUANTUM[values_kind]
    )
    check = _check_modes(tolerance)
    values = host_values.to(device)
    offsets = host_offsets.to(device)
    reference = reference.to(device)
    tolerance = tolerance.to(device)
    useful_bytes = _useful_bytes(statistics_summary["total"], segment_count)

    unable = _length_limits(max_length)
    padded_bytes = segment_count * max_length * _PADDED_BYTES_PER_ELEMENT
    memory_budget = free_bytes()
    if padded_bytes > memory_budget:
        unable["torch_padded"] = (
            f"padding {segment_count} segments to {max_length} elements "
            f"needs {padded_bytes} bytes and {memory_budget} are free"
        )
    warp_ids, cta_ids = _partition_tasks(torch, offsets)
    relaunches = _triton_relaunches(
        kernels, offsets, segment_count, warp_ids, cta_ids
    )
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
            allocated["torch_segment_reduce"] = torch.segment_reduce(
                values, "sum", offsets=offsets
            )
            return allocated["torch_segment_reduce"]

        return launch

    def padded():
        # The padding is prepared outside the timed launch, as the Swage
        # plan and the Triton task lists are; the output is preallocated.
        matrix, _ = _padded_inputs(torch, values, offsets)
        output = unwritten("torch_padded")
        return lambda: torch.sum(matrix, dim=1, out=output)

    def triton(candidate):
        output = unwritten(candidate)
        return lambda: relaunches[candidate](values, output)

    # Every candidate the row can run, in base order, with the setup that
    # makes its launch. Only a selected candidate is set up.
    setups = {
        "swage_warp": lambda: swage("warp"),
        "swage_cta": lambda: swage("cta"),
        "swage_mixed": lambda: swage("mixed"),
        "torch_segment_reduce": torch_reduce,
        "torch_padded": padded,
    }
    for candidate in _row_candidates(max_length):
        if candidate.startswith("triton_"):
            setups[candidate] = functools.partial(triton, candidate)
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
    _check_on_position_dependent_values(
        torch,
        {
            candidate: relaunches[candidate]
            for candidate in launches
            if candidate in relaunches
        },
        offsets,
        segment_count,
    )
    timings, timing_method = measure(launches, useful_bytes)
    orchestration = orchestrate(launches, lengths, values, offsets)
    row = {
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
            for block, warps in _triton_sum_configs(max_length)
        ],
        "triton_looped_sweep_configs": [
            {"block": block, "num_warps": warps}
            for block, warps in _triton_looped_configs()
        ],
        "matched_task_partition_triton": {
            "comparison": "same host task partition; not identical execution",
            "launches": (
                "one packed short-task kernel plus one CTA-task kernel "
                "when both partitions are nonempty"
            ),
            "warp_threshold_elements": _WARP_MAX_ELEMENTS,
            "warp_tasks": warp_ids.numel(),
            "cta_tasks": cta_ids.numel(),
            "short_tasks_per_program": _PACKED_TASKS,
            "cta_block_elements": _PLANNED_CTA_BLOCK,
            "primary_cta_num_warps": 1,
            "cta_num_warps_sweep": list(_PLANNED_WARPS),
        },
        "triton_fused_contract": {
            "result_name": "triton_fused",
            "launch_count": 1,
            "logical_lanes_per_program": _FUSED_LANES,
            "short_task_slots": _PACKED_TASKS,
            "lanes_per_short_task": 32,
            "program_order": (
                "packed short-task programs first, then one CTA task "
                "per later program"
            ),
            "cta_accumulation": (
                "128-lane block-stride loads through 4096 elements"
            ),
            "maximum_segment_length": _FUSED_MAX_ELEMENTS,
            "physical_num_warps": _FUSED_NUM_WARPS,
            "warp_programs": _packed_programs(warp_ids.numel()),
            "cta_programs": cta_ids.numel(),
            "grid_programs": _packed_programs(warp_ids.numel())
            + cta_ids.numel(),
        },
        "timing_method": timing_method,
        "timings": timings,
        "planning": orchestration["planning"],
        "end_to_end": orchestration["end_to_end"],
        "padded_layout": _padded_layout(lengths),
    }
    if "trace_provenance" in case:
        row["trace_provenance"] = case["trace_provenance"]
    return row


def _run_segmented_sum(
    torch, measure, arguments, *, cases, kernels
) -> list[dict[str, object]]:
    """Benchmark private segmented sum against Triton and torch baselines."""
    from swage._segmented_qualification import _prepare_planned_sum

    def orchestrate(launches, lengths, values, offsets):
        return _orchestration_measurements(
            torch,
            kernels,
            _prepare_planned_sum,
            launches,
            lengths,
            values,
            offsets,
            arguments.warmups,
            arguments.samples,
        )

    return [
        _segmented_row(
            torch,
            kernels,
            _prepare_planned_sum,
            measure,
            orchestrate,
            case,
            values_kind=arguments.values,
            device="cuda",
            free_bytes=lambda: _free_device_bytes(torch),
            synchronize=torch.cuda.synchronize,
            only=arguments.candidates,
            exclude=arguments.exclude_candidates,
        )
        for case in cases
    ]


def _start_provenance(torch):
    """Initialize the CUDA context and start the provenance block."""
    torch.ones(1, device="cuda").sum().item()
    return benchmark_provenance.start(torch, benchmark_provenance.swage_build())


def _methodology(arguments, ticks) -> dict[str, object]:
    """Describe how the record was measured."""
    return {
        "suite": arguments.suite,
        "warmups_per_candidate_per_measurement": arguments.warmups,
        "samples_per_candidate_per_measurement": arguments.samples,
        "distributions": list(arguments.distributions),
        "segment_count": arguments.segment_count,
        "seeds": list(arguments.seeds),
        "values": arguments.values,
        "candidate_filter": {
            "candidates": arguments.candidates,
            "exclude_candidates": list(arguments.exclude_candidates),
        },
        "candidate_sampling": "deterministic rotating/interleaved order",
        "batched_launches": _BATCHED_LAUNCHES,
        "graph_replay_launches": _BATCHED_LAUNCHES,
        "launch_batching": (
            "event and graph samples start as batches of batched_launches "
            "launches; while one event timer tick is not below one percent "
            "of a candidate's median sample, that candidate's batch "
            "doubles and every candidate is sampled again in new rotating "
            "rounds, so the kept samples satisfy the limit themselves; "
            "every timing entry records its launches_per_sample, the tick, "
            "and the tick as a fraction of the sample"
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
        "graph_capture_before_interleaved_replay": True,
        "kernel_timing_compilation_excluded": True,
        "end_to_end_compilation_excluded_after_explicit_warmup": True,
        "planning_output_preallocated": True,
        "planning_kernel_launch_excluded": True,
        "planning_compilation_excluded": True,
        "end_to_end_not_combined_with_graph_samples": True,
        "correctness_checked_before_timing": True,
        "correctness": (
            "every timed candidate is checked on the timed values "
            "against a float64 CPU reference: exactly where sums of the "
            "values are exact in f32 in any order, otherwise within "
            "gamma(n - 1) times the sum of magnitudes; each row counts "
            "its exact, bounded, and unchecked segments; every output "
            "starts as NaN, so a sum that was never written fails"
        ),
        "position_dependent_check": (
            "besides the check on the timed values, every timed Triton "
            "candidate runs again on seeded nonzero quarter multiples, "
            "which make a shifted, short, or long read visible, and is "
            "checked against a float64 CPU reference before timing"
        ),
        "skipped": (
            "a baseline that cannot produce a correct sum on a row is "
            "listed in the row's skipped mapping with the reason and "
            "is not timed"
        ),
        "excluded": (
            "a candidate the filter leaves out is listed in the row's "
            "excluded list; it is not prepared, launched, or checked"
        ),
        "triton_dependency": "optional runtime import; not a project dep",
        "triton_fixed": (
            "one program per segment, one masked block; blocks smaller "
            "than the longest segment are excluded"
        ),
        "triton_matched_task_partition": (
            "the host partition of the segments at 32 elements; short "
            "tasks are packed four per program and each longer task reads "
            "one block of 4096 elements, so a longer segment skips the "
            "baseline; the base name launches the CTA kernel with one "
            "warp and the _w suffix names the other warp counts"
        ),
        "triton_fused": (
            "one launch of packed short-task programs followed by one "
            "program per longer task, which reads at most 4096 elements, "
            "so a longer segment skips the baseline"
        ),
        "triton_looped": (
            "one program per segment, a loop over the segment in fixed "
            "blocks; no block is excluded for the longest segment"
        ),
        "triton_planned_looped": (
            "the task lists of triton_matched_task_partition and its "
            "packed short tasks; each longer task loops over its segment "
            "in fixed blocks, with the block and warp sweep of "
            "triton_looped, so no segment length skips it"
        ),
        "torch_padded": (
            "pure PyTorch: every segment padded with zeros to the longest "
            "one outside the timed launch, then torch.sum over each row "
            "into a preallocated output inside it; a row whose padded "
            "matrix does not fit the free device memory skips it"
        ),
    }


def main(argv=None):
    """Run the selected comparison benchmark and write JSON evidence."""
    arguments = _arguments(argv)

    import torch
    import triton
    from swage import _cuda_backend, env

    if not torch.cuda.is_available():
        raise RuntimeError("benchmark requires CUDA-enabled PyTorch")
    root = pathlib.Path(__file__).resolve().parents[1]
    native = env.report()["native"]
    if native["available"] is not True:
        raise RuntimeError("comparison requires available native metadata")
    compiler = {
        key: native[key]
        for key in (
            "package_version",
            "source_revision",
            "source_clean",
            "llvm_version",
            "build_type",
        )
    }
    compiler["llvm_pin"] = (
        (root / "cmake" / "llvm-version.txt").read_text().strip()
    )
    device = torch.cuda.current_device()
    provenance = _start_provenance(torch)
    properties = torch.cuda.get_device_properties(device)
    capability = torch.cuda.get_device_capability(device)
    ticks = {
        "clock": benchmark_provenance.clock_tick_us(),
        "event": _event_tick_us(torch, "cuda"),
    }

    def measure(launches, useful_bytes):
        return _timings(
            torch,
            launches,
            arguments.warmups,
            arguments.samples,
            ticks=ticks,
            useful_bytes=useful_bytes,
        )

    result = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "swage-triton-comparison",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "source": _git_metadata(root),
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "pytorch": str(torch.__version__),
            "pytorch_cuda": torch.version.cuda,
            "triton": triton.__version__,
            "cuda_driver": _cuda_backend.driver_version(),
            "gpu": torch.cuda.get_device_name(device),
            "compute_capability": f"sm_{capability[0]}{capability[1]}",
            "multiprocessors": properties.multi_processor_count,
            "total_memory_bytes": properties.total_memory,
            "compiler": compiler,
        },
        "provenance": provenance,
        "methodology": _methodology(arguments, ticks),
        "compilation": {
            "status": "not-run",
            "reason": "segmented-sum suite not selected",
        },
        "results": [],
    }
    segmented = arguments.suite in {"all", "segmented-sum"}
    if segmented:
        from real_traces import load_real_trace

        cases = list(
            _segmented_cases(
                arguments.distributions,
                arguments.segment_count,
                arguments.seeds,
                generate_lengths,
                load_real_trace,
            )
        )
        kernels = _make_triton_kernels()
        result["compilation"] = _measure_segmented_compilation(
            torch,
            cases,
            kernels,
            only=arguments.candidates,
            exclude=arguments.exclude_candidates,
        )
    if arguments.suite in {"all", "vadd"}:
        result["results"].extend(_run_vadd(torch, measure, arguments.seeds[0]))
    if segmented:
        result["results"].extend(
            _run_segmented_sum(
                torch, measure, arguments, cases=cases, kernels=kernels
            )
        )
    benchmark_provenance.finish(provenance)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2) + "\n")
    validate_child(result)
    print(json.dumps({"output": str(arguments.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
