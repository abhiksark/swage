# benchmarks/benchmark_fresh_offsets.py
"""Measure segmented sum when every call sees a new offsets layout.

The frozen benchmarks prepare one layout and time repeated launches of it.
This harness times the other regime: every iteration takes an offsets layout
that no earlier iteration used, and the timed region runs from offsets in to
result out. Preparation is therefore inside the timed region on purpose.

It is a research harness, not a CI gate. Triton is imported only when the
benchmark runs; the project does not depend on Triton, and the Triton
candidates are left out when it is not installed.

Candidates are timed in a new random order every iteration, so what runs
before a sample changes from sample to sample, and what it leaves behind in
the host caches and in the device's idle state lands in the next sample.
Each sample is therefore preceded by untimed calls of the same candidate on
one layout that is never timed (--warm-calls). --candidates and
--exclude-candidates choose what is timed.

Run with PYTHONPATH=python:build/python_packages and --output result.json.
A full run requires a clean worktree. Use --smoke to check the harness on a
few tiny layouts; a smoke record is labelled as not evidence. The device is
the current CUDA device; select another with CUDA_VISIBLE_DEVICES.
"""

import argparse
import array
import hashlib
import importlib.util
import itertools
import json
import pathlib
import platform
import random
import subprocess
import sys
import time
import types
from datetime import datetime, timezone
from typing import NamedTuple

import benchmark_provenance
from benchmark_triton_comparison import (
    _PADDED_BYTES_PER_ELEMENT,
    _PLANNED_CTA_BLOCK,
    _PLANNED_WARPS,
    _QUANTUM,
    _add_candidate_filter,
    _check_modes,
    _check_selectors,
    _check_sums,
    _family,
    _free_device_bytes,
    _gb_per_s,
    _launch_triton_looped,
    _launch_triton_planned,
    _make_triton_kernels,
    _median_iqr,
    _padded_inputs,
    _padded_sum,
    _partition_tasks,
    _select,
    _sum_reference,
    _triton_looped_configs,
    _useful_bytes,
    _values,
    _wanted_skips,
)
from distributions import generate_lengths, summarize_lengths, worst_case_total

_SEGMENT_COUNT = 32_768
_SEED = 7
_WARMUPS = 5
_SAMPLES = 100
_SMOKE_SEGMENT_COUNT = 2_048
_SMOKE_WARMUPS = 1
_SMOKE_SAMPLES = 3
_WARM_CALLS = 2
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
_I32_MAX = (1 << 31) - 1
# Exact quarter multiples, or random values checked within a bound.
_VALUE_KINDS = ("quarters", "normal")


class _Layout(NamedTuple):
    """One host offsets layout and the seed that generated it.

    The lengths and offsets are four-byte integer arrays, so a pool of
    layouts with 10^6 segments each stays small on the host.
    """

    seed: int
    lengths: array.array
    offsets: array.array


class _DeviceLayout(NamedTuple):
    """One uploaded layout with its inputs and its float64 reference.

    `offsets` are int32. `long_offsets` are the same offsets as int64, the
    width PyTorch produces, for the candidate that times the public call on
    them.
    """

    values: object
    offsets: object
    reference: object
    tolerance: object
    long_offsets: object


def _arguments(argv=None):
    """Parse the output path, the smoke switch, and the sample counts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Run a few small layouts to check the harness. Ignores "
            "--samples and --warmups, allows a dirty worktree, and labels "
            "the record as not evidence."
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
    parser.add_argument(
        "--distributions",
        nargs="+",
        choices=_DISTRIBUTIONS,
        default=list(_DISTRIBUTIONS),
        metavar="NAME",
        help="Distributions to run, one row each. The default is all nine.",
    )
    parser.add_argument(
        "--segment-count",
        type=int,
        help=(
            f"Segments per layout. The default is {_SEGMENT_COUNT}, or "
            f"{_SMOKE_SEGMENT_COUNT} with --smoke."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=_SEED,
        help=(
            "Seed of the first layout, of the values, and of the candidate "
            "orders. Layout i of a row uses seed + i."
        ),
    )
    parser.add_argument(
        "--values",
        choices=_VALUE_KINDS,
        default="quarters",
        help=(
            "Seeded nonzero quarter multiples, whose sums are checked "
            "exactly, or seeded standard normal values, which are checked "
            "within the f32 any-order bound."
        ),
    )
    _add_candidate_filter(parser)
    parser.add_argument(
        "--warm-calls",
        type=int,
        default=_WARM_CALLS,
        help=(
            "Untimed calls of a candidate, on one layout that is never "
            "timed, before each of its samples. 0 times every candidate "
            "straight after the previous one."
        ),
    )
    arguments = parser.parse_args(argv)
    if arguments.samples < 2 or arguments.warmups < 1:
        parser.error("samples must be >= 2 and warmups must be >= 1")
    if arguments.warm_calls < 0:
        parser.error("warm-calls must not be negative")
    try:
        _check_selectors(
            [*(arguments.candidates or ()), *arguments.exclude_candidates],
            _candidate_names(triton_available=True),
        )
    except ValueError as error:
        parser.error(str(error))
    segment_count = _sizes(arguments)[0]
    if segment_count <= 0:
        parser.error("segment-count must be positive")
    for name in arguments.distributions:
        total = worst_case_total(name, segment_count)
        if total > _I32_MAX:
            parser.error(
                f"{name} with {segment_count} segments can reach {total} "
                "elements, which does not fit i32 offsets"
            )
    return arguments


def _sizes(arguments):
    """Return the segment count, warmups, and samples for this run."""
    segment_count = arguments.segment_count
    if arguments.smoke:
        if segment_count is None:
            segment_count = _SMOKE_SEGMENT_COUNT
        return segment_count, _SMOKE_WARMUPS, _SMOKE_SAMPLES
    if segment_count is None:
        segment_count = _SEGMENT_COUNT
    return segment_count, arguments.warmups, arguments.samples


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


def _same_checkout(root, package, *, smoke):
    """Return whether the imported swage package belongs to this checkout.

    The revision in the record describes the checkout that holds this
    script. A package imported from another checkout, for example through
    ``PYTHONPATH`` or an editable install, would be measured under that
    revision without belonging to it.

    Args:
        root: Root of the checkout that holds the benchmark script.
        package: Directory of the imported ``swage`` package.
        smoke: Whether this is a smoke run, which records the mismatch.

    Returns:
        True when the package is inside the checkout.

    Raises:
        RuntimeError: If a full run imported the package from elsewhere.
    """
    root = root.resolve()
    package = package.resolve()
    inside = package.is_relative_to(root)
    if not inside and not smoke:
        raise RuntimeError(
            "fresh-offsets benchmark must measure its own checkout: the "
            f"imported swage package is {package}, but the benchmark runs "
            f"from {root}"
        )
    return inside


def _native_library_paths(native, locations):
    """Resolve the library files that the runtime's native identity names.

    Args:
        native: The ``native`` field of the runtime compiler identity, a
            list of ``[file name, size, mtime]``, or None.
        locations: Search locations of ``mlir_swage._mlir_libs``.

    Returns:
        One sorted path per named file that exists. The directory is
        resolved, so a build tree reached through a symlink is reported
        where it really is; the file name is kept as the identity lists it.
    """
    names = [name for name, *_ in native or ()]
    return sorted(
        str(path)
        for location in locations
        for name in names
        if (path := pathlib.Path(location).resolve() / name).is_file()
    )


def _imported_code(
    *,
    root,
    package,
    same_checkout,
    identity,
    native_extension,
    native_library_paths,
    mlir_swage_locations,
    llvm_linked,
):
    """Return which package and native build the run actually measured.

    Args:
        root: Resolved root of the checkout that holds this script.
        package: Resolved directory of the imported ``swage`` package.
        same_checkout: Whether that package is inside ``root``.
        identity: ``swage._runtime._compiler_identity()``, stored as
            returned. Its ``frontend`` digests the package sources and its
            ``native`` lists the native libraries with size and mtime.
        native_extension: File of the loaded native extension module.
        native_library_paths: Resolved paths of the ``native`` files.
        mlir_swage_locations: Search locations of ``mlir_swage``.
        llvm_linked: LLVM version the native extension was linked against.

    Returns:
        The imported-code block of the record.
    """
    return {
        "benchmark_checkout": str(root),
        "swage_package": str(package),
        "swage_package_in_checkout": same_checkout,
        "compiler_identity": identity,
        "native_extension": str(native_extension),
        "native_library_paths": list(native_library_paths),
        "native_libraries_in_checkout": bool(native_library_paths)
        and all(
            pathlib.Path(path).is_relative_to(root)
            for path in native_library_paths
        ),
        "mlir_swage_locations": list(mlir_swage_locations),
        "llvm_linked": llvm_linked,
    }


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
    """Return the names of every candidate, in a fixed order.

    Args:
        triton_available: Whether Triton is installed.

    Returns:
        The candidate names: the Swage candidates, the PyTorch baselines,
        and with Triton the looped sweep, the planned scheduler with its
        one-block tasks, and the planned scheduler with looping tasks.
    """
    names = [
        "swage_mixed",
        "swage_cta_call",
        "swage_public_call",
        "swage_public_call_int64",
        "torch",
        "torch_pad_to_max",
    ]
    if triton_available:
        looped = _triton_looped_configs()
        names.extend(f"triton_looped_b{b}_w{w}" for b, w in looped)
        names.extend(f"triton_planned_w{w}" for w in _PLANNED_WARPS)
        names.extend(f"triton_planned_looped_b{b}_w{w}" for b, w in looped)
    return tuple(names)


def _require_available(candidates, triton_available):
    """Refuse a run that names a candidate it cannot time.

    Args:
        candidates: The --candidates selectors, or None without the option.
        triton_available: Whether Triton is installed.

    Raises:
        RuntimeError: If a selector names only Triton candidates and Triton
            is not installed. Without the option the Triton candidates are
            left out and the record says so; a candidate that was asked for
            by name is not dropped silently.
    """
    if candidates is None or triton_available:
        return
    available = _candidate_names(triton_available=False)
    missing = [
        selector
        for selector in candidates
        if not _select(available, [selector], ())
    ]
    if missing:
        raise RuntimeError(
            f"--candidates names {', '.join(missing)}, but Triton is not "
            "installed"
        )


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
        lengths = array.array("i", generate_lengths(name, count, layout_seed))
        offsets = array.array("i", [0, *itertools.accumulate(lengths)])
        pool.append(_Layout(layout_seed, lengths, offsets))
    digests = {
        hashlib.sha256(layout.offsets.tobytes()).digest() for layout in pool
    }
    if len(digests) != size:
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
    warm=None,
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
        warm: Callable taking a candidate name, run before each sample of
            that candidate and outside its timer, or None. It is the same
            step for every candidate and every iteration.

    Returns:
        One entry per iteration with its timed flag, the candidate order it
        ran, and the microsecond sample of each candidate in that order.
    """
    iterations = []
    for index, (layout, order) in enumerate(zip(layouts, orders, strict=True)):
        samples = {}
        for name in order:
            if warm is not None:
                warm(name)
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


def _configuration(
    *,
    segment_count,
    warmups,
    samples,
    triton_available,
    distributions=_DISTRIBUTIONS,
    seed=_SEED,
    values="quarters",
    clock_tick_us=None,
    candidates=None,
    exclude_candidates=(),
    warm_calls=_WARM_CALLS,
):
    """Return the benchmark contract, including what the timer covers."""
    timed_region = {
        "swage_mixed": (
            "_prepare_planned_sum (offset validation and transfer to the "
            "host, classification, whatever kernel compilation and module "
            "loading the preparation path performs at this revision, task "
            "upload), then the mixed launch and synchronize"
        ),
        "swage_cta_call": (
            "one launch_gpu call (offset validation and transfer to the "
            "host, then one launch of the pure CTA kernel, which walks "
            "each segment in 128-thread blocks), then synchronize; no "
            "classification and no task upload"
        ),
        "swage_public_call": (
            "one swage.segment_reduce call with out= (the argument checks, "
            "offset validation and transfer to the host, classification, "
            "schedule selection, whatever kernel compilation and module "
            "loading the call performs at this revision, task upload, and "
            "the launches of the selected schedule), then synchronize"
        ),
        "swage_public_call_int64": (
            "swage_public_call with int64 offsets, which are uploaded with "
            "the layout and outside the timer: the same call, with the "
            "validation of the int64 host copy, its narrowing to int32, and "
            "the upload of the private int32 copy the kernels read"
        ),
        "torch": (
            "torch.segment_reduce on the device offsets with its output "
            "allocation, then synchronize"
        ),
        "torch_pad_to_max": (
            "pure PyTorch: reading the longest length from the device "
            "offsets, padding every segment with zeros to it, the masked "
            "row sum, their allocations, then synchronize"
        ),
    }
    if triton_available:
        timed_region["triton_looped"] = (
            "one launch of the looped kernel, then synchronize; it needs "
            "no host classification"
        )
        partition = (
            "the partition (two torch.nonzero calls over the segment "
            "lengths on the device, each waiting for the device, and the "
            "conversion of the ids to int32), then the packed launch of "
            "the short tasks, "
        )
        timed_region["triton_planned"] = (
            f"{partition}one launch that reads one block of "
            f"{_PLANNED_CTA_BLOCK} elements per longer task, then "
            "synchronize"
        )
        timed_region["triton_planned_looped"] = (
            f"{partition}one launch whose longer tasks loop over their "
            "segment in fixed blocks, then synchronize"
        )
    if warm_calls:
        warm_step = (
            f"before each sample, {warm_calls} untimed calls of the same "
            "candidate on the row's warm layout, each followed by a "
            "synchronize; the warm layout has its own seed and its own "
            "output buffer and is never timed or checked"
        )
    else:
        warm_step = (
            "none; a sample starts straight after the check of the "
            "previous candidate"
        )
    return {
        "distributions": list(distributions),
        "segment_count": segment_count,
        "seed": seed,
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
        "candidates": _select(
            _candidate_names(triton_available=triton_available),
            candidates,
            exclude_candidates,
        ),
        "candidate_filter": {
            "candidates": candidates,
            "exclude_candidates": list(exclude_candidates),
            "note": (
                "a name selects one candidate or a family; each row lists "
                "the candidates it timed under candidates, what the filter "
                "left out under excluded, and what it could not run under "
                "skipped"
            ),
        },
        "candidate_order": (
            "a new random permutation every iteration, shuffled by "
            "random.Random(order_seed) with order_seed "
            "'<seed>:<distribution>:<iteration>'; the seed and the order "
            "are recorded with each iteration"
        ),
        "warm_step": {
            "calls": warm_calls,
            "step": warm_step,
            "reason": (
                "the candidate that ran before a sample leaves the host "
                "caches and the device in a state that the next sample "
                "pays for; the step is the same for every candidate, so "
                "every sample follows a call of its own candidate"
            ),
        },
        "warp_max_elements": _WARP_MAX_ELEMENTS,
        "swage_policy": (
            "swage_mixed times _prepare_planned_sum with "
            "select_schedule=False. The private runner has no mixed-only "
            "preparation: one call returns the warp, CTA, and mixed "
            "policies, and only mixed is launched. The two unused policies "
            "cost their kernel and module memo lookups and the memoized "
            "identity task ids. swage_cta_call times the one private call "
            "that validates and launches a single policy; it does not "
            "classify. swage_public_call times the public "
            "swage.segment_reduce, the call a user makes: it selects the "
            "schedule automatically, which swage_mixed disables, so the two "
            "can run different kernels on one layout, and it has no "
            "separate preparation sample. swage_public_call_int64 is the "
            "same call on the int64 form of the same offsets"
        ),
        "triton_looped": (
            "every block and warp configuration is timed; none is selected"
            if triton_available
            else "skipped: Triton is not installed"
        ),
        "triton_planned": (
            "the packed-warp and one-block task kernels of the comparison "
            "harness, with the partition of every fresh layout inside the "
            "timed call; a row whose longest segment exceeds "
            f"{_PLANNED_CTA_BLOCK} elements skips it. "
            "triton_planned_looped packs the short tasks the same way and "
            "loops over the longer ones, with the block and warp sweep of "
            "triton_looped. triton_planned_partition_samples_us is the "
            "part of each sample spent in the partition"
            if triton_available
            else "skipped: Triton is not installed"
        ),
        "pad_to_max": (
            "timed on a row only when padding every layout of the row to "
            "its longest segment fits the free device memory at "
            f"{_PADDED_BYTES_PER_ELEMENT} bytes per padded element; "
            "otherwise the row lists it under skipped with the bytes it "
            "would need"
        ),
        "values_kind": values,
        "values": (
            "CPU seeded randint(1, 8) / 4, float32, never zero"
            if values == "quarters"
            else "CPU seeded standard normal, float32"
        )
        + "; one buffer per distribution, each layout reads its prefix",
        "correctness": (
            "every candidate in every iteration, warmups included, against "
            "a float64 CPU torch.segment_reduce reference: exactly where "
            "sums of the values are exact in f32 in any order, otherwise "
            "within gamma(n - 1) times the sum of magnitudes; each row "
            "counts its exact, bounded, and unchecked segments"
        ),
        "clock": "time.perf_counter_ns between two device synchronizations",
        "timer": {
            "tick_us": clock_tick_us,
            "tick": (
                "smallest advance of back-to-back time.perf_counter_ns "
                "reads in this process; each row reports it as a fraction "
                "of the median sample of every candidate"
            ),
            "batching": (
                "none; a sample is one call on one fresh layout, because "
                "a second call on the same layout would not be fresh"
                + (
                    f"; it follows {warm_calls} untimed calls of the same "
                    "candidate on another layout"
                    if warm_calls
                    else ""
                )
            ),
        },
        "effective_gb_per_s": (
            "per timed sample, the bytes a correct sum has to move on that "
            "layout (f32 values and i32 offsets read, f32 sums written) "
            "divided by the sample time; rows report the median and "
            "quartiles"
        ),
        "timed_region": timed_region,
        "swage_mixed_prepare_samples_us": (
            "the part of each swage_mixed sample spent inside "
            "_prepare_planned_sum"
        ),
        "excluded": [
            "layout generation",
            "offsets and values upload",
            "reference computation and the correctness check",
            "output allocation for the Swage and Triton candidates",
            "CUDA, native compiler, and Triton initialization (warmups)",
            "the warm step",
        ],
    }


def _require_identified_code(source, imported_code, provenance=None):
    """Refuse a full record whose measured code is not pinned down.

    The provenance block is known only after the run, so the check before
    the run passes None and the check on the finished record passes it.
    """
    if not source["worktree_clean"]:
        raise ValueError("a full record requires a clean source worktree")
    if not imported_code["swage_package_in_checkout"]:
        raise ValueError(
            "a full record requires the swage package of its own checkout: "
            f"imported {imported_code['swage_package']}, benchmark in "
            f"{imported_code['benchmark_checkout']}"
        )
    identity = imported_code["compiler_identity"]
    missing = [
        field for field in ("frontend", "native") if identity.get(field) is None
    ]
    if not imported_code["native_library_paths"]:
        missing.append("native_library_paths")
    if imported_code["llvm_linked"] is None:
        missing.append("llvm_linked")
    if provenance is not None:
        missing.extend(
            field
            for field in ("native_sha256", "loaded_ptx")
            if not provenance.get(field)
        )
    if missing:
        raise ValueError(
            "a full record must identify the code it measured; missing: "
            f"{', '.join(missing)}"
        )


def _record(
    *,
    source,
    environment,
    configuration,
    results,
    imported_code,
    provenance,
    smoke,
):
    """Assemble the JSON record, refusing one that cannot be attributed.

    Args:
        source: Revision and worktree state from ``_git_metadata``.
        environment: Machine identity from ``_environment``.
        configuration: Benchmark contract from ``_configuration``.
        results: One row per distribution.
        imported_code: Measured package and build from ``_imported_code``.
        provenance: Finished block from ``benchmark_provenance``: the
            hashes of the native libraries and of every loaded PTX module,
            the CPU, and the GPU state before and after the run.
        smoke: Whether this was a smoke run.

    Returns:
        The complete record.

    Raises:
        ValueError: If provenance is missing, or if a run that is not a
            smoke run has a dirty worktree, measured a package from another
            checkout, or cannot identify the code it measured.
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
    if not smoke:
        _require_identified_code(source, imported_code, provenance)
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
        "imported_code": imported_code,
        "provenance": provenance,
        "environment": environment,
        "configuration": configuration,
        "results": results,
    }


def _upload(torch, pool, device, values_kind, seed):
    """Upload one pool and compute each layout's float64 reference.

    Args:
        torch: The PyTorch module.
        pool: Layouts from ``_layout_pool``.
        device: Device that holds the inputs.
        values_kind: ``quarters`` or ``normal``.
        seed: Seed of the values.

    Returns:
        One ``_DeviceLayout`` per layout. Every layout reads a prefix of
        one values buffer.
    """
    largest_total = max(layout.offsets[-1] for layout in pool)
    host_values = _values(torch, values_kind, largest_total, seed)
    doubles = host_values.double()
    values = host_values.to(device)
    uploaded = []
    for layout in pool:
        total = layout.offsets[-1]
        host_offsets = torch.frombuffer(
            layout.offsets, dtype=torch.int32
        ).clone()
        reference, tolerance = _sum_reference(
            torch, doubles[:total], host_offsets, _QUANTUM[values_kind]
        )
        uploaded.append(
            _DeviceLayout(
                values[:total],
                host_offsets.to(device),
                reference.to(device),
                tolerance.to(device),
                host_offsets.long().to(device),
            )
        )
    return uploaded


def _candidates(
    torch,
    swage,
    kernels,
    names,
    segment_count,
    output_of,
    prepare_us,
    partition_us,
):
    """Return the named candidates as callables that take a layout.

    Args:
        torch: The PyTorch module.
        swage: The Swage entry points: the private ``prepare`` and
            ``launch``, and ``public``, which is ``swage.segment_reduce``.
        kernels: The Triton kernels, or None without Triton.
        names: Names of the candidates to build.
        segment_count: Segments per layout.
        output_of: Callable returning the output buffer of a named
            candidate. The timed candidates and the warm candidates are
            built with different buffers.
        prepare_us: List that receives the microseconds each
            ``swage_mixed`` call spends in its preparation.
        partition_us: Mapping that receives, per planned Triton candidate,
            the microseconds each call spends in its partition.

    Returns:
        The candidates by name, in the order of ``names``.
    """

    def swage_mixed():
        output = output_of("swage_mixed")

        def call(layout):
            start = time.perf_counter_ns()
            prepared = swage.prepare(
                layout.values,
                layout.offsets,
                output,
                warp_max_elements=_WARP_MAX_ELEMENTS,
            )
            prepare_us.append((time.perf_counter_ns() - start) / 1_000.0)
            prepared.mixed()
            return output

        return call

    def swage_cta_call():
        output = output_of("swage_cta_call")

        def call(layout):
            swage.launch(layout.values, layout.offsets, output, "sum")
            return output

        return call

    def swage_public_call():
        output = output_of("swage_public_call")

        def call(layout):
            # The buffer is the caller's, as for the other Swage
            # candidates, so the call allocates no result.
            return swage.public(
                layout.values, layout.offsets, "sum", out=output
            )

        return call

    def swage_public_call_int64():
        output = output_of("swage_public_call_int64")

        def call(layout):
            return swage.public(
                layout.values, layout.long_offsets, "sum", out=output
            )

        return call

    def torch_reduce():
        return lambda layout: torch.segment_reduce(
            layout.values, "sum", offsets=layout.offsets
        )

    def torch_pad_to_max():
        return lambda layout: _padded_sum(
            *_padded_inputs(torch, layout.values, layout.offsets)
        )

    def looped(candidate, block, warps):
        output = output_of(candidate)
        return lambda layout: _launch_triton_looped(
            kernels.looped,
            layout.values,
            layout.offsets,
            output,
            segment_count,
            block,
            warps,
        )

    def planned(candidate, warps, block=None):
        output = output_of(candidate)
        series = partition_us.setdefault(candidate, [])

        def call(layout):
            # A fresh layout has no task lists yet, so the partition is
            # part of the call, as the Swage preparation is.
            start = time.perf_counter_ns()
            warp_ids, cta_ids = _partition_tasks(torch, layout.offsets)
            series.append((time.perf_counter_ns() - start) / 1_000.0)
            return _launch_triton_planned(
                kernels,
                layout.values,
                layout.offsets,
                output,
                warp_ids,
                cta_ids,
                warps=warps,
                block=block,
            )

        return call

    builders = {
        "swage_mixed": swage_mixed,
        "swage_cta_call": swage_cta_call,
        "swage_public_call": swage_public_call,
        "swage_public_call_int64": swage_public_call_int64,
        "torch": torch_reduce,
        "torch_pad_to_max": torch_pad_to_max,
    }
    for block, warps in _triton_looped_configs():
        candidate = f"triton_looped_b{block}_w{warps}"
        builders[candidate] = (
            lambda candidate=candidate, block=block, warps=warps:
            looped(candidate, block, warps)
        )
        candidate = f"triton_planned_looped_b{block}_w{warps}"
        builders[candidate] = (
            lambda candidate=candidate, block=block, warps=warps:
            planned(candidate, warps, block)
        )
    for warps in _PLANNED_WARPS:
        candidate = f"triton_planned_w{warps}"
        builders[candidate] = (
            lambda candidate=candidate, warps=warps: planned(candidate, warps)
        )
    return {candidate: builders[candidate]() for candidate in names}


def _run_distribution(
    torch,
    swage,
    kernels,
    name,
    segment_count,
    warmups,
    samples,
    *,
    device,
    synchronize,
    free_bytes,
    clock_tick_us,
    seed=_SEED,
    values_kind="quarters",
    only=None,
    exclude=(),
    warm_calls=_WARM_CALLS,
):
    """Measure the selected candidates on one distribution's fresh layouts.

    Args:
        torch: The PyTorch module.
        swage: The private Swage entry points: ``prepare``, the planned-sum
            preparation, and ``launch``, the single-policy call.
        kernels: The Triton kernels ``looped``, ``packed``, ``cta``, and
            ``cta_looped``, or None without Triton.
        name: Distribution name.
        segment_count: Segments per layout.
        warmups: Leading iterations that are checked but not recorded.
        samples: Timed iterations.
        device: Device that holds the inputs and outputs.
        synchronize: Callable that waits for all work on that device.
        free_bytes: Callable returning the bytes the pad-to-max candidate
            may use; it is called once the layouts are on the device.
        clock_tick_us: Measured tick of the sample clock, or None.
        seed: Seed of the first layout, the values, and the orders.
        values_kind: ``quarters`` or ``normal``.
        only: Candidate filter selectors to keep, or None for all.
        exclude: Candidate filter selectors to leave out.
        warm_calls: Untimed calls of a candidate on the row's warm layout
            before each of its samples.

    Returns:
        The record row for this distribution. ``candidates`` lists what was
        timed, ``excluded`` what the filter left out, and ``skipped`` what
        the row cannot run, with the reason.

    Raises:
        ValueError: If the filter leaves no candidate the row can run.
    """
    timed = warmups + samples
    # With a warm step the pool holds one more layout, which is distinct
    # from every timed layout like any other and is never timed.
    pool = _layout_pool(name, segment_count, timed + bool(warm_calls), seed)
    uploaded = _upload(torch, pool, device, values_kind, seed)
    layouts = uploaded[:timed]
    longest = max(max(layout.lengths) for layout in pool)
    padded_bytes = segment_count * longest * _PADDED_BYTES_PER_ELEMENT
    budget = free_bytes()
    unable = {}
    if padded_bytes > budget:
        unable["torch_pad_to_max"] = (
            f"padding {segment_count} segments to {longest} elements needs "
            f"{padded_bytes} bytes and {budget} are free"
        )
    if kernels is not None and longest > _PLANNED_CTA_BLOCK:
        unable["triton_planned"] = (
            f"its task kernel reads one block of {_PLANNED_CTA_BLOCK} "
            f"elements and the longest segment of the row has {longest}"
        )
    # Every candidate is timed, excluded by the filter, or skipped because
    # the filter keeps it and the row cannot run it.
    universe = _candidate_names(triton_available=kernels is not None)
    wanted = _select(universe, only, exclude)
    skipped = _wanted_skips(unable, wanted)
    names = [name for name in wanted if _family(name) not in unable]
    if not names:
        raise ValueError(
            "the candidate filter leaves no candidate that can run on "
            f"{name} with {segment_count} segments"
        )
    # A result that was never written must not pass the correctness check.
    outputs = {
        candidate: torch.full((segment_count,), float("nan"), device=device)
        for candidate in names
        if not candidate.startswith("torch")
    }
    prepare_us = []
    partition_us = {}
    candidates = _candidates(
        torch,
        swage,
        kernels,
        names,
        segment_count,
        outputs.__getitem__,
        prepare_us,
        partition_us,
    )
    warm = None
    if warm_calls:
        warm_layout = uploaded[-1]
        # The warm calls write elsewhere, so a timed call that writes
        # nothing still leaves its own output unwritten.
        warm_output = torch.empty(segment_count, device=device)
        warm_candidates = _candidates(
            torch,
            swage,
            kernels,
            names,
            segment_count,
            lambda candidate: warm_output,
            [],
            {},
        )

        def warm(candidate):
            for _ in range(warm_calls):
                warm_candidates[candidate](warm_layout)
                synchronize()

    def check(candidate, result, layout):
        _check_sums(
            torch,
            f"{candidate} on {name}",
            result,
            layout.reference,
            layout.tolerance,
        )
        if candidate in outputs:
            outputs[candidate].fill_(float("nan"))

    orders = [
        _candidate_order(names, seed, name, index) for index in range(timed)
    ]
    iterations = _measure(
        layouts,
        [order for _, order in orders],
        candidates,
        check,
        warmups=warmups,
        synchronize=synchronize,
        warm=warm,
    )
    useful_bytes = [
        _useful_bytes(layout.offsets[-1], segment_count)
        for layout in pool[:timed]
    ]
    raw = _timed_samples(iterations, names)
    summary = {
        candidate: _median_iqr(timings) for candidate, timings in raw.items()
    }
    modes = [_check_modes(layout.tolerance) for layout in layouts]
    return {
        "distribution": name,
        "segment_count": segment_count,
        "seed": seed,
        "values": values_kind,
        "candidates": list(names),
        "excluded": [
            candidate for candidate in universe if candidate not in wanted
        ],
        "skipped": skipped,
        "warm_layout_seed": pool[-1].seed if warm_calls else None,
        "pad_to_max": {
            "longest_segment": longest,
            "padded_bytes": padded_bytes,
            "free_bytes": budget,
            "timed": "torch_pad_to_max" in names,
        },
        "check": {
            mode: sum(counts[mode] for counts in modes) for mode in modes[0]
        },
        "iterations": [
            {
                "layout_seed": layout.seed,
                "layout_statistics": summarize_lengths(layout.lengths),
                "useful_bytes": layout_bytes,
                "order_seed": order_seed,
                **iteration,
            }
            for layout, layout_bytes, (order_seed, _), iteration in zip(
                pool[:timed], useful_bytes, orders, iterations, strict=True
            )
        ],
        "raw_samples_us": raw,
        "summary_us": summary,
        "effective_gb_per_s": {
            candidate: _median_iqr(
                [
                    _gb_per_s(layout_bytes, sample)
                    for layout_bytes, sample in zip(
                        useful_bytes[warmups:], timings, strict=True
                    )
                ]
            )
            for candidate, timings in raw.items()
        },
        "tick_fraction_of_sample": {
            candidate: (
                None
                if clock_tick_us is None
                else clock_tick_us / timing["median"]
            )
            for candidate, timing in summary.items()
        },
        "swage_mixed_prepare_samples_us": prepare_us[warmups:],
        "triton_planned_partition_samples_us": {
            candidate: series[warmups:]
            for candidate, series in partition_us.items()
        },
        "correctness_passed": True,
    }


def _headline(summary):
    """Return the medians worth printing; the record keeps every sample."""
    medians = {
        candidate: round(timing["median"], 1)
        for candidate, timing in summary.items()
    }
    headline = {
        name: median for name, median in medians.items() if "triton" not in name
    }
    for family in ("triton_looped", "triton_planned", "triton_planned_looped"):
        swept = {
            candidate: median
            for candidate, median in medians.items()
            if _family(candidate) == family
        }
        if swept:
            fastest = min(swept, key=swept.get)
            headline[f"fastest {fastest}"] = swept[fastest]
    return headline


def main():
    """Run the fresh-offsets benchmark and write its JSON record."""
    arguments = _arguments()
    root = pathlib.Path(__file__).resolve().parents[1]
    _check_output(root, arguments.output, smoke=arguments.smoke)
    source = _git_metadata(root, allow_dirty=arguments.smoke)
    segment_count, warmups, samples = _sizes(arguments)

    import swage

    package = pathlib.Path(swage.__file__).resolve().parent
    same_checkout = _same_checkout(root, package, smoke=arguments.smoke)
    if not same_checkout:
        print(
            f"smoke: the imported swage package {package} is outside the "
            f"benchmark checkout {root}; recorded, not refused",
            flush=True,
        )

    import torch
    from mlir_swage._mlir_libs import _swageDialectsNanobind as native_extension
    from swage import _cuda_backend, _runtime
    from swage._segmented_qualification import (
        _prepare_planned_sum,
        launch_gpu,
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "fresh-offsets benchmark requires CUDA-enabled PyTorch"
        )
    torch.ones(1, device="cuda").sum().item()
    provenance = benchmark_provenance.start(
        torch, benchmark_provenance.swage_build()
    )
    clock_tick_us = benchmark_provenance.clock_tick_us()
    identity = _runtime._compiler_identity()
    imported_code = _imported_code(
        root=root,
        package=package,
        same_checkout=same_checkout,
        identity=identity,
        native_extension=pathlib.Path(native_extension.__file__).resolve(),
        native_library_paths=_native_library_paths(
            identity.get("native"),
            importlib.util.find_spec(
                "mlir_swage._mlir_libs"
            ).submodule_search_locations,
        ),
        mlir_swage_locations=importlib.util.find_spec(
            "mlir_swage"
        ).submodule_search_locations,
        llvm_linked=getattr(native_extension.swage, "__llvm_version__", None),
    )
    if not arguments.smoke:
        _require_identified_code(source, imported_code)
    triton = _optional_triton()
    environment = _environment(
        torch,
        cuda_driver=_cuda_backend.driver_version(),
        nvidia_driver=_nvidia_driver(),
        triton_version=triton.__version__ if triton else None,
    )
    _require_available(arguments.candidates, triton is not None)
    kernels = _make_triton_kernels() if triton else None
    if triton is None:
        print(
            "Triton is not installed; leaving out the Triton candidates",
            flush=True,
        )
    runner = types.SimpleNamespace(
        prepare=_prepare_planned_sum,
        launch=launch_gpu,
        public=swage.segment_reduce,
    )

    results = []
    for name in arguments.distributions:
        row = _run_distribution(
            torch,
            runner,
            kernels,
            name,
            segment_count,
            warmups,
            samples,
            device="cuda",
            synchronize=torch.cuda.synchronize,
            free_bytes=lambda: _free_device_bytes(torch),
            clock_tick_us=clock_tick_us,
            seed=arguments.seed,
            values_kind=arguments.values,
            only=arguments.candidates,
            exclude=arguments.exclude_candidates,
            warm_calls=arguments.warm_calls,
        )
        results.append(row)
        print(f"{name}: median_us={_headline(row['summary_us'])}", flush=True)
        for candidate, reason in row["skipped"].items():
            print(f"{name}: {candidate} not timed: {reason}", flush=True)

    record = _record(
        source=source,
        environment=environment,
        configuration=_configuration(
            segment_count=segment_count,
            warmups=warmups,
            samples=samples,
            triton_available=triton is not None,
            distributions=arguments.distributions,
            seed=arguments.seed,
            values=arguments.values,
            clock_tick_us=clock_tick_us,
            candidates=arguments.candidates,
            exclude_candidates=arguments.exclude_candidates,
            warm_calls=arguments.warm_calls,
        ),
        results=results,
        imported_code=imported_code,
        provenance=benchmark_provenance.finish(provenance),
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
