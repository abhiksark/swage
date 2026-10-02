# python/tests/mlir/test_segmented_numerics.py
"""Numerical evidence for private segmented sums and ragged softmax.

Four claims are checked here, and docs/internals/segmented-reductions.md and
docs/internals/ragged-softmax.md state them:

- A schedule reproduces its own f32 sum bit for bit, across launches,
  across a fresh preparation, and in a second process.
- Different schedules round differently. Which schedule a segment gets
  depends on its length, on the planning limits, and, unless the caller
  pins it, on the batch and the SM count of the device.
- Every schedule stays within k * eps32 * sum(|x|) of the exact sum, where
  k is the depth of that schedule's reduction tree, and propagates NaN,
  infinities, and subnormal values the way IEEE-754 addition does.
- Ragged softmax stays within a relative error that grows linearly with
  the distance of a logit from its segment maximum.

This file is also the program the second interpreter runs: executed as a
script it prints the result bits of every sum schedule as JSON.
"""

import importlib.util
import json
import math
import os
import pathlib
import re
import subprocess
import sys
from itertools import pairwise

import pytest
import swage
import torch
from mlir_swage import ir
from mlir_swage._mlir_libs._swageDialectsNanobind import swage as native_swage
from mlir_swage.dialects import swage as swage_dialect
from reduction_programs import reduction_module
from swage._segmented_qualification import (
    _SOFTMAX_MODULE,
    _element_of,
    _prepare_persistent_sum,
    _prepare_planned_reduction,
    _reduction_kernel,
    launch_gpu,
    launch_softmax_gpu,
)
from test_segmented_runtime import (
    _EPS32,
    _EPS64,
    _PLANNING_LIMITS,
    _bits,
    _offsets,
    _summation_depth,
)

_needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"
)

# The four static schedules. "split" is the mixed callable of a preparation
# whose limits send every segment longer than 16 elements through the
# partial and merge kernels (_PLANNING_LIMITS).
STATIC_POLICIES = ["warp", "cta", "mixed", "split"]

# The static schedules, the unprepared one-CTA launch, and the experimental
# persistent kernel: every path that produces an f32 sum on the device.
SUM_PATHS = [*STATIC_POLICIES, "one-cta", "persistent"]


def _sum_launch(path, values, offsets, output, kind="sum"):
    """Prepare one path with a pinned schedule and return its launch.

    The persistent path reduces an f32 sum only. Every other path takes
    `kind` and runs the program of the dtype of `values`.
    """
    if path == "one-cta":
        return lambda: launch_gpu(values, offsets, output, kind)
    if path == "persistent":
        return _prepare_persistent_sum(values, offsets, output).launch
    element = _element_of(torch, values)
    warp_max, chunk = _PLANNING_LIMITS["split" if path == "split" else "mixed"]
    prepared = _prepare_planned_reduction(
        values,
        offsets,
        output,
        module_text=reduction_module(kind, "identity", element),
        kernel_name=_reduction_kernel(kind, element),
        warp_max_elements=warp_max,
        cta_chunk_elements=chunk,
        select_schedule=False,
    )
    return getattr(prepared, "mixed" if path == "split" else path)


def _run_sum(path, host_values, host_offsets, launches=1, kind="sum"):
    """Launch one path on fresh device tensors and return each result."""
    values, offsets = host_values.cuda(), host_offsets.cuda()
    output = torch.empty(
        host_offsets.numel() - 1, dtype=host_values.dtype, device="cuda"
    )
    launch = _sum_launch(path, values, offsets, output, kind)
    results = []
    for _ in range(launches):
        output.fill_(float("nan"))
        launch()
        results.append(output.cpu())
    return results


# Lengths on both sides of the warp limit (32) and the chunk limit (4096),
# and one segment of seventeen chunks.
ORDER_LENGTHS = [0, 1, 31, 32, 33, 4095, 4096, 4097, 8193, 65537]


def _order_sensitive_case():
    """Seeded normal draws whose f32 sum depends on the order of addition.

    test_sum_bits_depend_on_the_schedule shows the dependence on the
    device: two schedules disagree on these values. A repeatability check
    on them can therefore fail, which one on exactly summable values
    cannot.
    """
    generator = torch.Generator().manual_seed(0)
    values = torch.randn(sum(ORDER_LENGTHS), generator=generator)
    return values, torch.tensor(_offsets(ORDER_LENGTHS), dtype=torch.int32)


def _order_sensitive_bits():
    """Return the result bits of every sum path on the order-sensitive case."""
    values, offsets = _order_sensitive_case()
    return {
        path: _bits(_run_sum(path, values, offsets)[0]) for path in SUM_PATHS
    }


@_needs_cuda
@pytest.mark.parametrize("path", SUM_PATHS)
def test_sum_bits_repeat_across_launches_and_preparations(path):
    """One schedule returns the same f32 bits every time it runs.

    Three launches of one preparation must agree bit for bit, and so must
    a second preparation on copies of the inputs with its own output,
    which gives the kernel different pointers and new task storage.
    """
    values, offsets = _order_sensitive_case()

    first = _run_sum(path, values, offsets, launches=3)
    second = _run_sum(path, values.clone(), offsets.clone())

    assert not first[0].isnan().any()
    for result in (*first[1:], *second):
        assert _bits(result) == _bits(first[0])


def _second_process_bits():
    """Run this file as a program in a fresh interpreter.

    The interpreter imports the same `swage` and `mlir_swage` packages as
    this process and inherits its kernel cache directory. It compiles and
    loads every kernel again in its own CUDA context.

    Returns:
        The result bits per sum path that the interpreter printed.
    """
    bindings = importlib.util.find_spec("mlir_swage._mlir_libs")
    bindings_parent = pathlib.Path(
        list(bindings.submodule_search_locations)[0]
    ).parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(pathlib.Path(swage.__file__).parents[1]), str(bindings_parent)]
    )
    completed = subprocess.run(
        [sys.executable, __file__],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.splitlines()[-1])


@_needs_cuda
def test_sum_bits_repeat_in_a_second_process():
    """A fresh process reproduces the bits of every sum path.

    The second interpreter shares no compiled kernel, loaded module, or
    CUDA context with this one, so equal bits mean the result is a
    function of the values and the schedule, not of process state.
    """
    assert _second_process_bits() == _order_sensitive_bits()


@_needs_cuda
def test_sum_bits_depend_on_the_schedule():
    """Pin which schedules share a reduction tree and which do not.

    The statements in docs/internals/segmented-reductions.md, on the
    order-sensitive case:

    - A segment of at most 32 elements has the same bits on every path
      here: each lane holds at most one element, so the warp and 128-lane
      trees add the same pairs, and the 16-element chunks of the split
      path are subtrees of that pairing.
    - The mixed schedule uses the warp tree up to the warp limit and the
      128-lane tree up to the chunk limit. The unprepared one-CTA launch
      uses the 128-lane tree at every length.
    - Above 32 elements the warp and 128-lane trees disagree. Above the
      chunk limit the 128-lane tree and the split tree disagree, and so do
      split trees with different chunk limits.
    - The persistent kernel uses the mixed split tree above the chunk
      limit. From 33 elements to the chunk limit it uses 512 lanes, which
      none of the prepared static schedules does.

    A disagreement is asserted on at least one segment, not on each: two
    trees can round one segment to the same f32 by chance. If a later
    change makes two schedules agree everywhere, this test and that page
    change together.
    """
    bits = _order_sensitive_bits()
    lengths = torch.tensor(ORDER_LENGTHS)

    def differing(first, second, mask):
        return sum(
            left != right
            for left, right, selected in zip(
                bits[first], bits[second], mask.tolist()
            )
            if selected
        )

    one_warp = lengths <= 32
    one_chunk = (lengths > 32) & (lengths <= 4096)
    chunked = lengths > 4096
    for path in SUM_PATHS:
        assert differing(path, "warp", one_warp) == 0
    assert differing("mixed", "cta", one_chunk) == 0
    assert differing("one-cta", "cta", lengths >= 0) == 0
    assert differing("persistent", "mixed", chunked) == 0

    assert differing("warp", "cta", ~one_warp) > 0
    assert differing("cta", "mixed", chunked) > 0
    assert differing("mixed", "split", chunked) > 0
    assert differing("persistent", "mixed", one_chunk) > 0


@_needs_cuda
@pytest.mark.parametrize("path", STATIC_POLICIES)
def test_pinned_sum_bits_do_not_depend_on_position_or_neighbours(path):
    """A pinned schedule sums a segment the same way wherever it sits.

    The same segments are laid out twice, the second time in reverse order
    with unrelated segments between them, so every one starts at another
    offset, next to other neighbours, in another task slot. Lanes and
    chunks are counted from the start of a segment, so its bits must not
    change.
    """
    generator = torch.Generator().manual_seed(1)
    segments = [
        torch.randn(length, generator=generator)
        for length in (17, 32, 33, 1000, 4096, 4097, 20000)
    ]
    fillers = [
        torch.randn(length, generator=generator)
        for length in (0, 1, 4099, 3, 77, 0, 5)
    ]
    moved = [
        piece
        for segment, filler in zip(reversed(segments), fillers)
        for piece in (filler, segment)
    ]

    def run(pieces):
        offsets = torch.tensor(
            _offsets([piece.numel() for piece in pieces]), dtype=torch.int32
        )
        return _run_sum(path, torch.cat(pieces), offsets)[0]

    straight = run(segments)
    scattered = run(moved)[1::2].flip(0)

    assert _bits(scattered) == _bits(straight)


@_needs_cuda
def test_schedule_selection_changes_bits_and_pinned_schedules_do_not():
    """Show the batch dependence of automatic selection and its remedy.

    With the default limits a batch of 8192-element segments is split,
    until it holds as many segments as the device has SMs. Then the mixed
    callable becomes the 128-lane CTA schedule. The segments the two
    batches share therefore change bits when one more segment joins the
    batch, and the batch size at which that happens depends on the GPU
    model.

    The three ways to pin the schedule today do not have that dependence:
    `select_schedule=False`, the `cta` callable, and the `warp` callable
    return the same bits for the shared segments in both batches.
    """
    count = torch.cuda.get_device_properties(0).multi_processor_count
    generator = torch.Generator().manual_seed(2)
    host_values = torch.randn(count * 8192, generator=generator)

    def run(segments, select_schedule):
        values = host_values[: segments * 8192].cuda()
        offsets = torch.tensor(
            _offsets([8192] * segments), dtype=torch.int32, device="cuda"
        )
        output = torch.empty(segments, device="cuda")
        prepared = _prepare_planned_reduction(
            values,
            offsets,
            output,
            module_text=reduction_module("sum", "identity"),
            kernel_name="segmented_sum",
            select_schedule=select_schedule,
        )
        results = {}
        for name in ("warp", "cta", "mixed"):
            output.fill_(float("nan"))
            getattr(prepared, name)()
            results[name] = _bits(output.cpu())[: count - 1]
        return results, prepared.mixed is prepared.cta

    selected_small, small_is_cta = run(count - 1, True)
    selected_full, full_is_cta = run(count, True)
    pinned_small, _ = run(count - 1, False)
    pinned_full, _ = run(count, False)

    assert not small_is_cta
    assert full_is_cta
    assert selected_small["mixed"] != selected_full["mixed"]
    assert selected_full["mixed"] == selected_full["cta"]
    assert pinned_small == pinned_full
    assert pinned_small["mixed"] == selected_small["mixed"]


LONG_LENGTHS = [100_003, 300_001, 1_048_577]
LONG_SUITES = ["wide", "cancelling"]

# Regression guard on the measured accuracy of the long sums, in units of
# eps32 * sum(|x|). The largest value measured on an RTX A6000 at sm_86 is
# 0.12. Unlike the tree bound this is not a guarantee for other values.
_MEASURED_SUM_EPS = 1.0


def _long_values(suite, count, generator):
    """Draw one long segment of mixed-sign values with a wide exponent range.

    - wide: random signs and magnitudes spread evenly over 80 binades, from
      2**-40 to 2**40.
    - cancelling: a third of the values are large, from 1 to 2**40, each
      paired with its exact negation, around small values from 2**-40 to 1,
      all shuffled so partners rarely share a lane or a chunk. The exact
      sum is the sum of the small values, below 1e-12 of sum(|x|).
    """
    signs = torch.randint(0, 2, (count,), generator=generator) * 2 - 1
    if suite == "wide":
        exponents = torch.empty(count).uniform_(-40, 40, generator=generator)
        return signs * torch.exp2(exponents)
    assert suite == "cancelling"
    pairs = count // 3
    large = torch.exp2(torch.empty(pairs).uniform_(0, 40, generator=generator))
    small = torch.exp2(
        torch.empty(count - 2 * pairs).uniform_(-40, 0, generator=generator)
    )
    values = torch.cat([large, -large, small * signs[: small.numel()]])
    return values[torch.randperm(count, generator=generator)]


@_needs_cuda
@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("suite", LONG_SUITES)
def test_long_sums_stay_within_the_tree_bound(suite, seed):
    """Bound every sum path against float64 at 10**5 elements and above.

    Each sum must lie within k * eps32 * sum(|x|) of the float64 sum of
    the same f32 values, with eps32 = 2**-23 and k the number of rounding
    additions on the longest path of the reduction tree (_summation_depth
    in test_segmented_runtime.py derives it, and the bound from it):

    - warp: k = ceil(n / 32) + 4.
    - cta and the one-CTA launch: k = ceil(n / 128) + 6.
    - mixed, whose default limits split these lengths into 4096-element
      chunks: k = 24 + ceil(ceil(n / 4096) / 512), which is 25 up to
      2097152 elements.
    - split, with 16-element chunks: k = 17 + ceil(ceil(n / 16) / 512).

    The float64 reference adds at most n * 2**-53 * sum(|x|) of its own
    error, below 2e-10 * sum(|x|) here and far inside the bound.

    The bound is a guarantee about the tree, not an estimate of the usual
    error. On an RTX A6000 at sm_86 the largest error over these inputs is
    0.12 * eps32 * sum(|x|), reached by the warp schedule, and 0.05 for
    the mixed schedule. Each sum must therefore also lie within
    _MEASURED_SUM_EPS * eps32 * sum(|x|), which is what notices a schedule
    that became less accurate on these inputs, or one that lost an element
    larger than that.

    This is an accuracy check, not an ownership check: both limits admit a
    dropped element that is small against sum(|x|).
    test_static_policies_own_every_element_of_long_segments compares
    exactly on position-dependent values.
    """
    generator = torch.Generator().manual_seed(seed)
    host_values = torch.cat(
        [_long_values(suite, length, generator) for length in LONG_LENGTHS]
    )
    assert host_values.dtype == torch.float32
    assert host_values.isfinite().all()
    host_offsets = torch.tensor(_offsets(LONG_LENGTHS), dtype=torch.int32)
    lengths = torch.tensor(LONG_LENGTHS)
    reference = torch.segment_reduce(
        host_values.double(), "sum", lengths=lengths
    )
    magnitude = torch.segment_reduce(
        host_values.double().abs(), "sum", lengths=lengths
    )
    if suite == "cancelling":
        assert (reference.abs() < 1e-12 * magnitude).all()

    for path in (*STATIC_POLICIES, "one-cta"):
        (actual,) = _run_sum(path, host_values, host_offsets)
        tree = "cta" if path == "one-cta" else path
        depth = torch.tensor(
            [_summation_depth(tree, length) for length in LONG_LENGTHS],
            dtype=torch.float64,
        )
        error = (actual.double() - reference).abs()
        bound = depth * _EPS32 * magnitude
        assert (error <= bound).all(), (
            f"{path}: error {error.tolist()} exceeds bound {bound.tolist()} "
            f"for lengths {LONG_LENGTHS}"
        )
        measured = error / (_EPS32 * magnitude)
        assert (measured <= _MEASURED_SUM_EPS).all(), (
            f"{path}: error is {measured.tolist()} eps32 * sum(|x|) "
            f"for lengths {LONG_LENGTHS}"
        )


# The longest segment the page bounds for the pinned mixed schedule, and
# the longest one that automatic selection moves to the CTA schedule.
_PINNED_MIXED_LIMIT = 2_097_152
_SELECTED_CTA_LIMIT = 8192
_SUM_ROUNDING_PAGE = (
    pathlib.Path(__file__).resolve().parents[3]
    / "docs"
    / "internals"
    / "segmented-reductions.md"
)


def _mixed_depth_limits():
    """Return the largest k of `mixed`, pinned and under selection.

    Pinned, `mixed` picks the tree from the length alone, and k is largest
    at the end of one of its three length classes. The scan also samples
    every class in case that stops being true. Under selection `mixed` is
    the CTA schedule for segments of 4097 to 8192 elements.
    """
    lengths = {1, 32, 33, 4096, 4097, _PINNED_MIXED_LIMIT}
    lengths.update(range(1, _PINNED_MIXED_LIMIT + 1, 257))
    pinned = max(_summation_depth("mixed", length) for length in lengths)
    selected = max(
        _summation_depth("cta", length)
        for length in range(4097, _SELECTED_CTA_LIMIT + 1)
    )
    return pinned, selected


def test_sum_rounding_page_states_both_mixed_depth_limits():
    """Keep the documented bound of the default schedule equal to the trees.

    The page quotes one k for `mixed` with `select_schedule=False` and a
    larger one for the batches in which automatic selection, the default,
    makes `mixed` the CTA schedule. Both come from _summation_depth, which
    the float64 bound tests use, so the page cannot state a bound that the
    tests do not check.
    """
    pinned, selected = _mixed_depth_limits()
    page = " ".join(_SUM_ROUNDING_PAGE.read_text().split())

    assert (pinned, selected) == (38, 70)
    for depth in (pinned, selected):
        assert f"at most {depth}" in page
        assert f"`{depth * _EPS32:.1e} * sum(|x|)`" in page


@_needs_cuda
@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("suite", LONG_SUITES)
def test_selected_schedule_stays_within_the_cta_tree_bound(suite, seed):
    """Bound the default `mixed` on a batch that automatic selection moves.

    With `select_schedule=True`, the default, a batch of as many
    8192-element segments as the device has SMs runs the CTA schedule
    instead of the split one. Its bound is therefore the CTA one,
    k = ceil(8192 / 128) + 6 = 70, and not the 38 that holds for `mixed`
    with a pinned schedule. test_schedule_selection_changes_bits_and_
    pinned_schedules_do_not compares the bits of this batch; this test
    compares its sums with float64.
    """
    count = torch.cuda.get_device_properties(0).multi_processor_count
    lengths = [_SELECTED_CTA_LIMIT] * count
    generator = torch.Generator().manual_seed(seed)
    host_values = torch.cat(
        [_long_values(suite, length, generator) for length in lengths]
    )
    assert host_values.dtype == torch.float32
    reference = torch.segment_reduce(
        host_values.double(), "sum", lengths=torch.tensor(lengths)
    )
    magnitude = torch.segment_reduce(
        host_values.double().abs(), "sum", lengths=torch.tensor(lengths)
    )
    values = host_values.cuda()
    offsets = torch.tensor(_offsets(lengths), dtype=torch.int32, device="cuda")
    output = torch.full((count,), float("nan"), device="cuda")

    prepared = _prepare_planned_reduction(
        values,
        offsets,
        output,
        module_text=reduction_module("sum", "identity"),
        kernel_name="segmented_sum",
    )
    prepared.mixed()

    pinned, selected = _mixed_depth_limits()
    assert prepared.mixed is prepared.cta
    assert _summation_depth("cta", _SELECTED_CTA_LIMIT) == selected > pinned
    error = (output.cpu().double() - reference).abs()
    assert (error <= selected * _EPS32 * magnitude).all(), (
        f"error {error.tolist()} exceeds {selected} * eps32 * sum(|x|)"
    )
    measured = error / (_EPS32 * magnitude)
    assert (measured <= _MEASURED_SUM_EPS).all(), (
        f"error is {measured.tolist()} eps32 * sum(|x|)"
    )


_DENORMAL = 2.0**-149

SPECIAL_LENGTHS = [2, 32, 33, 4096, 4097, 8193]


def _smallest_subnormal(dtype):
    """Return the smallest positive value of a float dtype."""
    return _DENORMAL if dtype == torch.float32 else 2.0**-1074


def _special_segment(case, count, dtype=torch.float32):
    """Build one segment whose sum has a single correct value in `dtype`.

    The descriptions below give the float32 values. A float64 segment uses
    the largest finite float64, the smallest float64 subnormal, 2**-1074,
    and the smallest float64 normal, 2**-1022, in their places, so its
    cases are out of reach of float32 arithmetic.


    Every case is independent of the order of addition:

    - nan: ones and a NaN in the last position, so the sum is NaN.
    - positive-infinity, negative-infinity: ones and one infinity in the
      middle, so the sum is that infinity.
    - opposite-infinities: positive infinity first and negative infinity
      last, so the sum is NaN.
    - overflow: copies of the largest finite f32, whose sum overflows to
      positive infinity.
    - denormal: copies of the smallest subnormal, 2**-149. Every partial
      sum is a subnormal multiple of 2**-149, so the sum is exact.
    - denormal-and-normal: the smallest normal, 2**-126, and copies of the
      smallest subnormal. Every partial sum is a multiple of 2**-149 below
      2**-125, so the sum is exact and crosses into the normal range.
    """
    values = torch.ones(count, dtype=dtype)
    if case == "nan":
        values[-1] = float("nan")
    elif case == "positive-infinity":
        values[count // 2] = float("inf")
    elif case == "negative-infinity":
        values[count // 2] = float("-inf")
    elif case == "opposite-infinities":
        values[0], values[-1] = float("inf"), float("-inf")
    elif case == "overflow":
        values.fill_(torch.finfo(dtype).max)
    elif case == "denormal":
        values.fill_(_smallest_subnormal(dtype))
    else:
        assert case == "denormal-and-normal"
        values.fill_(_smallest_subnormal(dtype))
        values[count // 2] = torch.finfo(dtype).tiny
    return values


# The exact sum of each special segment, as a function of its length and of
# its dtype.
SPECIAL_CASES = {
    "nan": lambda count, dtype=torch.float32: float("nan"),
    "positive-infinity": lambda count, dtype=torch.float32: float("inf"),
    "negative-infinity": lambda count, dtype=torch.float32: float("-inf"),
    "opposite-infinities": lambda count, dtype=torch.float32: float("nan"),
    "overflow": lambda count, dtype=torch.float32: float("inf"),
    "denormal": lambda count, dtype=torch.float32: (
        count * _smallest_subnormal(dtype)
    ),
    "denormal-and-normal": lambda count, dtype=torch.float32: (
        torch.finfo(dtype).tiny + (count - 1) * _smallest_subnormal(dtype)
    ),
}


@_needs_cuda
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64], ids=["float32", "float64"]
)
@pytest.mark.parametrize("policy", STATIC_POLICIES)
@pytest.mark.parametrize("case", SPECIAL_CASES)
def test_sum_propagates_special_values(case, policy, dtype):
    """NaN, both infinities, and subnormals sum as IEEE-754 addition does.

    Each case runs at lengths on both sides of the warp and chunk limits,
    so the mixed schedule takes its warp, CTA, and split paths, and the
    split schedule carries the special value through a partial and a
    merge. The special element sits in the middle or in the last position
    of its segment, so a lane other than lane zero, which stores the
    result, reads it.

    The subnormal cases are exact, and they fail if any addition flushes
    a subnormal operand or result to zero. Results that are not NaN are
    compared bit for bit.
    """
    host_values = torch.cat(
        [_special_segment(case, length, dtype) for length in SPECIAL_LENGTHS]
    )
    host_offsets = torch.tensor(_offsets(SPECIAL_LENGTHS), dtype=torch.int32)
    exact = torch.tensor(
        [SPECIAL_CASES[case](length, dtype) for length in SPECIAL_LENGTHS],
        dtype=torch.float64,
    )
    expected = exact.to(dtype)
    # A finite expected value must be a value of the dtype, or the
    # comparison would test the rounding of this table instead of the
    # kernel.
    finite = exact.isfinite()
    assert torch.equal(expected[finite].double(), exact[finite])

    (actual,) = _run_sum(policy, host_values, host_offsets)

    if case in ("nan", "opposite-infinities"):
        assert actual.isnan().all()
    else:
        assert _bits(actual) == _bits(expected)


@_needs_cuda
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64], ids=["float32", "float64"]
)
@pytest.mark.parametrize("policy", [*STATIC_POLICIES, "one-cta"])
@pytest.mark.parametrize("kind", ["min", "max"])
def test_extremes_propagate_nan_and_order_infinities(kind, policy, dtype):
    """A minimum and a maximum follow IEEE-754 minimum and maximum.

    The cases are written for the minimum and negated for the maximum. The
    finite values of a segment descend, so its minimum is its last element
    or the one before it, which a lane other than the one that stores the
    result reads. Each case runs at lengths on both sides of the warp and
    chunk limits, so the split schedule carries the special value through
    a partial and a merge. Results that are not NaN are compared bit for
    bit, which tells the two zeros apart: the minimum of zeros with one
    negative zero among them is the negative zero.
    """
    sign = 1.0 if kind == "min" else -1.0
    infinity = float("inf")
    subnormal = _smallest_subnormal(dtype)
    expected = []
    segments = []
    for length in SPECIAL_LENGTHS:
        ramp = torch.arange(length, 0, -1, dtype=dtype)
        cases = {
            "nan-first": (ramp.clone(), float("nan")),
            "nan-last": (ramp.clone(), float("nan")),
            "negative-infinity": (ramp.clone(), -infinity),
            "all-positive-infinity": (
                torch.full((length,), infinity, dtype=dtype),
                infinity,
            ),
            "positive-infinity": (ramp.clone(), 2.0),
            "opposite-infinities": (ramp.clone(), -infinity),
            "subnormal": (ramp.clone(), subnormal),
            "negative-zero": (torch.zeros(length, dtype=dtype), -0.0),
        }
        cases["nan-first"][0][0] = float("nan")
        cases["nan-last"][0][-1] = float("nan")
        cases["negative-infinity"][0][length // 2] = -infinity
        cases["positive-infinity"][0][-1] = infinity
        cases["opposite-infinities"][0][0] = infinity
        cases["opposite-infinities"][0][-1] = -infinity
        cases["subnormal"][0][-1] = subnormal
        cases["negative-zero"][0][length // 2] = -0.0
        for values, result in cases.values():
            segments.append(sign * values)
            expected.append(sign * result)
    host_values = torch.cat(segments)
    host_offsets = torch.tensor(
        _offsets([segment.numel() for segment in segments]),
        dtype=torch.int32,
    )
    expected = torch.tensor(expected, dtype=dtype)

    (actual,) = _run_sum(policy, host_values, host_offsets, kind=kind)

    nan = expected.isnan()
    assert actual[nan].isnan().all()
    assert _bits(actual[~nan]) == _bits(expected[~nan])


# One entry per kernel that adds f32 values, as (label, compile function,
# options). The persistent kernel admits only the identity sum.
_SUM_KERNELS = [
    ("one-cta", "_compile_segmented_reduction_ptx", {"block_size": 128}),
    (
        "warp-tasks",
        "_compile_segmented_reduction_ptx",
        {"block_size": 32, "use_task_ids": True},
    ),
    (
        "cta-tasks",
        "_compile_segmented_reduction_ptx",
        {"block_size": 128, "use_task_ids": True},
    ),
    ("fused-mixed", "_compile_fused_segmented_reduction_ptx", {}),
    ("split-partial", "_compile_split_partial_reduction_ptx", {}),
    ("split-merge", "_compile_split_merge_reduction_ptx", {}),
    ("persistent", "_compile_persistent_segmented_reduction_ptx", {}),
]

# An instruction with its dotted suffixes, as `add.rn.f32` or
# `cvt.rn.f32.f64`. One that names a float type in any suffix is a float
# instruction: a conversion names its float type before its last suffix.
_INSTRUCTION = re.compile(
    r"^\s*(?P<name>[a-z][a-z0-9]*)(?P<suffixes>(?:\.[A-Za-z0-9]+)+)\s",
    re.MULTILINE,
)
_FLOAT_TYPES = {"f16", "f32", "f64"}
_ROUNDED = {"add", "sub", "mul", "div"}


def _float_arithmetic(ptx, element="f32"):
    """Return the unsafe float instructions of a PTX text and its counts.

    `element` is the element type of the kernel, `"f32"` or `"f64"`. Every
    float instruction of a kernel has that type.

    PTX gives add, sub, mul, and div an optional rounding modifier. The
    PTX specification lets the driver fuse an add and a multiply that carry
    no modifier into a fused multiply-add, and treats one with an explicit
    modifier conservatively. The pinned NVPTX backend writes the bare
    spelling of an add, subtract, or multiply when contraction is allowed,
    by the target options or by the instruction's `contract` flag, and
    `.rn` otherwise. So a kernel whose f32 arithmetic is all `.rn`, with
    no `fma` or `mad`, performs the IEEE-754 round-to-nearest operations
    of the semantic program, each rounded once.

    `.ftz` would flush subnormal values to zero and `.sat` would clamp, so
    both are violations. So is any float type other than the element type:
    a narrower instruction in an f64 kernel would lose bits, and a wider
    one in an f32 kernel would round twice. The rule covers every suffix,
    so a conversion between the two widths, such as `cvt.rn.f32.f64`, is a
    violation in a kernel of either type.

    Returns:
        A list of violating instruction spellings and a dict that counts
        each accepted spelling.
    """
    violations = []
    counts = {}
    for match in _INSTRUCTION.finditer(ptx):
        name = match["name"]
        suffixes = match["suffixes"].split(".")[1:]
        types = [suffix for suffix in suffixes if suffix in _FLOAT_TYPES]
        if not types:
            continue
        modifiers = [suffix for suffix in suffixes if suffix not in types]
        spelling = f"{name}{match['suffixes']}"
        if name in ("fma", "mad") or any(kind != element for kind in types):
            violations.append(spelling)
        elif name in _ROUNDED and modifiers != ["rn"]:
            violations.append(spelling)
        elif "ftz" in modifiers or "sat" in modifiers:
            violations.append(spelling)
        else:
            counts[spelling] = counts.get(spelling, 0) + 1
    return violations, counts


def test_float_arithmetic_scan_flags_every_unsafe_spelling():
    """The PTX scan rejects contraction, other rounding, and flushing.

    The compiled kernels below contain none of these, so without this
    check a scan that matched nothing would pass them too.
    """
    unsafe = [
        "fma.rn.f32",
        "mad.f32",
        "add.f32",
        "mul.f32",
        "add.rz.f32",
        "sub.rm.f32",
        "div.rp.f32",
        "mul.rn.ftz.f32",
        "add.rn.sat.f32",
        "ex2.approx.ftz.f32",
        "add.rn.f64",
    ]
    safe = [
        "add.rn.f32",
        "mul.rn.f32",
        "max.f32",
        "max.NaN.f32",
        "ex2.approx.f32",
    ]
    integer = ["mad.lo.s64", "add.s64", "mul.lo.s32", "ld.global.u32"]
    ptx = "".join(
        f"\t{spelling} \t%r1, %r2, %r3;\n"
        for spelling in (*unsafe, *safe, *integer)
    )

    violations, counts = _float_arithmetic(ptx)

    assert violations == unsafe
    assert counts == dict.fromkeys(safe, 1)


def test_float_arithmetic_scan_holds_a_kernel_to_its_element_type():
    """The scan of an f64 kernel rejects f32 arithmetic, and the reverse.

    A scan that accepted either width would pass an f64 kernel that
    accumulated in f32. The f64 spellings are the ones the pinned backend
    writes for f64 kernels: round-to-nearest arithmetic, and the compare
    and select instructions of the NaN-propagating maximum and minimum.
    The f32 maximum and minimum are one instruction each, `max.NaN.f32`
    and `min.NaN.f32`.
    """
    narrow = [
        "add.rn.f32",
        "max.NaN.f32",
        "min.NaN.f32",
        "selp.f32",
        "ex2.approx.f32",
    ]
    wide = [
        "add.rn.f64",
        "mul.rn.f64",
        "div.rn.f64",
        "max.f64",
        "min.f64",
        "setp.nan.f64",
        "setp.eq.f64",
        "selp.f64",
    ]
    unsafe_wide = ["add.f64", "fma.rn.f64", "mul.rn.ftz.f64", "sub.rz.f64"]
    # A conversion between the widths belongs in no kernel, and one from an
    # integer belongs in the kernel of its float type only. The float type
    # of a conversion is not its last suffix.
    between = ["cvt.rn.f32.f64", "cvt.f64.f32"]
    from_integer = ["cvt.rn.f64.s32"]
    integer = ["cvt.s64.s32", "ld.global.b64", "selp.b64"]
    ptx = "".join(
        f"\t{spelling} \t%r1, %r2, %r3;\n"
        for spelling in (
            *narrow, *wide, *unsafe_wide, *between, *from_integer, *integer
        )
    )

    as_f64, counts_f64 = _float_arithmetic(ptx, "f64")
    as_f32, counts_f32 = _float_arithmetic(ptx, "f32")

    assert as_f64 == [*narrow, *unsafe_wide, *between]
    assert counts_f64 == dict.fromkeys([*wide, *from_integer], 1)
    assert as_f32 == [*wide, *unsafe_wide, *between, *from_integer]
    assert counts_f32 == dict.fromkeys(narrow, 1)


@pytest.mark.parametrize("target", ["sm_80", "sm_86"])
@pytest.mark.parametrize(
    ("kind", "transform"),
    [("sum", "identity"), ("sum", "square"), ("max", "identity"),
     ("min", "identity")],
)
def test_f64_kernels_hold_f64_round_to_nearest_arithmetic_only(
    kind, transform, target
):
    """Every f64 kernel computes in f64, rounded to nearest, uncontracted.

    A sum adds with `add.rn.f64`, and the square program keeps its
    `mul.rn.f64` beside it. The pinned backend writes no single
    NaN-propagating instruction for an f64 maximum or minimum: it expands
    each into `max.f64` or `min.f64` with a NaN test, a zero test, and
    selects, which the scan admits and which involve no rounding. No kernel
    holds an f32 instruction, which is what an f32 identity, load, or
    accumulator would be: scanned as an f32 kernel, each one is all
    violations.
    """
    for label, compiler, options in _SUM_KERNELS:
        if label == "persistent":
            continue
        with ir.Context() as context:
            swage_dialect.register_dialects(context)
            module = ir.Module.parse(reduction_module(kind, transform, "f64"))
            _, ptx = getattr(native_swage, compiler)(
                module,
                kernel_name=_reduction_kernel(kind, "f64"),
                target=target,
                **options,
            )

        violations, counts = _float_arithmetic(ptx, "f64")

        assert not violations, f"{label}: {violations}"
        if kind == "sum":
            assert counts.get("add.rn.f64", 0) > 0, label
            multiplies = transform == "square" and label != "split-merge"
            assert ("mul.rn.f64" in counts) == multiplies, label
        else:
            assert counts.get(f"{kind}.f64", 0) > 0, label
            assert counts.get("setp.nan.f64", 0) > 0, label
            assert not any(name.startswith("add.") for name in counts), label
        wrong_width, as_f32 = _float_arithmetic(ptx, "f32")
        assert wrong_width and not as_f32, label


# Lengths on both sides of the warp and chunk limits and one segment of
# seventeen chunks, as ORDER_LENGTHS, for the f64 bound.
def _f64_case(seed):
    """Draw f64 values over sixteen binades whose sum depends on the order."""
    generator = torch.Generator().manual_seed(seed)
    count = sum(ORDER_LENGTHS)
    exponents = torch.empty(count, dtype=torch.float64).uniform_(
        -8, 8, generator=generator
    )
    values = torch.randn(
        count, dtype=torch.float64, generator=generator
    ) * torch.exp2(exponents)
    return values, torch.tensor(_offsets(ORDER_LENGTHS), dtype=torch.int32)


@_needs_cuda
@pytest.mark.parametrize("seed", range(3))
def test_f64_sums_stay_within_the_tree_bound_of_the_exact_sum(seed):
    """Bound every f64 sum path against the exactly rounded sum.

    Each sum must lie within k * eps64 * sum(|x|) of the exact sum of its
    segment, with eps64 = 2**-52 and k the depth of the tree of its path,
    the same trees as for f32 (_summation_depth). The reference is
    `math.fsum`, the float64 nearest to the exact sum: a float64 reference
    summed by another tree would carry rounding of the size being bounded.

    The bound is a guarantee about the tree. On an RTX A6000 at sm_86 the
    largest error over these inputs is 0.35 * eps64 * sum(|x|), so each sum
    must also lie within _MEASURED_SUM_EPS * eps64 * sum(|x|).

    The schedules add in different orders, so their bits differ on these
    values, as for f32. Two launches of one path return the same bits.
    """
    host_values, host_offsets = _f64_case(seed)
    bounds = list(pairwise(host_offsets.tolist()))
    host = host_values.tolist()
    exact = torch.tensor(
        [math.fsum(host[begin:end]) for begin, end in bounds],
        dtype=torch.float64,
    )
    magnitude = torch.tensor(
        [math.fsum(map(abs, host[begin:end])) for begin, end in bounds],
        dtype=torch.float64,
    )
    patterns = set()

    for path in (*STATIC_POLICIES, "one-cta"):
        first, second = _run_sum(path, host_values, host_offsets, launches=2)
        assert first.dtype == torch.float64
        assert _bits(first) == _bits(second), path
        patterns.add(tuple(_bits(first)))
        tree = "cta" if path == "one-cta" else path
        depth = torch.tensor(
            [_summation_depth(tree, length) for length in ORDER_LENGTHS],
            dtype=torch.float64,
        )
        error = (first - exact).abs()
        assert (error <= depth * _EPS64 * magnitude).all(), (
            f"{path}: error {error.tolist()} exceeds the tree bound for "
            f"lengths {ORDER_LENGTHS}"
        )
        measured = (error / (_EPS64 * magnitude)).nan_to_num()
        assert (measured <= _MEASURED_SUM_EPS).all(), (
            f"{path}: error is {measured.tolist()} eps64 * sum(|x|)"
        )
    assert len(patterns) > 1


@_needs_cuda
@pytest.mark.parametrize("kind", ["sum", "max", "min"])
@pytest.mark.parametrize("policy", [*STATIC_POLICIES, "one-cta"])
def test_f64_reductions_do_not_round_through_f32(kind, policy):
    """Keep the bits of f64 values that f32 cannot represent.

    The values are multiples of 0.25 plus small multiples of 2**-30, and
    every partial sum stays below 2**21, so each sum is exact in f64
    whatever the order. A kernel that loaded, accumulated, or stored in f32
    would lose the 2**-30 part. The values depend on their position, so a
    moved window, or a dropped or repeated element, changes a result.
    """
    lengths = [0, 1, 31, 33, 97, 300, 4097, 8193, 65537]
    host_offsets = torch.tensor(_offsets(lengths), dtype=torch.int32)
    index = torch.arange(sum(lengths))
    host_values = (2 * (index % 67) - 65).double() / 4 + (
        index % 5
    ).double() * 2.0**-30
    assert not torch.equal(host_values.float().double(), host_values)
    reduce, identity = {
        "sum": (math.fsum, 0.0),
        "max": (max, float("-inf")),
        "min": (min, float("inf")),
    }[kind]
    host = host_values.tolist()
    expected = torch.tensor(
        [
            reduce(host[begin:end]) if end > begin else identity
            for begin, end in pairwise(host_offsets.tolist())
        ],
        dtype=torch.float64,
    )

    (actual,) = _run_sum(policy, host_values, host_offsets, kind=kind)

    assert _bits(actual) == _bits(expected)


@pytest.mark.parametrize("target", ["sm_80", "sm_86"])
@pytest.mark.parametrize(
    "transform", ["identity", "square", "maps", "affine4"]
)
def test_sum_kernels_add_in_round_to_nearest_without_contraction(
    transform, target
):
    """Every sum kernel keeps separate round-to-nearest f32 operations.

    The square and affine4 programs are the contraction candidates: square
    feeds a multiply into the accumulating add, and affine4 feeds one into
    an add inside the element program. Both must keep their `mul.rn.f32`
    next to `add.rn.f32`. The identity and maps programs have no multiply
    by a non-constant, so for them the check is on rounding alone.

    This needs no device. It pins the property that makes a sum a function
    of its values and its reduction tree: a change to the LLVM pin or to
    the target options that introduced contraction, another rounding
    mode, or flush-to-zero would fail here.
    """
    for label, compiler, options in _SUM_KERNELS:
        if label == "persistent" and transform != "identity":
            continue
        with ir.Context() as context:
            swage_dialect.register_dialects(context)
            module = ir.Module.parse(reduction_module("sum", transform))
            _, ptx = getattr(native_swage, compiler)(
                module, kernel_name="segmented_sum", target=target, **options
            )

        violations, counts = _float_arithmetic(ptx)

        assert not violations, f"{label}: {violations}"
        assert counts.get("add.rn.f32", 0) > 0, label
        multiplies = transform in ("square", "affine4") and (
            label != "split-merge"
        )
        assert ("mul.rn.f32" in counts) == multiplies, label


@pytest.mark.parametrize("target", ["sm_80", "sm_86"])
def test_softmax_kernel_rounds_to_nearest_around_the_approximate_exp2(target):
    """Softmax is IEEE-754 arithmetic around one approximate instruction.

    The subtract, multiply, add, and divide are round to nearest and not
    contracted, so `ex2.approx.f32` is the only operation whose result
    the IEEE-754 standard does not define.
    """
    with ir.Context() as context:
        swage_dialect.register_dialects(context)
        module = ir.Module.parse(_SOFTMAX_MODULE)
        _, ptx = native_swage._compile_segmented_reduction_ptx(
            module, kernel_name="ragged_softmax", target=target, block_size=128
        )

    violations, counts = _float_arithmetic(ptx)

    assert not violations
    assert counts["ex2.approx.f32"] == 2
    for spelling in ("sub.rn.f32", "mul.rn.f32", "add.rn.f32", "div.rn.f32"):
        assert counts.get(spelling, 0) > 0, spelling


# Relative error of ex2.approx.f32, in units of eps32, that the softmax
# bound below budgets for. The largest value measured on an RTX A6000 at
# sm_86 is 1.22, see test_gpu_exp2_stays_within_the_measured_bound.
_EX2_RELATIVE_EPS = 1.5


@_needs_cuda
def test_gpu_exp2_stays_within_the_measured_bound():
    """Measure `ex2.approx.f32` on the device against float64.

    A one-element sum of exp2(x) returns the instruction's result
    unchanged, because adding it to the zero identity and to the zeros of
    the idle lanes is exact. The inputs are 2**20 evenly spaced f32 values
    from -126 to 126, so every result is a normal f32, followed by every
    integer in that range.

    Measured on an RTX A6000 at sm_86: at most 2 ulp from the correctly
    rounded f32, at most 2.06 ulp and 1.22 * eps32 relative from the exact
    value, 48 percent of results correctly rounded, and every integer
    argument exact. The assertions keep those limits, with the relative
    bound rounded up to _EX2_RELATIVE_EPS. This is a measurement on one
    architecture. A different result on another one is a finding about
    that architecture, not a defect in this test.
    """
    grid = torch.linspace(-126, 126, 1 << 20, dtype=torch.float64).float()
    integers = torch.arange(-126, 127, dtype=torch.float32)
    host_values = torch.cat([grid, integers])
    count = host_values.numel()
    output = torch.full((count,), float("nan"), device="cuda")
    prepared = _prepare_planned_reduction(
        host_values.cuda(),
        torch.arange(count + 1, dtype=torch.int32, device="cuda"),
        output,
        module_text=reduction_module("sum", "exp2"),
        kernel_name="segmented_sum",
        select_schedule=False,
    )

    prepared.mixed()

    actual = output.cpu()
    exact = torch.exp2(host_values.double())
    rounded = exact.float()
    assert rounded.isfinite().all()
    assert (rounded >= torch.finfo(torch.float32).tiny).all()
    steps = actual.view(torch.int32).long() - rounded.view(torch.int32).long()
    assert steps.abs().max() <= 2
    relative = (actual.double() - exact).abs() / exact
    assert relative.max() <= _EX2_RELATIVE_EPS * _EPS32
    assert _bits(actual[grid.numel() :]) == _bits(torch.exp2(integers))


SOFTMAX_SPREADS = [8, 20, 50, 80]

# Growth of the relative error of one exponential, in units of eps32 per
# unit of distance below the segment maximum. See _softmax_bound.
_SOFTMAX_SLOPE_EPS = 1.12


def _softmax_segments(spread, generator):
    """Build segments whose logits span exactly `spread`.

    - grid: every multiple of 1/8 from the maximum down to the maximum
      minus the spread, shuffled, at four maxima. The shift by the maximum
      is exact for these, so the only rounding before `exp2` is the
      multiply.
    - uniform: f32 logits drawn uniformly over the spread, with both ends
      planted, at lengths from 2 to 4096 and maxima from -1000 to 40.
      The maxima are dyadic, so the planted ends are exact.
    - two-level: one logit at the maximum and all others at the maximum
      minus the spread. At spread 8 more than half of the probability
      sits at the far level, which is the case that makes the normalizer
      inherit the error of distant terms.

    The smallest output is above 1e-38 at spread 80, so every output is a
    normal f32 and is compared relatively.
    """
    segments = []
    steps = torch.arange(8 * spread + 1, dtype=torch.float32) / 8
    for maximum in (0.0, 3.0, -17.5, 40.0):
        grid = maximum - steps
        segments.append(grid[torch.randperm(grid.numel(), generator=generator)])
    for count, maximum in (
        (2, 0.25),
        (129, -3.0),
        (1024, 40.0),
        (4096, 0.375),
        (4096, -1000.0),
    ):
        uniform = maximum - spread * torch.rand(count, generator=generator)
        uniform[0], uniform[-1] = maximum, maximum - spread
        segments.append(uniform[torch.randperm(count, generator=generator)])
    for count in (1024, 4096):
        levels = torch.full((count,), 5.0 - spread)
        levels[count // 3] = 5.0
        segments.append(levels)
    return segments


def _softmax_bound(logits, reference):
    """Bound the relative error of each f32 softmax output of one segment.

    Write d = max - v for the distance of a logit below its segment
    maximum and u = eps32 / 2 for the unit roundoff. The kernel computes
    exp2(fl(fl(v - max) * log2e)) with the f32 constant log2e, sums the
    results, and divides. To first order:

    - One exponential is off by at most d * 1.112 * eps32 + E * eps32. The
      subtraction and the multiplication each round the exponent by at
      most u relative, and the f32 constant is 0.224 u low. A relative
      change t of the exponent d changes exp(-d) by d * t, which gives
      d * (1 + 1 + 0.224) u. E is the relative error of `ex2.approx.f32`,
      _EX2_RELATIVE_EPS.
    - The normalizer is a sum of positive terms, so it inherits the
      probability-weighted mean of the errors of its terms, at most
      (dbar * 1.112 + E) * eps32 with dbar = sum(p * d), plus k * u for
      the k rounding additions of the 128-lane tree.
    - The division rounds once, u.

    Together, with one eps32 added for the higher-order terms:

        (1.12 * (d + dbar) + 2 * E + 1.5 + k / 2) * eps32

    Both d and dbar are at most the spread of the segment, so the bound is
    at most (2.24 * spread + 2 * E + 1.5 + k / 2) * eps32 whatever the
    distribution of the logits. dbar is below 1.2 when the logits are
    spread evenly, and it cannot exceed the natural logarithm of the
    segment length.
    """
    distance = logits.max() - logits
    mean_distance = (reference * distance).sum()
    depth = _summation_depth("cta", logits.numel())
    return _EPS32 * (
        _SOFTMAX_SLOPE_EPS * (distance + mean_distance)
        + 2 * _EX2_RELATIVE_EPS
        + 1.5
        + depth / 2
    )


@_needs_cuda
@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("spread", SOFTMAX_SPREADS)
def test_gpu_softmax_error_grows_linearly_with_logit_spread(spread, seed):
    """Bound GPU softmax against float64 at logit spreads of 8 to 80.

    The reference is torch.softmax in float64 of the same f32 logits, so
    it carries the exact shift and no f32 rescaling. Every output must lie
    within the relative bound of _softmax_bound.

    Measured on an RTX A6000 at sm_86 over these inputs, the largest
    relative error and the largest fraction of the bound reached:

        spread  8: 5.5e-07 ( 4.6 eps32), 0.25 of the bound
        spread 20: 1.0e-06 ( 8.4 eps32), 0.32 of the bound
        spread 50: 3.4e-06 (28.2 eps32), 0.45 of the bound
        spread 80: 3.7e-06 (31.0 eps32), 0.44 of the bound

    The existing comparison against float32 PyTorch keeps its own
    tolerance, which is sized for a spread of at most 8
    (test_segmented_runtime.py).
    """
    generator = torch.Generator().manual_seed(seed)
    segments = _softmax_segments(spread, generator)
    host_values = torch.cat(segments)
    host_offsets = torch.tensor(
        _offsets([segment.numel() for segment in segments]), dtype=torch.int32
    )
    output = torch.full((host_values.numel(),), float("nan"), device="cuda")

    launch_softmax_gpu(host_values.cuda(), host_offsets.cuda(), output)

    actual = output.cpu().double()
    for begin, end in pairwise(host_offsets.tolist()):
        logits = host_values[begin:end].double()
        assert logits.max() - logits.min() == spread
        reference = torch.softmax(logits, 0)
        assert (reference >= torch.finfo(torch.float32).tiny).all()
        relative = (actual[begin:end] - reference).abs() / reference
        bound = _softmax_bound(logits, reference)
        assert (relative <= bound).all(), (
            f"segment [{begin}, {end}): relative error "
            f"{relative.max().item():.3e} exceeds its bound"
        )


if __name__ == "__main__":
    print(json.dumps(_order_sensitive_bits()))
