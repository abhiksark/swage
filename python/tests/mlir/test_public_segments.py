# python/tests/mlir/test_public_segments.py
"""Contract of the public segmented calls.

`swage.segment_reduce` and `swage.segment_softmax` wrap the private runner.
This file pins what the wrapper adds and what a caller can rely on:

- The argument contract. These tests use host tensors and need no GPU.
- That int64 offsets are validated before they are narrowed, give the bits
  of int32 offsets, and reach a kernel only as a private int32 copy.
- Results against `torch.segment_reduce`, `torch.softmax`, and float64
  references, on the shapes of the nine benchmark distributions, on empty
  batches and segments, on one long segment, and on special values. A
  float64 sum is held to an exactly rounded reference instead.
- That a sum is the documented schedule of its batch and nothing a caller
  can pin.
- The behavior under CUDA graph capture, under `torch.compile`, on
  another stream, on a thread that has not used CUDA, under
  `torch.inference_mode()`, and with `SWAGE_NO_COMPILE=1`.
- That calls with fresh offsets leave no task storage, event, or module
  behind.
"""

import collections
import fractions
import gc
import importlib.util
import math
import pathlib
import sys
import threading
import weakref
from itertools import pairwise

import pytest
import swage
import torch
from swage import _runtime, _segments
from swage import _segmented_qualification as qualification
from test_segmented_numerics import (
    SPECIAL_CASES,
    SPECIAL_LENGTHS,
    _order_sensitive_case,
    _softmax_bound,
    _special_segment,
)
from test_segmented_runtime import (
    _EPS32,
    _EPS64,
    _bits,
    _offsets,
    _summation_depth,
)

_needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"
)
_SENTINEL = -5.0


def _load_distributions():
    """Import the benchmark length distributions from the checkout."""
    path = (
        pathlib.Path(__file__).resolve().parents[3]
        / "benchmarks"
        / "distributions.py"
    )
    spec = importlib.util.spec_from_file_location("distributions", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_distributions = _load_distributions()
DISTRIBUTIONS = sorted(_distributions._NAMES)
# Small batches of every distribution. The power-law batch is larger: its
# longest segment grows with the count, and at this count it passes the
# 4096-element chunk limit and the 8192-element selection limit.
_SEGMENT_COUNTS = {name: 257 for name in DISTRIBUTIONS} | {"power-law": 2048}
KINDS = ["sum", "max", "min", "mean"]
DTYPES = [torch.float32, torch.float64]
_DTYPE_IDS = ["float32", "float64"]


def _host_case(lengths, generator, dtype=torch.float32):
    """Draw mixed-sign values over many binades for the given lengths."""
    count = sum(lengths)
    exponents = torch.empty(count, dtype=dtype).uniform_(
        -8, 8, generator=generator
    )
    values = torch.randn(
        count, dtype=dtype, generator=generator
    ) * torch.exp2(exponents)
    return values, torch.tensor(_offsets(lengths), dtype=torch.int32)


def _exact_sums(values, offsets):
    """Return the exactly rounded float64 sum of every segment.

    `math.fsum` returns the float64 nearest to the exact sum, which is the
    reference a float64 result needs: a float64 summation of the same
    values would carry rounding of its own.
    """
    bounds = offsets.tolist()
    host = values.double().tolist()
    return torch.tensor(
        [math.fsum(host[begin:end]) for begin, end in pairwise(bounds)],
        dtype=torch.float64,
    )


def _exact_means(values, offsets):
    """Return the exactly rounded float64 mean of every segment.

    The sum of a segment is formed in rational arithmetic and divided by
    its length before anything is rounded, so the result is the float64
    nearest to the exact mean. `math.fsum` divided by the length would
    round twice. An empty segment has no mean and gives NaN.
    """
    bounds = offsets.tolist()
    host = [fractions.Fraction(value) for value in values.double().tolist()]
    return torch.tensor(
        [
            float(sum(host[begin:end], fractions.Fraction()) / (end - begin))
            if end > begin
            else float("nan")
            for begin, end in pairwise(bounds)
        ],
        dtype=torch.float64,
    )


def _assert_same_results(actual, expected):
    """Require the same NaN positions and the same bits everywhere else.

    Two NaN results may differ in sign and payload, which the IEEE-754
    standard leaves open for an invalid operation.
    """
    assert actual.dtype == expected.dtype
    assert torch.equal(actual.isnan(), expected.isnan())
    finite = ~expected.isnan()
    assert _bits(actual[finite]) == _bits(expected[finite])


def _assert_mean_matches(host_values, host_offsets, actual):
    """Compare one public mean with its sum, a bound, and PyTorch.

    A mean is the sum of the call on the same batch, divided once by the
    length of the segment as a value of the dtype: the two results must be
    equal bit for bit. An empty segment gives NaN, zero divided by zero.

    The mean also lies within `(k + 1) * eps * sum(|x|) / n` of the exactly
    rounded mean, with the `k` of the sum: the extra unit covers the
    division and the conversion of the length. The bound is checked for a
    batch whose sums are finite, because a sum that overflows gives an
    infinite mean, and with the smallest subnormal of the dtype added,
    because a division that underflows is rounded to a subnormal.
    """
    lengths = host_offsets[1:] - host_offsets[:-1]
    final = int(host_offsets[-1])
    covered = host_values[:final]
    total = _reduce("sum", host_values, host_offsets)
    _assert_same_results(actual, total / lengths.to(host_values.dtype))
    empty = lengths == 0
    assert actual[empty].isnan().all()
    assert not actual[~empty].isnan().any() or not covered.isfinite().all()

    theirs = torch.segment_reduce(
        covered.cuda(), "mean", lengths=lengths.long().cuda()
    ).cpu()
    assert torch.equal(actual.isnan(), theirs.isnan())
    if not (covered.isfinite().all() and total.isfinite().all()):
        return
    reference = _exact_means(covered, host_offsets.long())
    magnitude = _exact_sums(covered.abs(), host_offsets.long())
    eps = _EPS64 if host_values.dtype == torch.float64 else _EPS32
    depth = torch.tensor(
        [_public_depth(length) for length in lengths.tolist()],
        dtype=torch.float64,
    )
    count = lengths.clamp(min=1).double()
    kept = ~empty
    underflow = torch.finfo(host_values.dtype).tiny * eps
    error = (actual.double() - reference).abs()[kept]
    bound = ((depth + 1) * eps * magnitude / count + underflow)[kept]
    assert (error <= bound).all(), (
        f"largest error {(error / bound).nan_to_num().max()} of the mean "
        "bound"
    )
    sequential = (lengths - 1).clamp(min=0).double()
    difference = (actual.double() - theirs.double()).abs()[kept]
    assert (
        difference
        <= (
            (depth + sequential + 2) * eps * magnitude / count + 2 * underflow
        )[kept]
    ).all()


def _public_depth(length):
    """Bound the rounding additions of a public sum of one segment.

    The call runs the `mixed` schedule with the default limits and with
    automatic selection, so a segment of 4097 to 8192 elements is reduced
    by the split tree or by the CTA tree, depending on its batch.
    """
    depth = _summation_depth("mixed", length)
    if 4096 < length <= 8192:
        depth = max(depth, _summation_depth("cta", length))
    return depth


def _assert_reduction_matches(kind, host_values, host_offsets, actual):
    """Compare one public result with PyTorch and with a reference.

    The result has the dtype of the values. A maximum and a minimum are
    exact. A sum lies within `k * eps * sum(|x|)` of the reference, the
    bound of docs/internals/segmented-reductions.md with the `eps` of the
    dtype, and within that bound plus the sequential one of
    `torch.segment_reduce`, whose own tree is not documented.

    The reference of a float32 sum is the float64 sum. The reference of a
    float64 sum is the exactly rounded sum of `math.fsum`.

    A mean is compared by `_assert_mean_matches`.
    """
    if kind == "mean":
        _assert_mean_matches(host_values, host_offsets, actual)
        return
    lengths = host_offsets[1:] - host_offsets[:-1]
    final = int(host_offsets[-1])
    covered = host_values[:final]
    covered_offsets = host_offsets.long()
    theirs = torch.segment_reduce(
        covered.cuda(), kind, lengths=lengths.long().cuda()
    ).cpu()
    reference = torch.segment_reduce(
        covered.double(), kind, lengths=lengths.long()
    )
    assert actual.dtype == host_values.dtype
    assert actual.shape == theirs.shape == reference.shape
    if kind != "sum":
        assert _bits(actual) == _bits(theirs)
        assert torch.equal(actual.double(), reference)
        return
    wide = host_values.dtype == torch.float64
    eps = _EPS64 if wide else _EPS32
    if wide:
        reference = _exact_sums(covered, covered_offsets)
        magnitude = _exact_sums(covered.abs(), covered_offsets)
    else:
        magnitude = torch.segment_reduce(
            covered.double().abs(), "sum", lengths=lengths.long()
        )
    depth = torch.tensor(
        [_public_depth(length) for length in lengths.tolist()],
        dtype=torch.float64,
    )
    error = (actual.double() - reference).abs()
    assert (error <= depth * eps * magnitude).all(), (
        f"largest error {(error / (eps * magnitude)).nan_to_num().max()} "
        "eps * sum(|x|) exceeds the tree bound"
    )
    sequential = (lengths - 1).clamp(min=0).double()
    difference = (actual.double() - theirs.double()).abs()
    assert (difference <= (depth + sequential) * eps * magnitude).all()


def _reduce(kind, host_values, host_offsets):
    """Run the public reduction on fresh device tensors."""
    return swage.segment_reduce(
        host_values.cuda(), host_offsets.cuda(), kind
    ).cpu()


def _host_segments(lengths=(2, 0, 3, 1)):
    """Return small host values and offsets with exactly summable values."""
    values = torch.arange(1, sum(lengths) + 1, dtype=torch.float32)
    return values, torch.tensor(_offsets(lengths), dtype=torch.int32)


def _call(function, values, offsets, **keywords):
    """Call either public function with the arguments both take."""
    if function is swage.segment_reduce:
        return function(values, offsets, "sum", **keywords)
    return function(values, offsets, **keywords)


def _result_count(function, values, offsets):
    """Return the number of result elements a call produces."""
    if function is swage.segment_reduce:
        return offsets.numel() - 1
    return values.numel()


FUNCTIONS = [swage.segment_reduce, swage.segment_softmax]


def test_public_package_exports_exactly_the_two_segmented_calls():
    """Keep the prepared launches, the policies, and the limits private."""
    assert swage.__all__ == [
        "CompilationError",
        "jit",
        "segment_reduce",
        "segment_softmax",
    ]
    public = {name for name in vars(swage) if not name.startswith("_")}
    # The submodules that another test of the run may have imported.
    assert public - {"compile", "env", "language"} == set(swage.__all__)


@pytest.mark.parametrize("kind", ["prod", "SUM", "Mean", "", None, 0, b"sum"])
def test_segment_reduce_rejects_an_unsupported_kind(kind):
    """Admit only the kinds the kernels implement, by their exact names."""
    values, offsets = _host_segments()

    with pytest.raises(
        ValueError, match="^kind must be 'sum', 'max', 'min', or 'mean', got"
    ):
        swage.segment_reduce(values, offsets, kind)


def test_segmented_calls_take_out_by_keyword_only():
    """Leave room for later positional arguments."""
    values, offsets = _host_segments()

    with pytest.raises(TypeError, match="positional"):
        swage.segment_reduce(values, offsets, "sum", torch.empty(4))
    with pytest.raises(TypeError, match="positional"):
        swage.segment_softmax(values, offsets, torch.empty(6))


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize("name", ["values", "offsets"])
def test_segmented_calls_reject_an_input_that_is_not_a_tensor(function, name):
    """Never convert a list or an array."""
    values, offsets = _host_segments()
    arguments = {"values": values, "offsets": offsets}
    arguments[name] = arguments[name].tolist()

    with pytest.raises(TypeError, match=f"^{name} must be a torch.Tensor$"):
        _call(function, arguments["values"], arguments["offsets"])


@pytest.mark.parametrize("function", FUNCTIONS)
def test_segmented_calls_reject_values_that_require_grad(function):
    """Refuse instead of returning a result cut from the autograd graph."""
    values, offsets = _host_segments()
    values.requires_grad_()

    with pytest.raises(
        ValueError,
        match=(
            "^values must not require grad; a segmented call records no "
            r"gradient, so pass values.detach\(\)$"
        ),
    ):
        _call(function, values, offsets)

    # The remedy the message names passes this check and reaches the next.
    with pytest.raises((TypeError, RuntimeError), match="CUDA"):
        _call(function, values.detach(), offsets)


def _wrong_values(case):
    values, _ = _host_segments()
    return {
        "float64": values.double(),
        "float16": values.half(),
        "int32": values.int(),
        "rank-two": values.reshape(2, 3),
        "rank-three": values.reshape(1, 2, 3),
        "strided": torch.arange(12, dtype=torch.float32)[::2],
        "negated": torch._neg_view(values),
    }[case]


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize(
    ("case", "error", "message"),
    [
        ("strided", ValueError, "values must be contiguous"),
        ("negated", ValueError, "values must not be a lazy negation view"),
    ],
)
def test_segmented_calls_reject_values_outside_the_data_model(
    function, case, error, message
):
    """Reject other layouts instead of converting."""
    _, offsets = _host_segments()

    with pytest.raises(error, match=f"^{message}"):
        _call(function, _wrong_values(case), offsets)


@pytest.mark.parametrize("function", FUNCTIONS)
def test_each_call_names_the_ranks_of_values_it_takes(function):
    """Take rank one and rank two in both calls, and refuse rank three.

    `test_segment_columns.py` covers the rank-two calls.
    """
    _, offsets = _host_segments()

    with pytest.raises(TypeError, match="^values must have rank one or two$"):
        _call(function, _wrong_values("rank-three"), offsets)


@pytest.mark.parametrize("case", ["float16", "int32"])
def test_segmented_calls_reject_values_of_another_dtype(case):
    """Name the dtypes each call takes, and cast nothing."""
    _, offsets = _host_segments()
    values = _wrong_values(case)

    with pytest.raises(
        TypeError,
        match="^values must have dtype torch.float32 or torch.float64$",
    ):
        swage.segment_reduce(values, offsets, "sum")
    with pytest.raises(
        TypeError, match="^values must have dtype torch.float32$"
    ):
        swage.segment_softmax(values, offsets)


def test_segment_softmax_refuses_float64_values_with_the_reason():
    """Say why there is no float64 softmax instead of naming a dtype only."""
    values, offsets = _host_segments()

    with pytest.raises(
        TypeError,
        match=(
            "^values must have dtype torch.float32; segment_softmax has no "
            "float64 kernel because the device has no 64-bit exp2$"
        ),
    ):
        swage.segment_softmax(values.double(), offsets)
    with pytest.raises(TypeError, match="has no float64 kernel"):
        swage.segment_softmax(
            values.double(), offsets, out=torch.empty(6, dtype=torch.float64)
        )


def test_segment_reduce_takes_float64_values_up_to_the_device_check():
    """Admit float64 values on the host, as float32 values are."""
    values, offsets = _host_segments()

    with pytest.raises((TypeError, RuntimeError), match="CUDA"):
        swage.segment_reduce(values.double(), offsets, "sum")


def test_out_has_the_dtype_of_the_values():
    """Refuse a result tensor of the other width, in either direction."""
    values, offsets = _host_segments()

    with pytest.raises(
        TypeError, match="^out must have the dtype of values, torch.float64$"
    ):
        swage.segment_reduce(
            values.double(), offsets, "sum", out=torch.empty(4)
        )
    with pytest.raises(
        TypeError, match="^out must have the dtype of values, torch.float32$"
    ):
        swage.segment_reduce(
            values, offsets, "sum", out=torch.empty(4, dtype=torch.float64)
        )


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize(
    ("offsets", "error", "message"),
    [
        (
            torch.tensor([0, 2, 2, 5, 6], dtype=torch.int16),
            TypeError,
            "offsets must have dtype torch.int32 or torch.int64",
        ),
        (
            torch.tensor([0, 2, 2, 5, 6], dtype=torch.uint8),
            TypeError,
            "offsets must have dtype torch.int32 or torch.int64",
        ),
        (
            torch.tensor([0.0, 2.0, 6.0]),
            TypeError,
            "offsets must have dtype torch.int32 or torch.int64",
        ),
        (
            torch.tensor([[0, 6]], dtype=torch.int32),
            TypeError,
            "offsets must have rank one",
        ),
        (
            torch.tensor([0, 9, 2, 9, 6], dtype=torch.int32)[::2],
            ValueError,
            "offsets must be contiguous",
        ),
        (
            torch.tensor([], dtype=torch.int32),
            ValueError,
            "offsets must contain at least the initial zero",
        ),
        (
            torch.tensor([1, 6], dtype=torch.int32),
            ValueError,
            "offsets must start at zero",
        ),
        (
            torch.tensor([0, 4, 3, 6], dtype=torch.int32),
            ValueError,
            "offsets must be nondecreasing",
        ),
        (
            torch.tensor([0, -1, 6], dtype=torch.int32),
            ValueError,
            "offsets must not be negative",
        ),
        (
            torch.tensor([0, 2, 7], dtype=torch.int32),
            ValueError,
            "final offset 7 exceeds value count 6",
        ),
    ],
)
def test_segmented_calls_reject_offsets_outside_the_contract(
    function, offsets, error, message
):
    """Validate the offsets on the host before anything is enqueued."""
    values, _ = _host_segments()

    with pytest.raises(error, match=f"^{message}$"):
        _call(function, values, offsets)


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize(
    ("offsets", "message"),
    [
        ([[0, 6]], "offsets must have rank one"),
        ([], "offsets must contain at least the initial zero"),
        ([1, 6], "offsets must start at zero"),
        ([0, 4, 3, 6], "offsets must be nondecreasing"),
        ([0, -1, 6], "offsets must not be negative"),
        ([0, 2, 7], "final offset 7 exceeds value count 6"),
        # Narrowing these to int32 first would give [0, 3, 5] and
        # [0, 2, 6], which are valid. They are refused as int64.
        ([0, 2**32 + 3, 5], "offsets must be nondecreasing"),
        (
            [0, 2, 2**32 + 6],
            "final offset 4294967302 exceeds value count 6",
        ),
        ([0, -(2**32), 6], "offsets must not be negative"),
        ([2**32, 2**32 + 6], "offsets must start at zero"),
    ],
)
def test_int64_offsets_are_validated_before_they_are_narrowed(
    function, offsets, message
):
    """Refuse int64 offsets by their 64-bit values, in the int32 words."""
    values, _ = _host_segments()
    wide = torch.tensor(offsets, dtype=torch.int64)
    error = TypeError if "rank" in message else ValueError

    with pytest.raises(error, match=f"^{message}$"):
        _call(function, values, wide)


def test_valid_int64_offsets_pass_the_offsets_check_on_the_host():
    """Admit int64 offsets up to the device check, as int32 offsets are."""
    values, offsets = _host_segments()

    for function in FUNCTIONS:
        with pytest.raises((TypeError, RuntimeError), match="CUDA"):
            _call(function, values, offsets.long())


def test_segment_softmax_requires_offsets_that_cover_every_value():
    """Leave no element of a softmax result unwritten.

    A reduction admits offsets that end below the value count, as
    `torch.segment_reduce` does: the values past them belong to no segment.
    A softmax result has one element per value, so there the same offsets
    would return an element that no segment wrote.
    """
    values, _ = _host_segments()
    short = torch.tensor([0, 2, 5], dtype=torch.int32)

    for offsets in (short, short.long()):
        with pytest.raises(
            ValueError,
            match=(
                "^offsets must end at the value count for a softmax: final "
                "offset 5, value count 6$"
            ),
        ):
            swage.segment_softmax(values, offsets)

    # The reduction passes its offsets check and stops at the device check.
    with pytest.raises((TypeError, RuntimeError), match="CUDA"):
        swage.segment_reduce(values, short, "sum")


def _wrong_out(case, count, values, offsets):
    """Build one `out` that the contract refuses for `count` elements."""
    if case == "list":
        return [0.0] * count
    if case == "float64":
        return torch.empty(count, dtype=torch.float64)
    if case == "rank-two":
        return torch.empty(count, 1)
    if case == "longer":
        return torch.empty(count + 1)
    if case == "shorter":
        return torch.empty(count - 1)
    if case == "strided":
        return torch.empty(2 * count)[::2]
    if case == "negated":
        return torch._neg_view(torch.empty(count))
    if case == "requires-grad":
        return torch.empty(count, requires_grad=True)
    if case == "values":
        return values[:count]
    assert case == "offsets"
    return offsets.view(torch.float32)[:count]


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize(
    ("case", "error", "message"),
    [
        ("list", TypeError, "out must be a torch.Tensor or None"),
        (
            "float64",
            TypeError,
            "out must have the dtype of values, torch.float32",
        ),
        ("rank-two", TypeError, "out must have rank one"),
        ("longer", ValueError, "out must have exactly [46] elements, one per"),
        ("shorter", ValueError, "out must have exactly [46] elements, one per"),
        ("strided", ValueError, "out must be contiguous"),
        ("negated", ValueError, "out must not be a lazy negation view"),
        ("requires-grad", ValueError, "out must not require grad"),
        ("values", ValueError, "out must not overlap values"),
        ("offsets", ValueError, "out must not overlap offsets"),
    ],
)
def test_segmented_calls_reject_an_out_outside_the_contract(
    function, case, error, message
):
    """Name `out` in every error about the result tensor, and never resize."""
    # Six values in four segments, and seven offsets words to alias.
    values, _ = _host_segments()
    offsets = torch.tensor([0, 2, 2, 5, 6, 6, 6], dtype=torch.int32)[:5]
    count = _result_count(function, values, offsets)
    out = _wrong_out(case, count, values, offsets.contiguous())
    if case == "offsets":
        offsets = torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.int32)
        count = _result_count(function, values, offsets)
        out = offsets.view(torch.float32)[:count]

    with pytest.raises(error, match=f"^{message}"):
        _call(function, values, offsets, out=out)


@pytest.mark.parametrize("function", FUNCTIONS)
def test_segmented_calls_reject_host_tensors_after_validating_them(function):
    """Reach the device check last, with no conversion to the GPU."""
    values, offsets = _host_segments()
    out = torch.full((_result_count(function, values, offsets),), _SENTINEL)

    if torch.cuda.is_available():
        expected = pytest.raises(
            TypeError, match="^values must be a CUDA tensor$"
        )
    else:
        expected = pytest.raises(
            RuntimeError, match="^CUDA is unavailable in PyTorch$"
        )
    with expected:
        _call(function, values, offsets, out=out)

    assert torch.all(out == _SENTINEL)


@pytest.mark.parametrize("function", FUNCTIONS)
def test_segmented_calls_name_numpy_when_it_cannot_be_imported(
    function, monkeypatch
):
    """Raise for a missing numpy before the result is allocated."""
    values, offsets = _host_segments()
    monkeypatch.setitem(sys.modules, "numpy", None)
    monkeypatch.setattr(
        torch, "empty", lambda *a, **k: pytest.fail("a result was allocated")
    )

    with pytest.raises(
        RuntimeError,
        match=(
            f"^Swage {function.__name__}\\(\\) requires numpy, which "
            "cannot be imported; nothing was launched"
        ),
    ):
        _call(function, values, offsets)


@_needs_cuda
@pytest.mark.parametrize("function", FUNCTIONS)
def test_segmented_calls_reject_an_out_on_another_device(function):
    """Never move the result to the device of the inputs."""
    values, offsets = _host_segments()
    out = torch.empty(_result_count(function, values, offsets))

    with pytest.raises(
        ValueError,
        match="^out must be on the device of values: found cpu, values are "
        "on cuda:0$",
    ):
        _call(function, values.cuda(), offsets.cuda(), out=out)


@_needs_cuda
@pytest.mark.parametrize("function", FUNCTIONS)
def test_segmented_calls_require_the_current_device(function, monkeypatch):
    """Launch only where PyTorch's current context is."""
    values, offsets = _host_segments()
    values, offsets = values.cuda(), offsets.cuda()
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)

    with pytest.raises(
        ValueError, match="^values must be on the current CUDA device$"
    ):
        _call(function, values, offsets)


@_needs_cuda
@pytest.mark.parametrize("dtype", DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("name", DISTRIBUTIONS)
def test_segment_reduce_matches_pytorch_and_float64(name, kind, seed, dtype):
    """Reduce the shape of every benchmark distribution at a small size.

    A float64 sum is compared with the exactly rounded sum of its segment.
    """
    lengths = _distributions.generate_lengths(
        name, _SEGMENT_COUNTS[name], seed
    )
    generator = torch.Generator().manual_seed(seed)
    host_values, host_offsets = _host_case(lengths, generator, dtype)

    actual = _reduce(kind, host_values, host_offsets)

    _assert_reduction_matches(kind, host_values, host_offsets, actual)


@_needs_cuda
@pytest.mark.parametrize("kind", [*KINDS, "softmax"])
@pytest.mark.parametrize("name", DISTRIBUTIONS)
def test_int64_offsets_give_the_bits_of_int32_offsets(name, kind):
    """Return the same result whichever width the offsets are stored in.

    The kernels read a private int32 copy of int64 offsets, so both calls
    run the same kernels on the same ranges.
    """
    lengths = _distributions.generate_lengths(name, _SEGMENT_COUNTS[name], 0)
    generator = torch.Generator().manual_seed(0)
    host_values, host_offsets = _host_case(lengths, generator)
    values, offsets = host_values.cuda(), host_offsets.cuda()

    def call(offsets):
        if kind == "softmax":
            return swage.segment_softmax(values, offsets)
        return swage.segment_reduce(values, offsets, kind)

    narrow = call(offsets)
    wide = call(offsets.long())

    assert wide.dtype == narrow.dtype
    assert _bits(wide.cpu()) == _bits(narrow.cpu())


@_needs_cuda
@pytest.mark.parametrize("kind", KINDS)
def test_int64_offsets_reach_the_selected_cta_schedule(kind):
    """Take the direct-CTA selection with int64 offsets too.

    That schedule uploads no task records, so the private copy of the
    offsets is an upload of its own.
    """
    blocks = torch.cuda.get_device_properties(0).multi_processor_count
    generator = torch.Generator().manual_seed(blocks)
    lengths = torch.randint(4097, 8193, (blocks + 3,), generator=generator)
    host_values, host_offsets = _host_case(lengths.tolist(), generator)
    values, offsets = host_values.cuda(), host_offsets.cuda()

    narrow = swage.segment_reduce(values, offsets, kind)
    wide = swage.segment_reduce(values, offsets.long(), kind)

    assert _bits(wide.cpu()) == _bits(narrow.cpu())
    _assert_reduction_matches(kind, host_values, host_offsets, wide.cpu())


@_needs_cuda
@pytest.mark.parametrize("function", FUNCTIONS)
def test_int64_offsets_of_an_empty_batch_need_no_upload(function):
    """Return an empty result for int64 offsets that hold no segment."""
    values = torch.empty(0, device="cuda")
    offsets = torch.zeros(1, dtype=torch.int64, device="cuda")

    assert _call(function, values, offsets).shape == (0,)


def _launched_and_retained(monkeypatch):
    """Record every launch and every tensor a call retains on a stream.

    Returns:
        Two lists that fill during the test: the pointer arguments of each
        launch, and the `(first byte, one past the last byte, stream)` of
        each tensor that was passed to `record_stream`.
    """
    driver = _runtime._get_driver()
    launches, retained = [], []

    def recording(original):
        def launch(function, grid, block, stream, arguments):
            launches.append(arguments)
            return original(function, grid, block, stream, arguments)

        return launch

    # The fused launch goes through the task-ID launch.
    for name in ("launch_segmented", "launch_segmented_tasks"):
        monkeypatch.setattr(driver, name, recording(getattr(driver, name)))
    record_stream = torch.Tensor.record_stream

    def retaining(tensor, stream):
        begin = tensor.data_ptr()
        retained.append(
            (begin, begin + tensor.numel() * tensor.element_size(), stream)
        )
        return record_stream(tensor, stream)

    monkeypatch.setattr(torch.Tensor, "record_stream", retaining)
    return launches, retained


@_needs_cuda
@pytest.mark.parametrize(
    "case", ["reduce-records", "reduce-selected-cta", "softmax"]
)
def test_a_kernel_reads_a_retained_private_copy_of_int64_offsets(
    case, monkeypatch
):
    """Pass every kernel an int32 copy that outlives the call on its stream.

    The caller's int64 tensor is never a kernel argument. The pointer in
    the offsets slot lies in a tensor the call retained on the launch
    stream with `record_stream`, so the memory of the copy is not reused
    before the kernels ran, although the call drops the tensor when it
    returns. The result has the bits of the int32 call, so the copy holds
    the offsets.
    """
    if case == "reduce-selected-cta":
        blocks = torch.cuda.get_device_properties(0).multi_processor_count
        lengths = [4097 + index for index in range(blocks)]
    else:
        lengths = [3, 40, 0, 4100]
    host_values, host_offsets = _host_segments(lengths)
    values = host_values.cuda()
    offsets = host_offsets.long().cuda()
    function = (
        swage.segment_softmax if case == "softmax" else swage.segment_reduce
    )
    expected = _call(function, values, host_offsets.cuda()).cpu()
    torch.cuda.synchronize()
    launches, retained = _launched_and_retained(monkeypatch)
    side = torch.cuda.Stream()

    with torch.cuda.stream(side):
        result = _call(function, values, offsets)
        side.synchronize()
    # The offsets are the second pointer of every kernel that takes them.
    # A merge kernel has six arguments and no offsets, and a partial
    # kernel, the five-argument launch of a reduction, takes range records.
    partial = 5 if function is swage.segment_reduce else None
    copies = [
        arguments[1]
        for arguments in launches
        if len(arguments) not in (6, partial)
    ]

    caller = (
        offsets.data_ptr(),
        offsets.data_ptr() + offsets.numel() * offsets.element_size(),
    )
    assert copies
    for pointer in copies:
        assert not caller[0] <= pointer < caller[1]
        size = 4 * offsets.numel()
        assert any(
            begin <= pointer and pointer + size <= end
            and stream.cuda_stream == side.cuda_stream
            for begin, end, stream in retained
        )
    assert not any(
        begin < caller[1] and caller[0] < end for begin, end, _ in retained
    )
    assert _bits(result.cpu()) == _bits(expected)


@_needs_cuda
def test_int64_calls_stay_correct_while_their_memory_is_reused():
    """Reuse device memory right after each call and keep every result.

    Each call returns before its kernels ran and drops its private offsets
    copy. The loop then allocates and overwrites int32 tensors of the same
    size on the same stream, which is where the allocator would hand the
    memory of the copy out again.
    """
    segments = 257
    values = torch.randn(segments * 200 + 14_000, device="cuda")
    host_values = values.cpu()
    side = torch.cuda.Stream()
    torch.cuda.synchronize()

    with torch.cuda.stream(side):
        for seed in range(40):
            generator = torch.Generator().manual_seed(seed)
            host_offsets = torch.tensor(
                _fresh_layout(generator, segments), dtype=torch.int32
            )
            offsets = host_offsets.long().cuda()
            kind = KINDS[seed % len(KINDS)]
            result = swage.segment_reduce(values, offsets, kind)
            weights = swage.segment_softmax(
                values[: int(host_offsets[-1])], offsets
            )
            for _ in range(4):
                torch.full(
                    (segments + 1,), -7, dtype=torch.int32, device="cuda"
                )
            _assert_reduction_matches(
                kind, host_values, host_offsets, result.cpu()
            )
            _assert_softmax_matches(
                host_values[: int(host_offsets[-1])],
                host_offsets,
                weights.cpu(),
            )


def test_the_differential_batches_reach_every_schedule():
    """Keep the suite on the warp, CTA, and split paths and on empties."""
    lengths = [
        length
        for name in DISTRIBUTIONS
        for seed in range(3)
        for length in _distributions.generate_lengths(
            name, _SEGMENT_COUNTS[name], seed
        )
    ]

    assert min(lengths) == 0
    assert any(0 < length <= 32 for length in lengths)
    assert any(32 < length <= 4096 for length in lengths)
    assert any(4096 < length <= 8192 for length in lengths)
    assert max(lengths) > 8192


@_needs_cuda
@pytest.mark.parametrize("dtype", DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("length", [100_003, 1_048_577])
def test_segment_reduce_reduces_one_long_segment(length, kind, dtype):
    """Split one segment into hundreds of chunks and merge them."""
    generator = torch.Generator().manual_seed(length)
    host_values, host_offsets = _host_case([length], generator, dtype)

    actual = _reduce(kind, host_values, host_offsets)

    _assert_reduction_matches(kind, host_values, host_offsets, actual)


@_needs_cuda
@pytest.mark.parametrize("dtype", DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("length", [4097, 100_003, 1_048_577])
def test_a_split_mean_divides_the_merged_sum_once(length, dtype):
    """Divide the sum of a split segment by its length, not by its chunks.

    A segment of 4097 elements is a chunk of 4096 and a chunk of one, so a
    mean of the chunk means, or a division by the number of chunks, is far
    from the mean. The batch holds a short segment, an empty one, and two
    split ones, so the merge reads the extent of more than one segment
    from the range records. `_assert_mean_matches` requires the bits of
    the sum of the same call divided by the length.
    """
    generator = torch.Generator().manual_seed(length)
    lengths = [5, length, 0, length + 4099]
    host_values, host_offsets = _host_case(lengths, generator, dtype)

    actual = _reduce("mean", host_values, host_offsets)

    _assert_reduction_matches("mean", host_values, host_offsets, actual)
    chunks = -(-length // 4096)
    total = _reduce("sum", host_values, host_offsets)
    assert actual[1] != total[1] / chunks


@_needs_cuda
def test_a_mean_divides_by_the_length_as_a_value_of_the_dtype():
    """Divide by the float32 nearest to a length that float32 cannot hold.

    A segment of 2**24 + 1 elements has a length that rounds to 2**24 in
    float32. The kernel converts the length once, to nearest, and divides
    by that value, as a float32 division on the host does.
    """
    length = 2**24 + 1
    generator = torch.Generator().manual_seed(24)
    host_values = torch.rand(length, generator=generator)
    host_offsets = torch.tensor([0, length], dtype=torch.int32)
    divisor = torch.tensor(length, dtype=torch.float32)
    assert float(divisor) == 2.0**24

    total = _reduce("sum", host_values, host_offsets)
    mean = _reduce("mean", host_values, host_offsets)

    assert _bits(mean) == _bits(total / divisor)


@_needs_cuda
@pytest.mark.parametrize("dtype", DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("case", SPECIAL_CASES)
def test_a_mean_of_special_values_is_their_sum_divided(case, dtype):
    """Follow the sum on NaN, infinities, overflow, and subnormals.

    A mean is the sum divided by the length. A sum that overflows gives an
    infinite mean, although the exact mean of those values is finite. The
    subnormal cases pin a division that is rounded once and not flushed.
    """
    host_values = torch.cat(
        [_special_segment(case, length, dtype) for length in SPECIAL_LENGTHS]
    )
    host_offsets = torch.tensor(_offsets(SPECIAL_LENGTHS), dtype=torch.int32)
    sums = torch.tensor(
        [SPECIAL_CASES[case](length, dtype) for length in SPECIAL_LENGTHS],
        dtype=torch.float64,
    ).to(dtype)
    expected = sums / torch.tensor(SPECIAL_LENGTHS, dtype=dtype)

    actual = _reduce("mean", host_values, host_offsets)

    _assert_same_results(actual, expected)
    if case == "overflow":
        assert (actual == float("inf")).all()
    _assert_reduction_matches("mean", host_values, host_offsets, actual)


@_needs_cuda
@pytest.mark.parametrize("dtype", DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("value_count", [0, 5])
def test_segment_reduce_returns_an_empty_result_for_an_empty_batch(
    value_count, kind, dtype
):
    """Return a tensor of no elements when the offsets hold no segment."""
    values = torch.ones(value_count, dtype=dtype, device="cuda")
    offsets = torch.zeros(1, dtype=torch.int32, device="cuda")

    result = swage.segment_reduce(values, offsets, kind)

    assert result.shape == (0,)
    assert result.dtype == dtype
    assert result.device == values.device


@_needs_cuda
@pytest.mark.parametrize("dtype", DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize(
    "lengths", [[0], [0] * 300, [0, 5, 0, 0, 33, 0, 4097, 0]]
)
def test_segment_reduce_gives_empty_segments_their_identity(lengths, dtype):
    """Return 0.0 for an empty sum and an infinity for an empty extreme.

    An empty maximum is negative infinity and an empty minimum is positive
    infinity, the identity of each kind. A mean has no identity: an empty
    mean is NaN, zero divided by zero, and is compared with `isnan`,
    because the sign and the payload of that NaN are not specified.
    """
    host_values, host_offsets = _host_segments(lengths)
    host_values = host_values.to(dtype)
    empty = torch.tensor(lengths) == 0

    total = _reduce("sum", host_values, host_offsets)
    maximum = _reduce("max", host_values, host_offsets)
    minimum = _reduce("min", host_values, host_offsets)
    mean = _reduce("mean", host_values, host_offsets)

    assert _bits(total[empty]) == _bits(
        torch.zeros(int(empty.sum()), dtype=dtype)
    )
    assert torch.all(maximum[empty] == float("-inf"))
    assert torch.all(minimum[empty] == float("inf"))
    assert mean[empty].isnan().all()
    assert not mean[~empty].isnan().any()
    _assert_reduction_matches("mean", host_values, host_offsets, mean)
    _assert_reduction_matches("sum", host_values, host_offsets, total)
    _assert_reduction_matches("max", host_values, host_offsets, maximum)
    _assert_reduction_matches("min", host_values, host_offsets, minimum)


@_needs_cuda
@pytest.mark.parametrize("kind", KINDS)
def test_segment_reduce_ignores_values_past_the_final_offset(kind):
    """Values that no segment covers do not reach a result."""
    host_values = torch.tensor([1.0, 2.0, 3.0, float("nan"), float("inf")])
    host_offsets = torch.tensor([0, 1, 3], dtype=torch.int32)

    actual = _reduce(kind, host_values, host_offsets)

    expected = {
        "sum": [1.0, 5.0],
        "max": [1.0, 3.0],
        "min": [1.0, 2.0],
        "mean": [1.0, 2.5],
    }[kind]
    assert actual.tolist() == expected
    _assert_reduction_matches(kind, host_values, host_offsets, actual)


@_needs_cuda
@pytest.mark.parametrize("dtype", DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("case", SPECIAL_CASES)
def test_segment_reduce_sums_special_values_as_ieee_addition_does(case, dtype):
    """Propagate NaN and infinities and keep subnormals, as PyTorch does.

    The float64 cases use the largest finite and the smallest subnormal
    float64, so a kernel that rounded through float32 would fail them.
    """
    host_values = torch.cat(
        [_special_segment(case, length, dtype) for length in SPECIAL_LENGTHS]
    )
    host_offsets = torch.tensor(_offsets(SPECIAL_LENGTHS), dtype=torch.int32)
    exact = torch.tensor(
        [SPECIAL_CASES[case](length, dtype) for length in SPECIAL_LENGTHS],
        dtype=torch.float64,
    )
    theirs = torch.segment_reduce(
        host_values.cuda(), "sum", offsets=host_offsets.long().cuda()
    ).cpu()

    actual = _reduce("sum", host_values, host_offsets)

    if case in ("nan", "opposite-infinities"):
        assert actual.isnan().all()
        assert theirs.isnan().all()
    else:
        assert _bits(actual) == _bits(exact.to(dtype))
        assert _bits(actual) == _bits(theirs)


@_needs_cuda
@pytest.mark.parametrize("dtype", DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("length", SPECIAL_LENGTHS)
def test_segment_reduce_max_propagates_nan_and_orders_infinities(
    length, dtype
):
    """Return NaN for a segment that holds one, wherever it sits."""
    ramp = torch.arange(length, dtype=dtype)
    segments = {
        "nan-first": ramp.clone(),
        "nan-last": ramp.clone(),
        "positive-infinity": ramp.clone(),
        "all-negative-infinity": torch.full(
            (length,), float("-inf"), dtype=dtype
        ),
        "negative-infinity": ramp.clone(),
        "opposite-infinities": ramp.clone(),
    }
    segments["nan-first"][0] = float("nan")
    segments["nan-last"][-1] = float("nan")
    segments["positive-infinity"][length // 2] = float("inf")
    segments["negative-infinity"][-1] = float("-inf")
    segments["opposite-infinities"][0] = float("-inf")
    segments["opposite-infinities"][-1] = float("inf")
    host_values = torch.cat(list(segments.values()))
    host_offsets = torch.tensor(
        _offsets([length] * len(segments)), dtype=torch.int32
    )
    theirs = torch.segment_reduce(
        host_values.cuda(), "max", offsets=host_offsets.long().cuda()
    ).cpu()

    actual = _reduce("max", host_values, host_offsets)

    assert actual[:2].isnan().all()
    assert theirs[:2].isnan().all()
    assert actual[2:].tolist() == [
        float("inf"),
        float("-inf"),
        float(length - 2),
        float("inf"),
    ]
    assert _bits(actual[2:]) == _bits(theirs[2:])


@_needs_cuda
@pytest.mark.parametrize("dtype", DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("length", SPECIAL_LENGTHS)
def test_segment_reduce_min_propagates_nan_and_orders_infinities(
    length, dtype
):
    """Return NaN for a segment that holds one, wherever it sits.

    The finite values descend, so the minimum of a segment is its last
    element or the one before it, which a lane other than the one that
    stores the result reads.
    """
    ramp = torch.arange(length, 0, -1, dtype=dtype)
    segments = {
        "nan-first": ramp.clone(),
        "nan-last": ramp.clone(),
        "negative-infinity": ramp.clone(),
        "all-positive-infinity": torch.full(
            (length,), float("inf"), dtype=dtype
        ),
        "positive-infinity": ramp.clone(),
        "opposite-infinities": ramp.clone(),
    }
    segments["nan-first"][0] = float("nan")
    segments["nan-last"][-1] = float("nan")
    segments["negative-infinity"][length // 2] = float("-inf")
    segments["positive-infinity"][-1] = float("inf")
    segments["opposite-infinities"][0] = float("inf")
    segments["opposite-infinities"][-1] = float("-inf")
    host_values = torch.cat(list(segments.values()))
    host_offsets = torch.tensor(
        _offsets([length] * len(segments)), dtype=torch.int32
    )
    theirs = torch.segment_reduce(
        host_values.cuda(), "min", offsets=host_offsets.long().cuda()
    ).cpu()

    actual = _reduce("min", host_values, host_offsets)

    assert actual[:2].isnan().all()
    assert theirs[:2].isnan().all()
    assert actual[2:].tolist() == [
        float("-inf"),
        float("inf"),
        2.0,
        float("-inf"),
    ]
    assert _bits(actual[2:]) == _bits(theirs[2:])


def _signed_zero_segments(length, dtype):
    """Return segments of zeros of both signs and their IEEE extremes.

    Returns:
        A list of `(segment, maximum, minimum)`, where the maximum is +0.0
        when the segment holds +0.0 and the minimum -0.0 when it holds
        -0.0, wherever that zero sits.
    """
    negative = torch.full((length,), -0.0, dtype=dtype)
    positive = torch.zeros(length, dtype=dtype)
    last_positive, first_positive = negative.clone(), negative.clone()
    last_positive[-1] = 0.0
    first_positive[0] = 0.0
    middle_negative = positive.clone()
    middle_negative[length // 2] = -0.0
    return [
        (last_positive, 0.0, -0.0),
        (first_positive, 0.0, -0.0),
        (middle_negative, 0.0, -0.0),
        (negative, -0.0, -0.0),
        (positive, 0.0, 0.0),
    ]


@_needs_cuda
@pytest.mark.parametrize("dtype", DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("kind", ["max", "min"])
def test_segment_reduce_orders_signed_zeros_as_ieee_maximum_and_minimum(
    kind, dtype
):
    """Return +0.0 as the maximum and -0.0 as the minimum of both zeros.

    IEEE-754 maximum and minimum order -0.0 below +0.0, whatever the
    position of either zero, and so does every schedule: the lengths reach
    warp, CTA, and split work. `torch.segment_reduce` is not the reference
    here, because on PyTorch 2.12 it returns the sign of the first zero of
    a segment, on the GPU and on the CPU.
    """
    cases = [
        case
        for length in SPECIAL_LENGTHS
        for case in _signed_zero_segments(length, dtype)
    ]
    host_values = torch.cat([segment for segment, _, _ in cases])
    host_offsets = torch.tensor(
        _offsets([segment.numel() for segment, _, _ in cases]),
        dtype=torch.int32,
    )
    expected = torch.tensor(
        [extremes[kind == "min"] for _, *extremes in cases], dtype=dtype
    )

    actual = _reduce(kind, host_values, host_offsets)

    assert _bits(actual) == _bits(expected)


def _softmax(host_values, host_offsets):
    """Run the public softmax on fresh device tensors."""
    return swage.segment_softmax(host_values.cuda(), host_offsets.cuda()).cpu()


def _assert_softmax_matches(host_values, host_offsets, actual):
    """Compare a public softmax with float64 `torch.softmax` per segment.

    Every output lies within the relative bound of
    docs/internals/ragged-softmax.md, which `_softmax_bound` computes. The
    outputs are normal f32 numbers, which the bound requires.
    """
    assert actual.shape == host_values.shape
    for begin, end in pairwise(host_offsets.tolist()):
        if begin == end:
            continue
        logits = host_values[begin:end].double()
        reference = torch.softmax(logits, 0)
        assert (reference >= torch.finfo(torch.float32).tiny).all()
        relative = (actual[begin:end].double() - reference).abs() / reference
        bound = _softmax_bound(logits, reference)
        assert (relative <= bound).all(), (
            f"segment [{begin}, {end}): relative error "
            f"{relative.max().item():.3e} exceeds its bound"
        )


@_needs_cuda
@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("name", DISTRIBUTIONS)
def test_segment_softmax_matches_float64_pytorch(name, seed):
    """Normalize the shape of every benchmark distribution at a small size."""
    lengths = _distributions.generate_lengths(
        name, _SEGMENT_COUNTS[name], seed
    )
    generator = torch.Generator().manual_seed(seed)
    host_values = 4 * torch.randn(sum(lengths), generator=generator)
    host_offsets = torch.tensor(_offsets(lengths), dtype=torch.int32)

    actual = _softmax(host_values, host_offsets)

    _assert_softmax_matches(host_values, host_offsets, actual)


@_needs_cuda
def test_segment_softmax_normalizes_one_long_segment():
    """Stride one segment of many passes through a single CTA."""
    generator = torch.Generator().manual_seed(7)
    host_values = 4 * torch.randn(100_003, generator=generator)
    host_offsets = torch.tensor([0, 100_003], dtype=torch.int32)

    actual = _softmax(host_values, host_offsets)

    _assert_softmax_matches(host_values, host_offsets, actual)


@_needs_cuda
def test_segment_softmax_returns_an_empty_result_for_an_empty_batch():
    """Return a tensor of no elements when there is no value."""
    values = torch.empty(0, device="cuda")
    offsets = torch.zeros(1, dtype=torch.int32, device="cuda")

    result = swage.segment_softmax(values, offsets)

    assert result.shape == (0,)
    assert result.dtype == torch.float32
    assert result.device == values.device


@_needs_cuda
def test_segment_softmax_writes_nothing_for_empty_segments():
    """Empty segments have no result element and disturb no neighbor."""
    lengths = [0, 3, 0, 0, 130, 0]
    generator = torch.Generator().manual_seed(11)
    host_values = torch.randn(sum(lengths), generator=generator)
    host_offsets = torch.tensor(_offsets(lengths), dtype=torch.int32)
    only_empty = torch.zeros(4, dtype=torch.int32, device="cuda")

    actual = _softmax(host_values, host_offsets)
    nothing = swage.segment_softmax(torch.empty(0, device="cuda"), only_empty)

    _assert_softmax_matches(host_values, host_offsets, actual)
    assert nothing.shape == (0,)


@_needs_cuda
@pytest.mark.parametrize("length", [2, 33, 129, 4097])
def test_segment_softmax_follows_pytorch_on_special_values(length):
    """Match `torch.softmax` where a segment holds NaN or an infinity.

    A NaN, a positive infinity, or nothing but negative infinities makes
    every element of the segment NaN. A negative infinity beside a finite
    maximum gives exactly zero. Neighboring segments are unaffected.
    """
    ramp = torch.linspace(-2.0, 2.0, length)
    segments = {
        "plain": ramp.clone(),
        "nan": ramp.clone(),
        "positive-infinity": ramp.clone(),
        "all-negative-infinity": torch.full((length,), float("-inf")),
        "negative-infinity": ramp.clone(),
        "last": ramp.clone(),
    }
    segments["nan"][length // 2] = float("nan")
    segments["positive-infinity"][-1] = float("inf")
    segments["negative-infinity"][0] = float("-inf")
    host_values = torch.cat(list(segments.values()))
    host_offsets = torch.tensor(
        _offsets([length] * len(segments)), dtype=torch.int32
    )

    actual = _softmax(host_values, host_offsets).reshape(len(segments), length)

    reference = torch.softmax(
        host_values.double().reshape(len(segments), length), 1
    )
    assert torch.equal(actual.isnan(), reference.isnan())
    assert actual[1:4].isnan().all()
    assert not actual[[0, 4, 5]].isnan().any()
    assert actual[4, 0] == 0.0
    torch.testing.assert_close(
        actual[[0, 4, 5]].double(), reference[[0, 4, 5]], rtol=1e-5, atol=0
    )


@_needs_cuda
def test_segment_reduce_sum_is_the_default_mixed_schedule_of_its_batch():
    """Return the bits of the documented schedule, on every call.

    The values are ones whose f32 sum depends on the order of addition.
    The public sum equals the private `mixed` launch of a preparation with
    the default limits and automatic selection, bit for bit, and two calls
    return the same bits.
    """
    host_values, host_offsets = _order_sensitive_case()
    values, offsets = host_values.cuda(), host_offsets.cuda()
    output = torch.empty(host_offsets.numel() - 1, device="cuda")
    qualification._prepare_planned_reduction(
        values,
        offsets,
        output,
        module_text=qualification._semantic_module("sum"),
        kernel_name="segmented_sum",
    ).mixed()

    first = swage.segment_reduce(values, offsets, "sum")
    second = swage.segment_reduce(values, offsets.clone(), "sum")

    assert _bits(first.cpu()) == _bits(output.cpu())
    assert _bits(second.cpu()) == _bits(first.cpu())


def _prepared_mixed(kind, values, offsets):
    """Run the private prepared `mixed` launch at its defaults."""
    output = torch.full(
        (offsets.numel() - 1,),
        float("nan"),
        dtype=values.dtype,
        device=values.device,
    )
    element = qualification._element_of(torch, values)
    qualification._prepare_planned_reduction(
        values,
        offsets,
        output,
        module_text=qualification._semantic_module(kind, element),
        kernel_name=qualification._reduction_kernel(kind, element),
    ).mixed()
    return output


def _assert_equals_the_prepared_launch(kind, host_values, host_offsets):
    """Compare a public reduction with the prepared launch, bit for bit."""
    values, offsets = host_values.cuda(), host_offsets.cuda()
    expected = _prepared_mixed(kind, values, offsets).cpu()

    actual = swage.segment_reduce(values, offsets, kind).cpu()

    # The prepared launch poisons its output with NaN, so a NaN result is
    # one it never wrote, except for an empty mean, which is NaN.
    empty = host_offsets[1:] == host_offsets[:-1]
    written_nan = empty if kind == "mean" else torch.zeros_like(empty)
    assert torch.equal(actual.isnan(), written_nan)
    _assert_same_results(actual, expected)


@_needs_cuda
@pytest.mark.parametrize("dtype", DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("name", DISTRIBUTIONS)
def test_segment_reduce_equals_the_prepared_mixed_launch(
    name, kind, seed, dtype
):
    """Keep the bits of the path that prepares all three policies.

    The public call validates, classifies, and enqueues the mixed schedule
    in one step and prepares nothing else. For every batch of the
    differential suite it returns the bits of `mixed` of a preparation with
    the default limits and automatic selection.
    """
    lengths = _distributions.generate_lengths(
        name, _SEGMENT_COUNTS[name], seed
    )
    generator = torch.Generator().manual_seed(seed)

    _assert_equals_the_prepared_launch(
        kind, *_host_case(lengths, generator, dtype)
    )


def _selection_lengths(segments):
    """Return lengths of 4097 to 8192 elements for a batch of `segments`."""
    generator = torch.Generator().manual_seed(5)
    return torch.randint(4097, 8193, (segments,), generator=generator).tolist()


@_needs_cuda
@pytest.mark.parametrize("dtype", DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(
    "case", ["selected", "one-short-of-selection", "one-long", "empty-only"]
)
def test_segment_reduce_equals_the_prepared_launch_at_the_edges(
    case, kind, dtype
):
    """Match the prepared launch where the schedule changes.

    A batch that the selection rule sends to the pure CTA kernel, the same
    batch with one segment fewer, which is split, one segment of hundreds
    of chunks, and a batch of empty segments.
    """
    count = torch.cuda.get_device_properties(0).multi_processor_count
    lengths = {
        "selected": _selection_lengths(count),
        "one-short-of-selection": _selection_lengths(count - 1),
        "one-long": [300_001],
        "empty-only": [0] * 70,
    }[case]
    generator = torch.Generator().manual_seed(len(lengths))

    _assert_equals_the_prepared_launch(
        kind, *_host_case(lengths, generator, dtype)
    )


def _held_kernels():
    """Return what the process holds, as (compile function, options)."""
    return sorted(
        (
            compile_ptx.__name__,
            tuple(
                (name, value)
                for name, value in options
                if name in ("block_size", "use_task_ids")
            ),
        )
        for compile_ptx, _, options in qualification._ptx_memo
    )


_TASK_ID_CTA = (
    "_compile_segmented_reduction_ptx",
    (("block_size", 128), ("use_task_ids", True)),
)
_FUSED = ("_compile_fused_segmented_reduction_ptx", ())
_PARTIAL = ("_compile_split_partial_reduction_ptx", ())
_MERGE = ("_compile_split_merge_reduction_ptx", ())


@_needs_cuda
@pytest.mark.parametrize(
    ("case", "kernels"),
    [
        ("short", [_FUSED]),
        ("short-and-long", sorted([_FUSED, _PARTIAL, _MERGE])),
        ("long", sorted([_PARTIAL, _MERGE])),
        ("selected", [_TASK_ID_CTA]),
    ],
)
def test_segment_reduce_prepares_only_the_kernels_it_launches(
    case, kernels, empty_kernel_memo, driver_calls
):
    """Compile and load nothing for a policy the call does not launch.

    The pure warp kernel is never compiled. The pure CTA kernel is compiled
    only for a batch that the selection rule sends to it, which then needs
    no fused, partial, or merge kernel.
    """
    count = torch.cuda.get_device_properties(0).multi_processor_count
    lengths = {
        "short": [3, 40, 0, 4096],
        "short-and-long": [3, 40, 0, 4100],
        "long": [4100, 9000],
        "selected": _selection_lengths(count),
    }[case]
    generator = torch.Generator().manual_seed(3)
    host_values, host_offsets = _host_case(lengths, generator)

    actual = _reduce("sum", host_values, host_offsets)

    assert _held_kernels() == kernels
    assert driver_calls["cuModuleLoadData"] == len(kernels)
    _assert_reduction_matches("sum", host_values, host_offsets, actual)


@_needs_cuda
def test_segment_reduce_reads_the_shared_segment_ids_only_when_selected(
    monkeypatch,
):
    """Leave the table of segment ids alone unless the CTA kernel runs."""
    host_values, host_offsets = _host_segments([3, 40, 0, 4100])
    values, offsets = host_values.cuda(), host_offsets.cuda()
    expected = swage.segment_reduce(values, offsets, "sum").cpu()
    used = []
    identity_ids = qualification._identity_ids

    def recording(*arguments):
        used.append(arguments[2])
        return identity_ids(*arguments)

    monkeypatch.setattr(qualification, "_identity_ids", recording)

    actual = swage.segment_reduce(values, offsets, "sum").cpu()
    assert used == []
    assert _bits(actual) == _bits(expected)

    count = torch.cuda.get_device_properties(0).multi_processor_count
    lengths = _selection_lengths(count)
    selected = torch.tensor(_offsets(lengths), dtype=torch.int32).cuda()
    swage.segment_reduce(torch.ones(sum(lengths)).cuda(), selected, "sum")
    assert used == [count]


@_needs_cuda
def test_segment_reduce_sum_bits_can_change_with_the_batch():
    """Show that the public call does not pin the sum schedule.

    A batch of 8192-element segments is split into chunks until it holds as
    many segments as the device has SMs, and is reduced by the CTA schedule
    from then on. The segments that two such batches share keep their
    values and change their bits when one more segment joins. The call has
    no argument that prevents it, and both results stay inside the
    documented bound.
    """
    count = torch.cuda.get_device_properties(0).multi_processor_count
    generator = torch.Generator().manual_seed(2)
    host_values = torch.randn(count * 8192, generator=generator)

    def run(segments):
        host_offsets = torch.tensor(
            _offsets([8192] * segments), dtype=torch.int32
        )
        covered = host_values[: segments * 8192]
        actual = _reduce("sum", covered, host_offsets)
        _assert_reduction_matches("sum", covered, host_offsets, actual)
        return actual

    small = run(count - 1)
    full = run(count)
    again = run(count)

    assert _bits(small) != _bits(full[: count - 1])
    assert _bits(again) == _bits(full)


@_needs_cuda
@pytest.mark.parametrize("function", FUNCTIONS)
def test_segmented_calls_allocate_the_result_on_the_device(function):
    """Return a new f32 tensor of the documented size beside the inputs."""
    values, offsets = _host_segments()
    values, offsets = values.cuda(), offsets.cuda()

    result = _call(function, values, offsets)

    assert result.shape == (_result_count(function, values, offsets),)
    assert result.dtype == torch.float32
    assert result.device == values.device
    assert not result.requires_grad


@_needs_cuda
@pytest.mark.parametrize("kind", KINDS)
def test_a_float64_reduction_returns_float64_and_writes_a_float64_out(kind):
    """Reduce float64 values in float64, into a new tensor or into `out`.

    The values differ from 1 only below float32 precision, so a reduction
    that cast them, or that accumulated in float32, would return another
    result.
    """
    lengths = [3, 0, 40, 4100]
    host_offsets = torch.tensor(_offsets(lengths), dtype=torch.int32)
    index = torch.arange(sum(lengths), dtype=torch.float64)
    host_values = 1.0 + (index % 7 - 3) * 2.0**-40
    assert not torch.equal(host_values.float().double(), host_values)
    values, offsets = host_values.cuda(), host_offsets.cuda()
    out = torch.full((4,), _SENTINEL, dtype=torch.float64, device="cuda")

    result = swage.segment_reduce(values, offsets, kind)
    returned = swage.segment_reduce(values, offsets, kind, out=out)

    assert returned is out
    assert result.dtype == torch.float64
    assert _bits(out.cpu()) == _bits(result.cpu())
    _assert_reduction_matches(kind, host_values, host_offsets, result.cpu())
    if kind == "sum":
        # Every partial sum is a multiple of 2**-40 below 2**13, so the sum
        # is exact in float64 on every schedule.
        assert _bits(result.cpu()) == _bits(
            _exact_sums(host_values, host_offsets.long())
        )


@_needs_cuda
@pytest.mark.parametrize("function", FUNCTIONS)
def test_segmented_calls_write_out_and_return_it(function):
    """Write a caller's tensor in place and mark it as written.

    The version counter of `out` advances, as after an in-place PyTorch
    operation, and the counters of the inputs do not.
    """
    values, offsets = _host_segments()
    values, offsets = values.cuda(), offsets.cuda()
    expected = _call(function, values, offsets).cpu()
    out = torch.full_like(expected, float("nan"), device="cuda")
    versions = (out._version, values._version, offsets._version)

    result = _call(function, values, offsets, out=out)

    assert result is out
    assert _bits(out.cpu()) == _bits(expected)
    assert out._version > versions[0]
    assert (values._version, offsets._version) == versions[1:]


@_needs_cuda
@pytest.mark.parametrize("function", FUNCTIONS)
def test_segmented_calls_accept_a_contiguous_view_as_out(function):
    """Write a slice of a larger buffer and nothing around it."""
    values, offsets = _host_segments()
    values, offsets = values.cuda(), offsets.cuda()
    expected = _call(function, values, offsets).cpu()
    buffer = torch.full((expected.numel() + 8,), _SENTINEL, device="cuda")
    out = buffer[4:-4]

    _call(function, values, offsets, out=out)

    assert _bits(out.cpu()) == _bits(expected)
    assert torch.all(buffer[:4] == _SENTINEL)
    assert torch.all(buffer[-4:] == _SENTINEL)


@_needs_cuda
def test_an_overwritten_out_fails_a_backward_pass_that_saved_it():
    """Let autograd notice the write instead of using the new values."""
    values, offsets = _host_segments()
    out = torch.ones(4, device="cuda")
    weight = torch.ones(4, device="cuda", requires_grad=True)
    loss = (weight * out).sum()

    swage.segment_reduce(values.cuda(), offsets.cuda(), "sum", out=out)

    with pytest.raises(RuntimeError, match="modified by an inplace"):
        loss.backward()


@_needs_cuda
@pytest.mark.parametrize("function", FUNCTIONS)
def test_segmented_calls_take_tensors_made_under_inference_mode(function):
    """Serve inside `torch.inference_mode()` with tensors made inside it.

    The offsets, the values, and `out` are all inference tensors. A call
    enqueues what it classified before it returns, so it compares no
    version counter, which an inference tensor does not have. The result
    has no counter to advance either.
    """
    host_values, host_offsets = _host_segments([3, 40, 0, 4100])
    expected = _call(function, host_values.cuda(), host_offsets.cuda()).cpu()
    with torch.inference_mode():
        values, offsets = host_values.cuda(), host_offsets.cuda()
        out = torch.full_like(expected, float("nan"), device="cuda")
        assert all(tensor.is_inference() for tensor in (values, offsets, out))

        allocated = _call(function, values, offsets)
        written = _call(function, values, offsets, out=out)

        assert written is out
        assert allocated.is_inference()
        assert _bits(allocated.cpu()) == _bits(expected)
        assert _bits(out.cpu()) == _bits(expected)


def _compile_case(rank, offsets_dtype):
    """Return values of one rank and offsets of one dtype, on the device."""
    generator = torch.Generator().manual_seed(rank)
    shape = (1000,) if rank == 1 else (1000, 8)
    values = torch.randn(shape, generator=generator).cuda()
    offsets = torch.tensor([0, 3, 3, 500, 1000], dtype=offsets_dtype).cuda()
    return values, offsets


@_needs_cuda
@pytest.mark.parametrize(
    "offsets_dtype", [torch.int32, torch.int64], ids=["int32", "int64"]
)
@pytest.mark.parametrize("rank", [1, 2])
@pytest.mark.parametrize("backend", ["eager", "inductor"])
@pytest.mark.parametrize("function", FUNCTIONS)
def test_segmented_calls_run_eagerly_inside_torch_compile(
    function, backend, rank, offsets_dtype, monkeypatch
):
    """Leave a call out of the compiled graph and run it as an eager call.

    Dynamo does not trace into either call: the call is a graph break,
    runs eagerly between the compiled graphs around it, and returns the
    bits of the same call outside `torch.compile`. A second call with
    other offsets of the same shape follows the new offsets. The wrapper
    that keeps Dynamo out is made at the first call of a process, which
    here is inside the trace.
    """
    torch._dynamo.reset()
    monkeypatch.setattr(_segments, "_UNTRACED", {})
    values, offsets = _compile_case(rank, offsets_dtype)

    def model(values, offsets):
        return _call(function, values * 2, offsets) + 1

    compiled = torch.compile(model, backend=backend)
    actual = compiled(values, offsets)
    other = torch.tensor([0, 10, 990, 990, 1000], dtype=offsets_dtype).cuda()
    again = compiled(values, other)

    assert _bits(actual.cpu()) == _bits(model(values, offsets).cpu())
    assert _bits(again.cpu()) == _bits(model(values, other).cpu())


@_needs_cuda
@pytest.mark.parametrize("rank", [1, 2])
@pytest.mark.parametrize("function", FUNCTIONS)
def test_segmented_calls_refuse_a_full_graph(function, rank):
    """Refuse `fullgraph=True`, which forbids the graph break of a call.

    A call copies its offsets to the host and launches through the driver,
    which no graph can hold, so a function that makes the call cannot be
    compiled into one graph.
    """
    torch._dynamo.reset()
    values, offsets = _compile_case(rank, torch.int32)
    compiled = torch.compile(
        lambda values, offsets: _call(function, values, offsets) + 1,
        backend="eager",
        fullgraph=True,
    )

    with pytest.raises(torch._dynamo.exc.Unsupported):
        compiled(values, offsets)


@_needs_cuda
@pytest.mark.parametrize("function", FUNCTIONS)
def test_segmented_calls_mix_inference_and_ordinary_tensors(function):
    """Take offsets from either side of the context, inside or outside it."""
    host_values, host_offsets = _host_segments([3, 40, 0, 4100])
    values, offsets = host_values.cuda(), host_offsets.cuda()
    expected = _call(function, values, offsets).cpu()
    with torch.inference_mode():
        inside = host_offsets.cuda()
        from_outside = _call(function, values, offsets)

    from_inside = _call(function, values, inside)

    assert _bits(from_outside.cpu()) == _bits(expected)
    assert _bits(from_inside.cpu()) == _bits(expected)
    assert not from_inside.is_inference()


@_needs_cuda
@pytest.mark.parametrize("function", FUNCTIONS)
def test_segmented_calls_are_refused_under_cuda_graph_capture(function):
    """Raise before any device work and leave the capture usable.

    A call copies its offsets to the host, which a capturing stream cannot
    do. The refusal comes first, so the graph that was being captured still
    records and replays the PyTorch work around the call.
    """
    values, offsets = _host_segments()
    values, offsets = values.cuda(), offsets.cuda()
    out = torch.full(
        (_result_count(function, values, offsets),), _SENTINEL, device="cuda"
    )
    doubled = torch.zeros_like(values)
    _call(function, values, offsets)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()

    with torch.cuda.graph(graph):
        with pytest.raises(
            RuntimeError,
            match=(
                f"^{function.__name__} cannot run while the current stream "
                "captures a CUDA graph"
            ),
        ):
            _call(function, values, offsets, out=out)
        torch.mul(values, 2.0, out=doubled)

    doubled.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(doubled, 2 * values)
    assert torch.all(out == _SENTINEL)


@_needs_cuda
@pytest.mark.parametrize("function", FUNCTIONS)
def test_segmented_calls_launch_on_the_current_stream(function, monkeypatch):
    """Enqueue every kernel on the stream that is current at the call."""
    host_values, host_offsets = _host_segments([3, 40, 0, 4100])
    values, offsets = host_values.cuda(), host_offsets.cuda()
    expected = _call(function, values, offsets).cpu()
    # Inputs made on the default stream must be complete before another
    # stream reads them, as for any PyTorch operation.
    torch.cuda.synchronize()
    driver = _runtime._get_driver()
    streams = []

    def recording(original):
        def launch(function, grid, block, stream, arguments):
            streams.append(stream)
            return original(function, grid, block, stream, arguments)

        return launch

    for name in ("launch_segmented", "launch_segmented_tasks"):
        monkeypatch.setattr(driver, name, recording(getattr(driver, name)))
    side = torch.cuda.Stream()

    with torch.cuda.stream(side):
        result = _call(function, values, offsets)
    side.synchronize()

    assert streams
    assert set(streams) == {side.cuda_stream}
    assert _bits(result.cpu()) == _bits(expected)


@_needs_cuda
@pytest.mark.parametrize("function", FUNCTIONS)
def test_segmented_calls_prepare_on_every_call_and_wait_for_nothing_else(
    function,
):
    """Copy the offsets to the host once per call and synchronize nowhere.

    No layout is kept between calls: a second call with the same offsets
    tensor copies it to the host again. That copy is where a warm call
    waits for the device. The call asks for no other synchronization, also
    not after it enqueued its kernels.
    """
    host_values, host_offsets = _host_segments([3, 40, 0, 4100])
    values, offsets = host_values.cuda(), host_offsets.cuda()
    expected = _call(function, values, offsets).cpu()
    torch.cuda.synchronize()
    copied = []
    to_host = torch.Tensor.cpu

    def counting(tensor, *arguments, **keywords):
        if tensor.is_cuda:
            copied.append(tensor.data_ptr())
        return to_host(tensor, *arguments, **keywords)

    def refuse(*arguments, **keywords):
        raise AssertionError("a segmented call synchronized with the device")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torch.Tensor, "cpu", counting)
        patch.setattr(torch.Tensor, "item", refuse)
        patch.setattr(torch.Tensor, "tolist", refuse)
        patch.setattr(torch.cuda, "synchronize", refuse)
        patch.setattr(torch.cuda.Stream, "synchronize", refuse)
        patch.setattr(torch.cuda.Event, "synchronize", refuse)
        patch.setattr(torch.cuda.Event, "wait", refuse)
        first = _call(function, values, offsets)
        second = _call(function, values, offsets)

    assert copied == [offsets.data_ptr()] * 2
    assert _bits(first.cpu()) == _bits(expected)
    assert _bits(second.cpu()) == _bits(expected)


@_needs_cuda
def test_segmented_calls_run_on_a_thread_that_has_not_used_cuda():
    """Work from a new thread, whose first CUDA use is the call itself."""
    host_values, host_offsets = _host_segments([3, 40, 0, 4100])
    values, offsets = host_values.cuda(), host_offsets.cuda()
    expected = {
        function: _call(function, values, offsets).cpu()
        for function in FUNCTIONS
    }
    torch.cuda.synchronize()
    results = {}
    errors = []

    def call_both():
        try:
            for function in FUNCTIONS:
                results[function] = _call(function, values, offsets)
        except Exception as error:  # Reported by the assertion below.
            errors.append(error)

    worker = threading.Thread(target=call_both)
    worker.start()
    worker.join()

    assert errors == []
    for function in FUNCTIONS:
        assert _bits(results[function].cpu()) == _bits(expected[function])


@pytest.fixture
def empty_kernel_memo(monkeypatch):
    """Give the test a process that holds no compiled segmented kernel."""
    monkeypatch.setattr(
        qualification,
        "_ptx_memo",
        _runtime._BoundedCache(_runtime._CACHE_LIMIT),
    )
    monkeypatch.setattr(
        qualification, "_load_memo", weakref.WeakKeyDictionary()
    )


@_needs_cuda
@pytest.mark.parametrize("function", FUNCTIONS)
def test_no_compile_mode_refuses_a_call_whose_kernels_are_not_held(
    function, empty_kernel_memo, monkeypatch
):
    """Raise instead of compiling, and serve what the process compiled.

    The segmented kernels are kept in the process and never in the
    persistent cache, so a process that starts with `SWAGE_NO_COMPILE=1`
    cannot run a segmented call. Once the process holds the kernels of a
    call, the same call runs with the switch set.
    """
    host_values, host_offsets = _host_segments([3, 40, 0, 4100])
    values, offsets = host_values.cuda(), host_offsets.cuda()
    out = torch.full(
        (_result_count(function, values, offsets),), _SENTINEL, device="cuda"
    )
    monkeypatch.setenv("SWAGE_NO_COMPILE", "1")

    with pytest.raises(
        RuntimeError, match="^SWAGE_NO_COMPILE=1 refuses to compile kernel"
    ):
        _call(function, values, offsets, out=out)
    torch.cuda.synchronize()
    assert torch.all(out == _SENTINEL)

    monkeypatch.setenv("SWAGE_NO_COMPILE", "0")
    expected = _call(function, values, offsets).cpu()
    monkeypatch.setenv("SWAGE_NO_COMPILE", "1")
    _call(function, values, offsets, out=out)
    assert _bits(out.cpu()) == _bits(expected)


@_needs_cuda
def test_no_compile_mode_serves_an_empty_batch(empty_kernel_memo, monkeypatch):
    """A batch without segments needs no kernel."""
    monkeypatch.setenv("SWAGE_NO_COMPILE", "1")
    offsets = torch.zeros(1, dtype=torch.int32, device="cuda")
    values = torch.empty(0, device="cuda")

    assert swage.segment_reduce(values, offsets, "max").shape == (0,)
    assert swage.segment_softmax(values, offsets).shape == (0,)


@_needs_cuda
@pytest.mark.parametrize("function", FUNCTIONS)
def test_a_call_that_enqueues_nothing_leaves_the_out_version_alone(
    function, empty_kernel_memo, monkeypatch
):
    """Advance the counter of `out` once, and only after an enqueue.

    A call that is refused and a call on a batch without segments write
    nothing, and their `out` keeps its counter. A call that enqueues
    advances it by one.
    """
    host_values, host_offsets = _host_segments([3, 40, 0, 4100])
    values, offsets = host_values.cuda(), host_offsets.cuda()
    no_segments = torch.zeros(1, dtype=torch.int32, device="cuda")
    no_values = torch.empty(0, device="cuda")
    out = torch.full(
        (_result_count(function, values, offsets),), _SENTINEL, device="cuda"
    )
    empty = torch.empty(0, device="cuda")
    monkeypatch.setenv("SWAGE_NO_COMPILE", "1")

    with pytest.raises(RuntimeError, match="SWAGE_NO_COMPILE=1 refuses"):
        _call(function, values, offsets, out=out)
    _call(function, no_values, no_segments, out=empty)
    torch.cuda.synchronize()

    assert (out._version, empty._version) == (0, 0)
    assert torch.all(out == _SENTINEL)

    monkeypatch.setenv("SWAGE_NO_COMPILE", "0")
    _call(function, values, offsets, out=out)
    assert out._version == 1


@pytest.fixture
def driver_calls(monkeypatch):
    """Count the loads, unloads, and waits the real driver performs."""
    driver = _runtime._get_driver()
    gc.collect()
    torch.cuda.synchronize()
    driver.unload_retired()
    calls = collections.Counter()

    def counting(name, original):
        def call(*arguments):
            result = original(*arguments)
            if result == 0:
                calls[name] += 1
            return result

        return call

    for name in ("cuModuleLoadData", "cuModuleUnload", "cuCtxSynchronize"):
        monkeypatch.setattr(
            driver.library, name, counting(name, getattr(driver.library, name))
        )
    return calls


@pytest.fixture
def event_counts(monkeypatch):
    """Count the CUDA events PyTorch creates and destroys during the test.

    An event of this PyTorch runs `__del__` when it is destroyed and does
    not run weak reference callbacks, so the finalizer does the counting.
    """
    counts = collections.Counter()

    class CountedEvent(torch.cuda.Event):
        """An event that reports its creation and its destruction."""

        def __new__(cls, *arguments, **keywords):
            counts["created"] += 1
            return super().__new__(cls, *arguments, **keywords)

        def __del__(self):
            counts["destroyed"] += 1

    monkeypatch.setattr(torch.cuda, "Event", CountedEvent)
    return counts


def _fresh_layout(generator, segments):
    """Draw offsets that give a call warp, CTA, and split work to prepare."""
    lengths = torch.randint(0, 200, (segments,), generator=generator)
    lengths[generator.initial_seed() % segments] = 9000
    lengths[int(lengths.sum()) % segments] = 4097
    return _offsets(lengths.tolist())


@_needs_cuda
def test_calls_with_fresh_offsets_leave_nothing_behind(
    driver_calls, event_counts
):
    """Run many calls and keep memory, modules, and events flat.

    Every call gets offsets it has not seen, so it classifies them,
    uploads task records, and allocates split scratch. After two warming
    calls, 300 further ones must load no module, wait for the context
    never, unload nothing, compile nothing, create no CUDA event, and end
    with the device memory of the start. The cycle collector is off for
    the loop and finds nothing afterwards, so each call freed what it
    prepared when it returned, by reference counting alone.
    """
    segments, calls = 257, 300
    values = torch.randn(segments * 200 + 14_000, device="cuda")
    host_values = values.cpu()
    out = torch.empty(segments, device="cuda")

    def one_call(seed):
        generator = torch.Generator().manual_seed(seed)
        host_offsets = torch.tensor(
            _fresh_layout(generator, segments), dtype=torch.int32
        )
        offsets = host_offsets.cuda()
        kind = KINDS[seed % 2]
        swage.segment_reduce(values, offsets, kind, out=out)
        swage.segment_softmax(values[: int(host_offsets[-1])], offsets)
        if seed % 50 == 0:
            _assert_reduction_matches(
                kind, host_values, host_offsets, out.cpu()
            )

    for seed in (0, 1):
        one_call(seed)
    torch.cuda.synchronize()
    loads = driver_calls["cuModuleLoadData"]
    kernels = len(qualification._ptx_memo)
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    event_counts.clear()
    gc.collect()

    # Reading the memory statistics creates reference cycles of its own,
    # so the loop reads none and the collector runs before the next read.
    gc.disable()
    try:
        for seed in range(2, calls + 2):
            one_call(seed)
        torch.cuda.synchronize()
        unreachable = gc.collect()
    finally:
        gc.enable()

    assert unreachable == 0
    assert torch.cuda.memory_allocated() == allocated
    assert torch.cuda.memory_reserved() == reserved
    assert driver_calls["cuModuleLoadData"] == loads
    assert driver_calls["cuModuleUnload"] == 0
    assert driver_calls["cuCtxSynchronize"] == 0
    assert len(qualification._ptx_memo) == kernels
    assert event_counts == {}
